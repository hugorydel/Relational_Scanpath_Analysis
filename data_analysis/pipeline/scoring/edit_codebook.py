"""
pipeline/scoring/edit_codebook.py
===================================
Phase 1.5 — Automated Ground Truth Verification.

Takes the Phase 1 raw codebook JSONs and their corresponding stimulus images,
passes both to a Vision-Language Model, and produces edited codebooks with
status fields verified against actual visual evidence.

The VLM can:
  - Correct status (correct/incorrect) by checking the image directly
  - Shatter contradictory broad nodes (e.g. "person") into specific sub-nodes
  - Fix taxonomy errors (gender/age words misclassified as identity vs attribute)
  - Enforce strict atomicity on any remaining compound concepts

Output layout
-------------
  output/codebooks/edited/{StimID}_codebook.json   — verified, edited codebooks
  output/codebooks/edit_results.jsonl              — successful runs (append)
  output/codebooks/edit_errors.jsonl               — failed runs (append)
  output/codebooks/edit_manifest.json              — run summary

Resume behaviour
----------------
Any StimID already present in edit_results.jsonl is skipped. Run with --force
to reprocess.

Usage
-----
    python pipeline/scoring/edit_codebook.py
    python pipeline/scoring/edit_codebook.py --dry-run
    python pipeline/scoring/edit_codebook.py --stim 2349475
    python pipeline/scoring/edit_codebook.py --stim 2349475 --force
    python pipeline/scoring/edit_codebook.py --max-concurrency 3
    python pipeline/scoring/edit_codebook.py --model gpt-4.1
"""

import argparse
import asyncio
import base64
import getpass
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DA_DIR = _HERE.parent.parent
sys.path.insert(0, str(_DA_DIR))

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-5.4"
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0
DEFAULT_CONCURRENCY = 3  # Vision calls are heavy — keep this low

RAW_DIR = config.OUTPUT_DIR / "codebooks" / "raw"
EDITED_DIR = config.OUTPUT_DIR / "codebooks" / "edited"
IMAGES_DIR = config.DATA_METADATA_DIR / "images"
RESULTS_PATH = config.OUTPUT_DIR / "codebooks" / "edit_results.jsonl"
ERRORS_PATH = config.OUTPUT_DIR / "codebooks" / "edit_errors.jsonl"
MANIFEST_PATH = config.OUTPUT_DIR / "codebooks" / "edit_manifest.json"

VALID_CONTENT_TYPES = {
    "object_identity",
    "object_attribute",
    "action_relation",
    "spatial_relation",
    "scene_gist",
}
VALID_EVIDENCE_TYPES = {"literal", "latent", "speculative"}
VALID_STATUSES = {"correct", "incorrect"}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert visual data validator for a cognitive psychology study on visual memory.

You are given an image and a draft JSON Codebook of memory concepts extracted from participant text descriptions. Your job is to verify and edit the draft Codebook against the actual image, acting as the definitive ground truth.

CRITICAL EDITING RULES:

1. VERIFY STATUS (The "Blind AI" Fix)
   Look at the image carefully. For each node:
   - If the concept is physically present or visually deducible (latent), set status to "correct".
   - If the concept is a participant hallucination — not in the image and not reasonably inferred — set status to "incorrect".
   - Override the draft status whenever the image contradicts it. Example: if participants recalled a "glass barrier" and you can see a reflective glass facade, mark it "correct" even if the draft said "incorrect".

2. SHATTER CONTRADICTIONS (The "Person" Fix)
   Review source_phrases for broad nodes where participants described mutually exclusive things (e.g., one said "blonde man", another said "woman" for the same person node).
   - Split these into distinct nodes.
   - Create one node for the visually verified truth (e.g., concept: "woman", status: "correct").
   - Create separate nodes for the hallucinated alternatives (e.g., concept: "man", status: "incorrect").
   - Each split node gets an appended suffix on the original node_id (e.g., "2387122_002a", "2387122_002b").

3. CORRECT THE TAXONOMY
   - Standard English nouns used as gender or age modifiers (e.g., "girl", "boy", "baby", "woman", "man" when modifying another noun like "child" or "person") must be reclassified as object_attribute, not object_identity.
   - The core referent noun (e.g., "child", "person") must remain as object_identity.
   - Maintain strict atomicity: never allow a count, colour, or adjective fused to a noun in the concept field. If you find any ("two dogs", "green shirt"), split them.

4. PRESERVE IDs
   - If you keep a node unchanged, preserve its original node_id exactly.
   - If you split a node into two or more, use suffix notation: original_id + "a", "b", "c" (e.g., "2387122_002a").
   - New nodes you add that have no original counterpart should use the pattern: "{StimID}_{next_available_index}".

5. PRESERVE FIELDS
   - Every output node must have exactly these fields: node_id, concept, context, content_type, evidence_type, status, source_phrases.
   - Do not add or remove fields.
   - source_phrases: for each participant who mentioned this concept, extract the 1-2 words
     immediately surrounding the concept that confirm the match. Do NOT copy full sentences.
     Examples: concept "dog" → ["big dog", "the dog"] NOT ["I saw a big brown dog running"]
               concept "red" → ["red shirt", "red bag"] NOT ["she was wearing a red shirt"]
   - source_phrases for split or corrected nodes should contain only the phrases that map to that specific concept.

Return ONLY a valid JSON array of nodes. No preamble, no explanation, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Structured output schema (same as Phase 1)
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "codebook_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "concept": {"type": "string"},
                            "context": {"type": "string"},
                            "content_type": {
                                "type": "string",
                                "enum": sorted(VALID_CONTENT_TYPES),
                            },
                            "evidence_type": {
                                "type": "string",
                                "enum": sorted(VALID_EVIDENCE_TYPES),
                            },
                            "status": {
                                "type": "string",
                                "enum": sorted(VALID_STATUSES),
                            },
                            "source_phrases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "node_id",
                            "concept",
                            "context",
                            "content_type",
                            "evidence_type",
                            "status",
                            "source_phrases",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["nodes"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_stims(stim_filter: str | None) -> list[str]:
    """Return list of StimIDs that have both a raw codebook and an image."""
    if not RAW_DIR.exists():
        logger.error(f"Raw codebook directory not found: {RAW_DIR}")
        return []

    stims = []
    for json_path in sorted(RAW_DIR.glob("*_codebook.json")):
        stim_id = json_path.stem.replace("_codebook", "")
        img_path = _find_image(stim_id)
        if img_path is None:
            logger.warning(f"  [{stim_id}] no image found — skipping")
            continue
        if stim_filter and stim_id != stim_filter:
            continue
        stims.append(stim_id)

    return stims


def _find_image(stim_id: str) -> Path | None:
    """Try .jpg, .jpeg, .png in IMAGES_DIR."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMAGES_DIR / f"{stim_id}{ext}"
        if p.exists():
            return p
    return None



# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def encode_image(img_path: Path) -> tuple[str, str]:
    """
    Return (base64_data, media_type) for the image at img_path.
    """
    suffix = img_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(suffix, "image/jpeg")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, media_type


# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------


async def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Async VLM scorer
# ---------------------------------------------------------------------------


class CodebookEditor:
    """Async OpenAI vision client with semaphore + exponential backoff."""

    def __init__(self, api_key: str, model: str, max_concurrency: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.processed = 0
        self.errors = 0
        self._lock = asyncio.Lock()

    async def _call_vlm(
        self, stim_id: str, nodes: list, img_b64: str, media_type: str
    ) -> list | None:
        """Send image + draft codebook to VLM. Returns edited node list or None."""
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{img_b64}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": (
                    f"StimID: {stim_id}\n\n"
                    f"Below is the draft Codebook JSON for this image. "
                    f"Please verify and edit it according to your instructions.\n\n"
                    f"DRAFT CODEBOOK:\n{json.dumps(nodes, indent=2, ensure_ascii=False)}"
                ),
            },
        ]

        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=RESPONSE_SCHEMA,
                )
                raw = response.choices[0].message.content
                parsed = json.loads(raw)
                return parsed["nodes"]

            except Exception as e:
                err_str = str(e)
                is_rate = any(
                    kw in err_str.lower()
                    for kw in ("rate_limit", "429", "529", "overloaded", "too many")
                )
                if is_rate:
                    wait = delay * attempt
                    logger.warning(
                        f"  [{stim_id}] rate limit — waiting {wait:.1f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                elif attempt < MAX_RETRIES:
                    logger.warning(
                        f"  [{stim_id}] attempt {attempt}/{MAX_RETRIES}: "
                        f"{err_str[:120]} — retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"  [{stim_id}] all {MAX_RETRIES} attempts failed: "
                        f"{err_str[:200]}"
                    )
                delay *= 2

        return None

    async def process_stim(self, stim_id: str, force: bool) -> dict:
        output_path = EDITED_DIR / f"{stim_id}_codebook.json"

        if output_path.exists() and not force:
            logger.info(f"  [{stim_id}] already edited — skipping")
            return {
                "stim_id": stim_id,
                "status": "skipped",
                "n_nodes_in": None,
                "n_nodes_out": None,
                "path": str(output_path),
            }

        # Load raw codebook
        raw_path = RAW_DIR / f"{stim_id}_codebook.json"
        try:
            with open(raw_path, encoding="utf-8") as f:
                nodes = json.load(f)
        except Exception as e:
            logger.error(f"  [{stim_id}] failed to load raw codebook: {e}")
            await _append_jsonl(
                ERRORS_PATH, {"stim_id": stim_id, "error": f"load failed: {e}"}
            )
            return {
                "stim_id": stim_id,
                "status": "failed",
                "n_nodes_in": None,
                "n_nodes_out": 0,
                "path": str(output_path),
            }

        # Encode image
        img_path = _find_image(stim_id)
        try:
            img_b64, media_type = encode_image(img_path)
        except Exception as e:
            logger.error(f"  [{stim_id}] failed to encode image: {e}")
            await _append_jsonl(
                ERRORS_PATH, {"stim_id": stim_id, "error": f"image encode failed: {e}"}
            )
            return {
                "stim_id": stim_id,
                "status": "failed",
                "n_nodes_in": len(nodes),
                "n_nodes_out": 0,
                "path": str(output_path),
            }

        async with self.semaphore:
            logger.info(
                f"  [{stim_id}] calling {self.model} "
                f"({len(nodes)} nodes, image {img_path.name}) ..."
            )
            edited_nodes = await self._call_vlm(stim_id, nodes, img_b64, media_type)

        if edited_nodes is None:
            async with self._lock:
                self.errors += 1
            await _append_jsonl(
                ERRORS_PATH,
                {"stim_id": stim_id, "error": "VLM call failed after retries"},
            )
            return {
                "stim_id": stim_id,
                "status": "failed",
                "n_nodes_in": len(nodes),
                "n_nodes_out": 0,
                "path": str(output_path),
            }

        # Write edited codebook
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(edited_nodes, f, indent=2, ensure_ascii=False)

        result = {
            "stim_id": stim_id,
            "status": "ok",
            "n_nodes_in": len(nodes),
            "n_nodes_out": len(edited_nodes),
            "path": str(output_path),
        }
        await _append_jsonl(RESULTS_PATH, result)

        async with self._lock:
            self.processed += 1
        logger.info(
            f"  [{stim_id}] saved {len(edited_nodes)} nodes "
            f"(was {len(nodes)}, delta {len(edited_nodes) - len(nodes):+d})"
        )
        return result


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def save_manifest(results: list, model: str) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total": len(results),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Edit manifest written -> {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------


async def run(args) -> None:
    stims = discover_stims(args.stim)
    if not stims:
        logger.error(
            "No stims found. Check that raw codebooks exist in output/codebooks/raw/ "
            "and images exist in the configured images directory."
        )
        sys.exit(1)

    if not args.force:
        stims = [
            s for s in stims
            if not (EDITED_DIR / f"{s}_codebook.json").exists()
        ]
        if not stims:
            logger.info("All codebook files already edited on disk. Use --force to rerun.")
            return

    logger.info(f"Processing {len(stims)} stim(s) with model={args.model} ...\n")

    if args.dry_run:
        for stim_id in stims:
            raw_path = RAW_DIR / f"{stim_id}_codebook.json"
            img_path = _find_image(stim_id)
            with open(raw_path, encoding="utf-8") as f:
                nodes = json.load(f)
            print(f"\n{'='*70}")
            print(
                f"DRY RUN - StimID: {stim_id}  "
                f"({len(nodes)} nodes, image: {img_path})"
            )
            print(f"{'='*70}")
            print("--- SYSTEM PROMPT (first 400 chars) ---")
            print(SYSTEM_PROMPT[:400] + "...")
            print("\n--- DRAFT CODEBOOK (first 2 nodes) ---")
            print(json.dumps(nodes[:2], indent=2))
        return

    print("\n" + "=" * 70)
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        logger.error("No API key provided.")
        sys.exit(1)
    print("API key received")
    print("=" * 70 + "\n")

    editor = CodebookEditor(
        api_key=api_key.strip(),
        model=args.model,
        max_concurrency=args.max_concurrency,
    )
    tasks = [editor.process_stim(stim_id, args.force) for stim_id in stims]
    results = await asyncio.gather(*tasks)

    save_manifest(list(results), args.model)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_failed = sum(1 for r in results if r["status"] == "failed")

    print(f"\n{'='*50}")
    print(f"Done.  ok={n_ok}  skipped={n_skipped}  failed={n_failed}")
    if n_failed:
        failed = [r["stim_id"] for r in results if r["status"] == "failed"]
        print(f"Failed StimIDs: {failed}")
        print("Re-run (without --force) to retry only failed stims.")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.5: VLM ground truth verification of codebooks."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without making API calls."
    )
    parser.add_argument("--stim", default=None, help="Process a single StimID only.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if edited file already exists.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY}).",
    )
    args = parser.parse_args()

    logger.info("Phase 1.5: Automated Ground Truth Verification")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Dry run         : {args.dry_run}")
    logger.info(f"  Force           : {args.force}")
    logger.info(f"  Max concurrency : {args.max_concurrency}")
    logger.info(f"  Raw codebooks   : {RAW_DIR}")
    logger.info(f"  Images          : {IMAGES_DIR}")
    logger.info(f"  Output          : {EDITED_DIR}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()