"""
pipeline/scoring/generate_codebooks.py
=======================================
Phase 1 — Automated Ontology Generation.

For every unique StimID across all participant decoding responses, aggregates
the free-text recall and sends it to an LLM to extract a deduplicated
"Master List" of memory concepts. Each concept is classified by content_type,
evidence_type, and a draft status. Output is one JSON file per StimID for
human review and correction before Phase 2 (automated scoring).

Output layout
-------------
  output/codebooks/raw/{StimID}_codebook.json   — one per image
  output/codebooks/results.jsonl                — successful runs (append)
  output/codebooks/errors.jsonl                 — failed runs (append)
  output/codebooks/manifest.json                — run summary

Resume behaviour
----------------
Any StimID already present in results.jsonl is skipped. Run again after
failures; only missing/failed images will be processed.

Usage
-----
    python pipeline/scoring/generate_codebooks.py
    python pipeline/scoring/generate_codebooks.py --dry-run          # print prompts, no API calls
    python pipeline/scoring/generate_codebooks.py --stim 2349475     # single image
    python pipeline/scoring/generate_codebooks.py --force            # reprocess even if exists
    python pipeline/scoring/generate_codebooks.py --model gpt-4.1    # override model
    python pipeline/scoring/generate_codebooks.py --max-concurrency 5
"""

import argparse
import asyncio
import base64
import csv
import getpass
import json
import logging
import sys
import time
from collections import defaultdict
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
DEFAULT_CONCURRENCY = 30

OUTPUT_DIR = config.OUTPUT_DIR / "codebooks" / "raw"
RESULTS_PATH = config.OUTPUT_DIR / "codebooks" / "results.jsonl"
ERRORS_PATH = config.OUTPUT_DIR / "codebooks" / "errors.jsonl"
MANIFEST_PATH = config.OUTPUT_DIR / "codebooks" / "manifest.json"

# Stimulus images directory — override with --image-dir at runtime
try:
    IMAGES_DIR = config.IMAGES_DIR
except AttributeError:
    IMAGES_DIR = _DA_DIR / "data" / "stimuli" / "images"

_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

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
SYSTEM_PROMPT = """You are an expert data annotator for a cognitive psychology study on visual memory.

Given a set of participant text descriptions of a single image, extract an exhaustive, deduplicated list of every distinct concept mentioned across all responses.

Rules for extraction:
- STRICT ATOMICITY: Do NOT combine distinct items into one entry. 
  - NEVER output a number and a noun together. "Two dogs" MUST be split into "two" (object_attribute) and "dog" (object_identity).
  - NEVER output an adjective and a noun together. "Green shirt" MUST be split into "green" (object_attribute) and "shirt" (object_identity).
  - The `object_identity` field must contain a noun ONLY. Adjectives and materials are forbidden in this field.
- DO merge surface variations of the same concept (e.g. "smelling", "sniffing", "smell" -> one entry).
- Extract only concepts that are genuinely memory-relevant; ignore filler phrases like "there is" or "I remember".

Classify each concept using this strict schema:

context (string):
  A brief description of what the concept modifies, relates to, or acts upon in the scene, so we know exactly how it was used. (e.g. "modifies the bag being held", "action performed by the woman", "reaching for the apple").

content_type (choose exactly one):
  object_identity  - Distinct physical entities / nouns (e.g. "woman", "dog", "surfboard")
  object_attribute - Descriptive properties: count, colour, size, material (e.g. "two", "green", "large")
  action_relation  - Dynamic interactions or motions (e.g. "running", "holding", "smelling")
  spatial_relation - Static physical positioning (e.g. "on the left", "inside", "behind")
  scene_gist       - Overarching setting or environment (e.g. "beach", "kitchen", "outdoors")

evidence_type (choose exactly one):
  literal      - Relies strictly on visible pixels; no synthesis required. Basic biology, geometry,
                 colour, physical movement. (e.g. "woman", "red", "two", "touching")
  latent       - Pragmatic shortcut describing a highly probable visual reality. Synthesises visible
                 cues into standard social or functional labels. Reasonable inference, but not
                 directly pixel-readable. (e.g. "mother and daughter" inferred from age gap and
                 proximity; "chef" from white coat; "smelling wine" from nose in glass)
  speculative  - Introduces narrative detail, internal mental states, or specific background
                 knowledge not strongly afforded by the image. (e.g. "they are tourists",
                 "she is sad", "waiting for a bus")

status (choose exactly one):
  correct   - Concept is plausibly shared memory content for this image
  incorrect - Concept appears to be a hallucination or confabulation
"""

# ---------------------------------------------------------------------------
# Structured output schema
# OpenAI requires a top-level object, so nodes are wrapped in {"nodes": [...]}
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "codebook_extraction",
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
# Data loading
# ---------------------------------------------------------------------------


def load_responses() -> dict:
    """
    Scan OUTPUT_BEHAVIORAL_DIR for all *_decoding.csv files.
    Returns {stim_id: [{"subject_id": ..., "free_response": ...}, ...]}
    Only includes rows with non-empty FreeResponse.
    """
    by_stim = defaultdict(list)
    dec_dir = config.OUTPUT_BEHAVIORAL_DIR
    if not dec_dir.exists():
        logger.error(f"Behavioral output dir not found: {dec_dir}")
        return {}

    for csv_path in sorted(dec_dir.glob("*_decoding.csv")):
        subject_id = csv_path.stem.replace("_decoding", "")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stim_id = str(row.get("StimID", "")).strip()
                free_response = str(row.get("FreeResponse", "")).strip()
                if stim_id and free_response:
                    by_stim[stim_id].append(
                        {"subject_id": subject_id, "free_response": free_response}
                    )

    n_stims = len(by_stim)
    n_total = sum(len(v) for v in by_stim.values())
    logger.info(f"Loaded {n_total} non-empty responses across {n_stims} stimuli.")
    return dict(by_stim)


def load_processed_stims() -> set:
    """Return set of StimIDs already present in results.jsonl."""
    processed = set()
    if not RESULTS_PATH.exists():
        return processed
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    processed.add(json.loads(line)["stim_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    logger.info(f"Found {len(processed)} already-processed stim(s) in results.jsonl.")
    return processed


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_b64(stim_id: str, images_dir: Path) -> tuple[str, str] | None:
    """
    Find and base64-encode the stimulus image for stim_id.
    Tries .jpg, .jpeg, .png, .webp in order.
    Returns (base64_data, media_type) or None if not found.
    """
    for ext in _IMAGE_EXTENSIONS:
        path = images_dir / f"{stim_id}{ext}"
        if path.exists():
            media_type = (
                "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
            )
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            logger.debug(f"  [{stim_id}] loaded image: {path.name}")
            return b64, media_type
    logger.warning(
        f"  [{stim_id}] no image found in {images_dir} — falling back to text-only"
    )
    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_user_prompt(stim_id: str, responses: list) -> str:
    lines = [
        f"StimID: {stim_id}",
        f"Number of participant responses: {len(responses)}",
        "",
    ]
    for r in responses:
        pid = r["subject_id"].replace("Encode-Decode_Experiment-", "P")
        lines.append(f"[{pid}] {r['free_response']}")
    lines += [
        "",
        f"Extract all distinct memory concepts from the responses above. "
        f'Use node_id prefix "{stim_id}_" with zero-padded 3-digit indices '
        f"(e.g. {stim_id}_001, {stim_id}_002, ...).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------


async def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Async scorer
# ---------------------------------------------------------------------------


class CodebookScorer:
    """Async OpenAI client with semaphore concurrency and exponential backoff."""

    def __init__(
        self, api_key: str, model: str, max_concurrency: int, images_dir: Path
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.images_dir = images_dir
        self.processed = 0
        self.errors = 0
        self._lock = asyncio.Lock()

    async def _extract(
        self, stim_id: str, user_prompt: str, image: tuple | None
    ) -> list | None:
        """Call API with retry. Returns list of nodes or None on failure."""
        # Build user message — multimodal if image is available, text-only otherwise
        if image is not None:
            b64, media_type = image
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {"type": "text", "text": user_prompt},
            ]
        else:
            user_content = user_prompt

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

    async def process_stim(self, stim_id: str, responses: list, force: bool) -> dict:
        output_path = OUTPUT_DIR / f"{stim_id}_codebook.json"

        if output_path.exists() and not force:
            logger.info(f"  [{stim_id}] already exists — skipping")
            return {
                "stim_id": stim_id,
                "status": "skipped",
                "n_nodes": None,
                "path": str(output_path),
            }

        user_prompt = build_user_prompt(stim_id, responses)
        image = _load_image_b64(stim_id, self.images_dir)

        async with self.semaphore:
            logger.info(
                f"  [{stim_id}] calling {self.model} "
                f"({'image+text' if image else 'text-only'}, {len(responses)} responses) ..."
            )
            nodes = await self._extract(stim_id, user_prompt, image)

        if nodes is None:
            async with self._lock:
                self.errors += 1
            await _append_jsonl(
                ERRORS_PATH,
                {"stim_id": stim_id, "error": "API failed after retries"},
            )
            return {
                "stim_id": stim_id,
                "status": "failed",
                "n_nodes": 0,
                "path": str(output_path),
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2, ensure_ascii=False)

        result = {
            "stim_id": stim_id,
            "status": "ok",
            "n_nodes": len(nodes),
            "path": str(output_path),
        }
        await _append_jsonl(RESULTS_PATH, result)

        async with self._lock:
            self.processed += 1
        logger.info(f"  [{stim_id}] saved {len(nodes)} nodes")
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
    logger.info(f"Manifest written -> {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------


async def run(args) -> None:
    all_responses = load_responses()
    if not all_responses:
        logger.error("No responses found. Check OUTPUT_BEHAVIORAL_DIR in config.")
        sys.exit(1)

    if args.stim:
        if args.stim not in all_responses:
            logger.error(f"StimID '{args.stim}' not found in responses.")
            sys.exit(1)
        stim_items = [(args.stim, all_responses[args.stim])]
    else:
        stim_items = sorted(all_responses.items())

    if not args.force:
        processed = load_processed_stims()
        stim_items = [(s, r) for s, r in stim_items if s not in processed]
        if not stim_items:
            logger.info("All stims already processed. Use --force to rerun.")
            return

    logger.info(f"Processing {len(stim_items)} stim(s) ...\n")

    if args.dry_run:
        for stim_id, responses in stim_items:
            print(f"\n{'='*70}")
            print(f"DRY RUN - StimID: {stim_id}  ({len(responses)} responses)")
            print(f"{'='*70}")
            print("--- SYSTEM PROMPT (first 400 chars) ---")
            print(SYSTEM_PROMPT[:400] + "...")
            print("\n--- USER PROMPT ---")
            print(build_user_prompt(stim_id, responses))
        return

    print("\n" + "=" * 70)
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        logger.error("No API key provided.")
        sys.exit(1)
    print("API key received")
    print("=" * 70 + "\n")

    scorer = CodebookScorer(
        api_key=api_key.strip(),
        model=args.model,
        max_concurrency=args.max_concurrency,
        images_dir=Path(args.image_dir),
    )
    tasks = [
        scorer.process_stim(stim_id, responses, args.force)
        for stim_id, responses in stim_items
    ]
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
        description="Phase 1: automated codebook generation via LLM."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print prompts without making API calls."
    )
    parser.add_argument("--stim", default=None, help="Process a single StimID only.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if output file already exists.",
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
    parser.add_argument(
        "--image-dir",
        default=str(IMAGES_DIR),
        help="Directory containing stimulus images (jpg/png). Passed alongside text to the VLM.",
    )
    args = parser.parse_args()

    logger.info("Phase 1: Automated Codebook Generation")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Dry run         : {args.dry_run}")
    logger.info(f"  Force           : {args.force}")
    logger.info(f"  Max concurrency : {args.max_concurrency}")
    logger.info(f"  Image dir       : {args.image_dir}")
    logger.info(f"  Output dir      : {OUTPUT_DIR}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
