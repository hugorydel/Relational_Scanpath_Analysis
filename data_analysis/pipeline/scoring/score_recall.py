"""
pipeline/scoring/score_recall.py
==================================
Phase 2 — Automated Per-Participant Recall Scoring.

For every (SubjectID, StimID) pair, takes the participant's individual
free-text recall response and scores it against the edited codebook for
that image, producing a binary recall decision per node.

This replaces the manual scoring UI at the node level. Output is a flat
CSV with one row per (SubjectID, StimID, node_id), directly joinable with
trial_features_all.csv via SubjectID × StimID.

Output layout
-------------
  output/scoring/recall_scores.csv           — flat per-node scores (one row per SubjectID × StimID × node_id)
  output/scoring/score_results.jsonl         — successful API calls (append)
  output/scoring/score_errors.jsonl          — failed API calls (append)
  output/scoring/score_manifest.json         — run summary

Run aggregate_recall.py afterwards to produce recall_by_category.csv.

Schema of recall_scores.csv
----------------------------
  SubjectID, StimID, node_id, concept, content_type, evidence_type,
  status, recalled, matched_phrase

  recalled:        1 if the participant mentioned this concept, 0 if not
  matched_phrase:  the exact phrase from the response that matched (empty if not recalled)

Resume behaviour
----------------
Any (SubjectID, StimID) pair already in score_results.jsonl is skipped.
Run with --force to reprocess everything.

Usage
-----
    python pipeline/scoring/score_recall.py
    python pipeline/scoring/score_recall.py --dry-run
    python pipeline/scoring/score_recall.py --stim 2349475
    python pipeline/scoring/score_recall.py --subject Encode-Decode_Experiment-1-1
    python pipeline/scoring/score_recall.py --stim 2349475 --force
    python pipeline/scoring/score_recall.py --max-concurrency 5
    python pipeline/scoring/score_recall.py --model gpt-4.1
"""

import argparse
import asyncio
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
DEFAULT_CONCURRENCY = 10

FINAL_DIR = config.OUTPUT_DIR / "codebooks" / "final"
RAW_DIR = config.OUTPUT_DIR / "codebooks" / "raw"
SCORING_DIR = config.OUTPUT_DIR / "scoring"
SCORES_CSV = SCORING_DIR / "recall_scores.csv"
RESULTS_PATH = SCORING_DIR / "score_results.jsonl"
ERRORS_PATH = SCORING_DIR / "score_errors.jsonl"
MANIFEST_PATH = SCORING_DIR / "score_manifest.json"

SCORES_FIELDNAMES = [
    "SubjectID",
    "StimID",
    "node_id",
    "concept",
    "content_type",
    "evidence_type",
    "status",
    "recalled",
    "matched_phrase",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert memory scorer for a cognitive psychology study on visual memory.

You are given a participant's free-text recall description of an image, and a Codebook of verified memory concepts for that image. Your task is to score whether the participant mentioned each concept in the Codebook.

SCORING RULES:

1. RECALL DECISION
   For each node in the Codebook, decide whether the participant's response contains this concept.
   Only return nodes where the concept WAS recalled. Omit nodes that were not recalled.
   Nodes absent from your response will automatically be scored as recalled=false.

2. SEMANTIC MATCHING (not keyword matching)
   Match on meaning, not exact words. Examples:
   - "kitty" matches concept "cat"
   - "typing" in the context of "typing on keyboard" matches concept "typing"
   - "sitting on something blue" matches concept "blue" AND concept "on" (spatial)
   - "furry animal" does NOT match concept "cat" — too vague, could be any animal
   Use judgement: the match must be specific enough that a human coder would agree.

3. MATCHED PHRASE
   For each recalled node, copy the shortest phrase from the response that triggered the match.

4. ONLY SCORE CORRECT NODES
   Nodes with status "incorrect" in the Codebook are hallucinated concepts — do not return them
   even if the participant mentions them.

5. PRESERVE NODE IDs
   Use node_id values exactly as they appear in the Codebook. Do not add or invent node IDs.

Return ONLY a valid JSON array of recalled nodes. No preamble, no explanation, no markdown fences.
Each element must have exactly: node_id (string), matched_phrase (string).
If no nodes were recalled, return an empty array: []
"""

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "recall_scoring",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "matched_phrase": {"type": "string"},
                        },
                        "required": ["node_id", "matched_phrase"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_responses() -> dict:
    """
    Load all *_decoding.csv files.
    Returns {stim_id: [{"subject_id": ..., "free_response": ...}, ...]}
    Includes empty responses (so we can score them as all-zero).
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
                if stim_id:
                    by_stim[stim_id].append(
                        {
                            "subject_id": subject_id,
                            "free_response": free_response,
                        }
                    )

    total = sum(len(v) for v in by_stim.values())
    logger.info(f"Loaded {total} responses across {len(by_stim)} stimuli.")
    return dict(by_stim)


def load_codebook(stim_id: str) -> list | None:
    """
    Load the codebook for a StimID.
    Tries final/ first, falls back to raw/ if not found.
    Returns None if neither exists.
    """
    for directory in (FINAL_DIR, RAW_DIR):
        path = directory / f"{stim_id}_codebook.json"
        if path.exists():
            logger.info(f"  [{stim_id}] loading codebook from {directory.name}/")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    logger.warning(f"  [{stim_id}] no codebook found in final/ or raw/ — skipping")
    return None


def load_processed_pairs() -> set:
    """
    Return set of (subject_id, stim_id) pairs that already have at least one
    row in recall_scores.csv. This is the source of truth — the jsonl can be
    stale if a run was interrupted after logging but before writing CSV rows.
    """
    processed = set()
    if not SCORES_CSV.exists():
        return processed
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subj = row.get("SubjectID", "").strip()
            stim = row.get("StimID", "").strip()
            if subj and stim:
                processed.add((subj, stim))
    logger.info(f"Found {len(processed)} already-scored (subject, stim) pairs in recall_scores.csv.")
    return processed


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


def build_user_prompt(
    subject_id: str, stim_id: str, free_response: str, nodes: list
) -> str:
    pid = subject_id.replace("Encode-Decode_Experiment-", "P")
    # Only pass correct nodes for scoring — incorrect ones are pre-set to 0
    correct_nodes = [n for n in nodes if n.get("status") == "correct"]
    lines = [
        f"StimID: {stim_id}",
        f"Participant: {pid}",
        f"",
        f"PARTICIPANT RESPONSE:",
        free_response if free_response else "[no response]",
        f"",
        f"CODEBOOK ({len(correct_nodes)} nodes to score, "
        f"{len(nodes) - len(correct_nodes)} incorrect nodes pre-set to not-recalled):",
        json.dumps(correct_nodes, indent=2, ensure_ascii=False),
        f"",
        f"Score each node above. Return ONLY the nodes that were recalled (omit unrecalled nodes).",
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


class RecallScorer:
    """Async OpenAI client for per-participant recall scoring."""

    def __init__(self, api_key: str, model: str, max_concurrency: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.processed = 0
        self.errors = 0
        self._lock = asyncio.Lock()

    async def _call_api(
        self, subject_id: str, stim_id: str, user_prompt: str
    ) -> list | None:
        """Call API with retry. Returns list of score dicts or None."""
        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=RESPONSE_SCHEMA,
                )
                raw = response.choices[0].message.content
                parsed = json.loads(raw)
                return parsed["scores"]

            except Exception as e:
                err_str = str(e)
                is_rate = any(
                    kw in err_str.lower()
                    for kw in ("rate_limit", "429", "529", "overloaded", "too many")
                )
                if is_rate:
                    wait = delay * attempt
                    logger.warning(
                        f"  [{subject_id}×{stim_id}] rate limit — "
                        f"waiting {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait)
                elif attempt < MAX_RETRIES:
                    logger.warning(
                        f"  [{subject_id}×{stim_id}] attempt {attempt}/{MAX_RETRIES}: "
                        f"{err_str[:100]} — retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"  [{subject_id}×{stim_id}] all {MAX_RETRIES} attempts failed"
                    )
                delay *= 2

        return None

    async def score_pair(
        self,
        subject_id: str,
        stim_id: str,
        free_response: str,
        nodes: list,
        force: bool,
    ) -> dict:
        """Score one (subject, stim) pair. Returns result record."""

        # Empty response: nothing to score — fill_zeros() will handle these
        if not free_response:
            logger.info(
                f"  [{subject_id}×{stim_id}] empty response — skipping API call"
            )
            await _append_jsonl(
                RESULTS_PATH,
                {
                    "subject_id": subject_id,
                    "stim_id": stim_id,
                    "status": "ok_empty",
                    "n_nodes": len(nodes),
                    "n_recalled": 0,
                },
            )
            async with self._lock:
                self.processed += 1
            return {"subject_id": subject_id, "stim_id": stim_id, "status": "ok_empty"}

        user_prompt = build_user_prompt(subject_id, stim_id, free_response, nodes)

        async with self.semaphore:
            scores = await self._call_api(subject_id, stim_id, user_prompt)

        if scores is None:
            async with self._lock:
                self.errors += 1
            await _append_jsonl(
                ERRORS_PATH,
                {
                    "subject_id": subject_id,
                    "stim_id": stim_id,
                    "error": "API failed after retries",
                },
            )
            return {"subject_id": subject_id, "stim_id": stim_id, "status": "failed"}

        # Guard: drop any node_ids not in this codebook (model hallucination)
        valid_ids = {n["node_id"] for n in nodes if n.get("status") == "correct"}
        n_before = len(scores)
        scores = [s for s in scores if s.get("node_id") in valid_ids]
        if len(scores) < n_before:
            logger.warning(
                f"  [{subject_id}×{stim_id}] dropped {n_before - len(scores)} "
                "unknown/incorrect node_ids from API response"
            )

        rows = _build_score_rows(subject_id, stim_id, nodes, scores)
        await _write_score_rows(rows)

        n_recalled = len(rows)
        await _append_jsonl(
            RESULTS_PATH,
            {
                "subject_id": subject_id,
                "stim_id": stim_id,
                "status": "ok",
                "n_nodes": len(nodes),
                "n_recalled": n_recalled,
            },
        )

        async with self._lock:
            self.processed += 1
        logger.info(
            f"  [{subject_id}×{stim_id}] {n_recalled}/{len(valid_ids)} nodes recalled"
        )
        return {"subject_id": subject_id, "stim_id": stim_id, "status": "ok"}


# ---------------------------------------------------------------------------
# Score row construction
# ---------------------------------------------------------------------------


def _build_score_rows(
    subject_id: str, stim_id: str, nodes: list, scores: list
) -> list[dict]:
    """
    Build CSV rows for nodes recalled by this participant (recalled=1 only).
    The caller is responsible for filling in recalled=0 for absent nodes via fill_zeros().
    incorrect nodes are never included regardless of what the API returned.
    """
    # Build lookup from whatever the API returned
    matched: dict[str, str] = {}
    for s in scores or []:
        nid = s.get("node_id", "")
        if nid:
            matched[nid] = s.get("matched_phrase", "")

    # Index nodes for metadata lookup
    node_meta = {n["node_id"]: n for n in nodes}

    rows = []
    for nid, phrase in matched.items():
        node = node_meta.get(nid)
        if node is None:
            logger.warning(
                f"  [{subject_id}×{stim_id}] API returned unknown node_id {nid!r} — skipping"
            )
            continue
        if node.get("status") == "incorrect":
            # Silently drop — model shouldn't return these but guard anyway
            continue
        rows.append(
            {
                "SubjectID": subject_id,
                "StimID": stim_id,
                "node_id": nid,
                "concept": node.get("concept", ""),
                "content_type": node.get("content_type", ""),
                "evidence_type": node.get("evidence_type", ""),
                "status": node.get("status", ""),
                "recalled": 1,
                "matched_phrase": phrase,
            }
        )
    return rows


_csv_lock = asyncio.Lock()


async def _write_score_rows(rows: list[dict]) -> None:
    """Append rows to recall_scores.csv (thread-safe via lock)."""
    async with _csv_lock:
        SCORING_DIR.mkdir(parents=True, exist_ok=True)
        write_header = not SCORES_CSV.exists()
        with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCORES_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Category aggregation (post-processing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gap-fill: write recalled=0 for every (subject, stim, node_id) not in CSV
# ---------------------------------------------------------------------------


def fill_zeros(pairs: list, codebooks: dict) -> int:
    """
    After all API scoring, append recalled=0 rows for any (SubjectID, StimID, node_id)
    combination not already present in recall_scores.csv.

    pairs    : list of (subject_id, stim_id, free_response, nodes) — full scoring scope
    codebooks: {stim_id: [node, ...]} — all loaded codebooks

    Returns count of rows written.
    """
    # Read existing (subject, stim, node_id) keys from CSV
    existing: set[tuple[str, str, str]] = set()
    if SCORES_CSV.exists():
        with open(SCORES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add((row["SubjectID"], row["StimID"], row["node_id"]))

    zero_rows = []
    for subject_id, stim_id, _, nodes in pairs:
        for node in nodes:
            nid = node["node_id"]
            key = (subject_id, stim_id, nid)
            if key not in existing:
                zero_rows.append(
                    {
                        "SubjectID": subject_id,
                        "StimID": stim_id,
                        "node_id": nid,
                        "concept": node.get("concept", ""),
                        "content_type": node.get("content_type", ""),
                        "evidence_type": node.get("evidence_type", ""),
                        "status": node.get("status", ""),
                        "recalled": 0,
                        "matched_phrase": "",
                    }
                )

    if zero_rows:
        SCORING_DIR.mkdir(parents=True, exist_ok=True)
        write_header = not SCORES_CSV.exists()
        with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCORES_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(zero_rows)
        logger.info(f"fill_zeros: wrote {len(zero_rows)} recalled=0 rows")
    else:
        logger.info("fill_zeros: no gaps found")

    return len(zero_rows)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def save_manifest(results: list, model: str) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total": len(results),
        "n_ok": sum(1 for r in results if r["status"] in ("ok", "ok_empty")),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Score manifest written -> {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------


async def run(args) -> None:
    all_responses = load_responses()
    if not all_responses:
        logger.error("No responses found.")
        sys.exit(1)

    # Filter by --stim / --subject if provided
    pairs = []
    for stim_id, resp_list in sorted(all_responses.items()):
        if args.stim and stim_id != args.stim:
            continue
        nodes = load_codebook(stim_id)
        if nodes is None:
            continue
        for r in resp_list:
            if args.subject and r["subject_id"] != args.subject:
                continue
            pairs.append((r["subject_id"], stim_id, r["free_response"], nodes))

    if not pairs:
        logger.error("No (subject, stim) pairs to process after filtering.")
        sys.exit(1)

    if not args.force:
        processed = load_processed_pairs()
        pairs = [(s, st, fr, n) for s, st, fr, n in pairs if (s, st) not in processed]
        if not pairs:
            logger.info("All pairs already scored. Use --force to rerun.")
            return

    logger.info(f"Scoring {len(pairs)} (subject, stim) pairs ...\n")

    if args.dry_run:
        for subject_id, stim_id, free_response, nodes in pairs[:3]:
            print(f"\n{'='*70}")
            print(f"DRY RUN — {subject_id} × {stim_id}")
            print(f"  Response: {free_response[:120]}...")
            print(
                f"  Nodes:    {len(nodes)} ({sum(1 for n in nodes if n.get('status')=='correct')} correct)"
            )
            print(f"{'='*70}")
        if len(pairs) > 3:
            print(f"  ... and {len(pairs) - 3} more pairs")
        return

    print("\n" + "=" * 70)
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        logger.error("No API key provided.")
        sys.exit(1)
    print("API key received")
    print("=" * 70 + "\n")

    scorer = RecallScorer(
        api_key=api_key.strip(),
        model=args.model,
        max_concurrency=args.max_concurrency,
    )
    tasks = [
        scorer.score_pair(subj, stim, fr, nodes, args.force)
        for subj, stim, fr, nodes in pairs
    ]
    results = await asyncio.gather(*tasks)

    logger.info("\nFilling in recalled=0 for unscored nodes ...")
    fill_zeros(pairs, {stim_id: nodes for _, stim_id, _, nodes in pairs})

    save_manifest(list(results), args.model)

    n_ok = sum(1 for r in results if r["status"] in ("ok", "ok_empty"))
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'='*50}")
    print(f"Done.  ok={n_ok}  failed={n_failed}")
    print(f"Per-node scores -> {SCORES_CSV}")
    if n_failed:
        failed = [
            (r["subject_id"], r["stim_id"]) for r in results if r["status"] == "failed"
        ]
        print(f"Failed pairs: {failed}")
        print("Re-run (without --force) to retry only failed pairs.")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: per-participant per-node recall scoring."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without making API calls."
    )
    parser.add_argument("--stim", default=None, help="Score one StimID only.")
    parser.add_argument("--subject", default=None, help="Score one SubjectID only.")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess even if already scored."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY}).",
    )
    args = parser.parse_args()

    logger.info("Phase 2: Automated Per-Participant Recall Scoring")
    logger.info(f"  Model           : {args.model}")
    logger.info(f"  Dry run         : {args.dry_run}")
    logger.info(f"  Force           : {args.force}")
    logger.info(f"  Stim filter     : {args.stim or 'all'}")
    logger.info(f"  Subject filter  : {args.subject or 'all'}")
    logger.info(f"  Max concurrency : {args.max_concurrency}")
    logger.info(f"  Codebooks (final/): {FINAL_DIR}")
    logger.info(f"  Codebooks (raw/):   {RAW_DIR}  (fallback)")
    logger.info(f"  Output          : {SCORING_DIR}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()