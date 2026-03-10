"""
scoring_app.py
==============
Flask server for the manual memory scoring tool (Module 4 annotation step).

Reads all *_decoding.csv files from output/behavioral/, diffs against any
existing memory_scores.csv to find unscored (SubjectID, StimID) pairs, and
serves a single-page annotation UI.

Usage:
    python pipeline/scoring/scoring_app.py
    # then open http://localhost:5000 in a browser

Outputs written to output/data_scoring/:
    memory_scores.csv        — one row per SubjectID × StimID with counts
    memory_annotations.json  — one record per highlighted text span
"""

import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

# ---------------------------------------------------------------------------
# Path setup — resolve config relative to this file's location
# data_analysis/pipeline/scoring/scoring_app.py → parent.parent.parent = data_analysis/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # pipeline/scoring/
_DA_DIR = _HERE.parent.parent  # data_analysis/
sys.path.insert(0, str(_DA_DIR))

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(_HERE), static_url_path="/static")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_decoding_responses() -> list[dict]:
    """
    Scan OUTPUT_BEHAVIORAL_DIR for all *_decoding.csv files and return a list
    of {subject_id, stim_id, free_response} dicts.
    """
    rows = []
    dec_dir = config.OUTPUT_BEHAVIORAL_DIR
    if not dec_dir.exists():
        logger.warning(f"Behavioral output dir not found: {dec_dir}")
        return rows

    for csv_path in sorted(dec_dir.glob("*_decoding.csv")):
        subject_id = csv_path.stem.replace("_decoding", "")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stim_id = str(row.get("StimID", "")).strip()
                free_response = str(row.get("FreeResponse", "")).strip()
                if stim_id:
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "stim_id": stim_id,
                            "free_response": free_response,
                        }
                    )

    logger.info(f"Loaded {len(rows)} decoding responses from {dec_dir}")
    return rows


def _load_scored_pairs() -> set[tuple[str, str]]:
    """
    Return set of (subject_id, stim_id) pairs already present in
    memory_scores.csv so they can be excluded from the session queue.
    """
    scored = set()
    if not config.MEMORY_SCORES_FILE.exists():
        return scored
    with open(config.MEMORY_SCORES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scored.add((row["SubjectID"], row["StimID"]))
    logger.info(f"Found {len(scored)} already-scored pairs in memory_scores.csv")
    return scored


def _build_queue(responses: list[dict], scored: set[tuple[str, str]]) -> list[dict]:
    """
    Build the ordered scoring queue: one item per StimID, each carrying only
    the unscored (subject_id, free_response) pairs for that image.
    Images where all participants are already scored are dropped entirely.
    """
    # Group by stim_id, preserving insertion order (CSVs already sorted)
    by_stim: dict[str, list[dict]] = defaultdict(list)
    for r in responses:
        key = (r["subject_id"], r["stim_id"])
        if key not in scored:
            by_stim[r["stim_id"]].append(
                {
                    "subject_id": r["subject_id"],
                    "free_response": r["free_response"],
                }
            )

    queue = [
        {"stim_id": stim_id, "responses": resp_list}
        for stim_id, resp_list in by_stim.items()
        if resp_list  # drop if all already scored
    ]
    logger.info(f"Scoring queue: {len(queue)} images remaining")
    return queue


# Initialise on startup
_ALL_RESPONSES = _load_decoding_responses()
_SCORED_PAIRS = _load_scored_pairs()
_QUEUE = _build_queue(_ALL_RESPONSES, _SCORED_PAIRS)


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the scoring UI."""
    return send_from_directory(_HERE, "scoring_ui.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.route("/api/queue")
def api_queue():
    """Return the full remaining queue."""
    return jsonify(
        {
            "total": len(_QUEUE),
            "items": _QUEUE,
        }
    )


@app.route("/api/image/<stim_id>")
def api_image(stim_id: str):
    """Serve the stimulus image for a given stim_id."""
    img_path = config.DATA_METADATA_IMAGES_DIR / f"{stim_id}.jpg"
    if not img_path.exists():
        # Try .png fallback
        img_path = config.DATA_METADATA_IMAGES_DIR / f"{stim_id}.png"
    if not img_path.exists():
        return jsonify({"error": f"Image not found: {stim_id}"}), 404
    return send_file(str(img_path), mimetype="image/jpeg")


@app.route("/api/save", methods=["POST"])
def api_save():
    """
    Receive annotation payload for one image, persist to disk.

    Each response now carries 20 count fields (5 content types × 4 statuses):
      n_{content_type}_{status}  e.g. n_action_relation_correct
    And spans store content_type + status instead of a single category string.
    """
    global _QUEUE, _SCORED_PAIRS

    payload = request.get_json(force=True)
    stim_id = str(payload.get("stim_id", "")).strip()
    responses = payload.get("responses", [])

    if not stim_id or not responses:
        return jsonify({"error": "Missing stim_id or responses"}), 400

    config.OUTPUT_SCORING_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Build fieldnames from taxonomy ---------------------------------
    content_types = [
        "object_identity",
        "object_attribute",
        "action_relation",
        "spatial_relation",
        "scene_gist",
    ]
    statuses = ["correct", "incorrect", "inference", "repeat"]
    count_fields = [f"n_{ct}_{st}" for ct in content_types for st in statuses]

    fieldnames = (
        ["SubjectID", "StimID"] + count_fields + ["empty_response", "wrong_image"]
    )

    # ---- Write memory_scores.csv ----------------------------------------
    scores_path = config.MEMORY_SCORES_FILE
    write_header = not scores_path.exists()
    score_rows = []

    for resp in responses:
        row = {
            "SubjectID": resp["subject_id"],
            "StimID": stim_id,
            "empty_response": resp.get("empty_response", 0),
            "wrong_image": resp.get("wrong_image", 0),
        }
        for field in count_fields:
            row[field] = resp.get(field, 0)
        score_rows.append(row)

    with open(scores_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(score_rows)

    # ---- Write memory_annotations.json ----------------------------------
    ann_path = config.MEMORY_ANNOTATIONS_FILE
    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = []
    else:
        annotations = []

    for resp in responses:
        for span in resp.get("spans", []):
            annotations.append(
                {
                    "SubjectID": resp["subject_id"],
                    "StimID": stim_id,
                    "content_type": span["content_type"],
                    "status": span["status"],
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "text": span["text"],
                }
            )

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # ---- Update in-memory state -----------------------------------------
    for resp in responses:
        _SCORED_PAIRS.add((resp["subject_id"], stim_id))
    _QUEUE = _build_queue(_ALL_RESPONSES, _SCORED_PAIRS)

    logger.info(
        f"Saved {len(responses)} response(s) for StimID={stim_id}. "
        f"Queue remaining: {len(_QUEUE)}"
    )
    return jsonify({"ok": True, "remaining": len(_QUEUE)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading
    import webbrowser

    logger.info(f"Scoring app starting — {len(_QUEUE)} images in queue")
    logger.info(f"Open http://localhost:5000 in your browser")
    # Open browser after a short delay to let Flask bind the port first
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, port=5000)
