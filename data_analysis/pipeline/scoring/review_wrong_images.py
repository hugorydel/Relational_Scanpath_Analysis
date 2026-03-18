"""
review_wrong_images.py
======================
UI tool to flag (SubjectID, StimID) pairs where a participant's free-text
recall response appears to describe a different image than the one shown.

Workflow
--------
1. Load recall_scores.csv — compute n_correct_recalled per (SubjectID, StimID)
2. Load decoding CSVs — get FreeResponse per (SubjectID, StimID)
3. Filter to pairs where FreeResponse is non-empty AND n_correct_recalled
   is below --threshold (default X = up to X correct nodes recalled)
4. Sort ascending by n_correct_recalled (worst offenders first)
5. Skip pairs already reviewed in wrong_image_flags.csv
6. For each pair: show stimulus image (left) + participant response (right)
   Press W = wrong image, R = correct image (both advance to next)

Output
------
  output/scoring/wrong_image_flags.csv
    Columns: SubjectID, StimID, flagged (1=wrong image, 0=correct)

Downstream
----------
  loader.py apply_exclusions() reads this file and drops flagged rows
  from enc and enc_long before any analysis.

Usage
-----
    python review_wrong_images.py
    python review_wrong_images.py --threshold 1   # also review pairs with ≤1 recall
    python review_wrong_images.py --scores path/to/recall_scores.csv
    python review_wrong_images.py --reset          # re-review all pairs (ignore existing flags)
"""

import argparse
import csv
import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

_HERE = Path(__file__).resolve().parent  # pipeline/scoring/
_ROOT = _HERE.parent.parent  # data_analysis/
sys.path.insert(0, str(_ROOT))

try:
    import config

    DEFAULT_SCORES = config.OUTPUT_DIR / "scoring" / "recall_scores.csv"
    DEFAULT_FLAGS = config.OUTPUT_DIR / "scoring" / "wrong_image_flags.csv"
    DEFAULT_BEH_DIR = config.OUTPUT_BEHAVIORAL_DIR
    DEFAULT_IMG_DIR = config.DATA_METADATA_DIR / "images"
except Exception:
    DEFAULT_SCORES = Path("output/scoring/recall_scores.csv")
    DEFAULT_FLAGS = Path("output/scoring/wrong_image_flags.csv")
    DEFAULT_BEH_DIR = Path("output/behavioral")
    DEFAULT_IMG_DIR = Path("data/metadata/images")

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_recall_counts(scores_path: Path) -> dict:
    """
    Returns {(SubjectID, StimID): n_correct_recalled}
    Only counts nodes with status=correct and recalled=1.
    """
    counts = {}
    with open(scores_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["SubjectID"], row["StimID"])
            if row.get("status") == "correct" and str(row.get("recalled")) == "1":
                counts[key] = counts.get(key, 0) + 1
            elif key not in counts:
                counts[key] = 0
    return counts


def load_free_responses(beh_dir: Path) -> dict:
    """
    Returns {(SubjectID, StimID): FreeResponse}
    Scans all *_decoding.csv files.
    """
    responses = {}
    for csv_path in sorted(beh_dir.glob("*_decoding.csv")):
        subject_id = csv_path.stem.replace("_decoding", "")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stim_id = str(row.get("StimID", "")).strip()
                text = str(row.get("FreeResponse", "")).strip()
                if stim_id and text:
                    responses[(subject_id, stim_id)] = text
    return responses


def load_existing_flags(flags_path: Path) -> set:
    """Returns set of (SubjectID, StimID) already reviewed."""
    reviewed = set()
    if not flags_path.exists():
        return reviewed
    with open(flags_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviewed.add((row["SubjectID"], row["StimID"]))
    return reviewed


def find_image(stim_id: str, img_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = img_dir / f"{stim_id}{ext}"
        if p.exists():
            return p
    return None


def build_review_queue(
    counts: dict,
    responses: dict,
    existing: set,
    threshold: int,
) -> list:
    """
    Returns list of (SubjectID, StimID, n_correct, response_text)
    sorted ascending by n_correct (worst offenders first).
    """
    queue = []
    for (subj, stim), text in responses.items():
        if (subj, stim) in existing:
            continue
        if (subj, stim) not in counts:
            continue  # not yet scored — skip entirely
        n = counts[(subj, stim)]
        if n <= threshold:
            queue.append((subj, stim, n, text))
    queue.sort(key=lambda x: x[2])
    return queue


# ---------------------------------------------------------------------------
# Flag writer
# ---------------------------------------------------------------------------


def append_flag(flags_path: Path, subject_id: str, stim_id: str, flagged: int) -> None:
    write_header = not flags_path.exists()
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flags_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["SubjectID", "StimID", "flagged"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"SubjectID": subject_id, "StimID": stim_id, "flagged": flagged}
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class ReviewApp:
    IMG_W = 700
    IMG_H = 560
    PANEL_W = 480

    def __init__(
        self,
        queue: list,
        flags_path: Path,
        img_dir: Path,
    ):
        try:
            from PIL import Image, ImageTk

            self.Image = Image
            self.ImageTk = ImageTk
        except ImportError:
            print("Pillow not found. Install with: pip install Pillow")
            sys.exit(1)

        self.queue = queue
        self.flags_path = flags_path
        self.img_dir = img_dir
        self.idx = 0
        self.total = len(queue)

        self.root = tk.Tk()
        self.root.title("Wrong Image Review")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry(f"{self.IMG_W + self.PANEL_W + 40}x{self.IMG_H + 100}")
        self.root.minsize(self.IMG_W + self.PANEL_W + 40, self.IMG_H + 100)
        self.root.resizable(True, True)

        self._build_ui()
        self._bind_keys()
        self._load_pair()

    def _build_ui(self):
        mono = tkfont.Font(family="Consolas", size=11)
        title_font = tkfont.Font(family="Consolas", size=12, weight="bold")
        small = tkfont.Font(family="Consolas", size=9)

        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#2d2d2d", pady=6)
        top.pack(fill="x")

        self.progress_var = tk.StringVar()
        tk.Label(
            top,
            textvariable=self.progress_var,
            bg="#2d2d2d",
            fg="#aaaaaa",
            font=small,
        ).pack(side="left", padx=12)

        self.pair_var = tk.StringVar()
        tk.Label(
            top,
            textvariable=self.pair_var,
            bg="#2d2d2d",
            fg="#ffffff",
            font=title_font,
        ).pack(side="left", padx=12)

        self.recall_var = tk.StringVar()
        tk.Label(
            top,
            textvariable=self.recall_var,
            bg="#2d2d2d",
            fg="#ff9944",
            font=title_font,
        ).pack(side="right", padx=12)

        # ── Main area ────────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(fill="both", expand=True, padx=0, pady=0)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)

        # Left: image
        left = tk.Frame(main, bg="#111111", width=self.IMG_W)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_propagate(False)

        self.img_label = tk.Label(left, bg="#111111")
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")

        # Right: response text + controls
        right = tk.Frame(main, bg="#1e1e1e", width=self.PANEL_W)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_propagate(False)

        tk.Label(
            right,
            text="PARTICIPANT RESPONSE",
            bg="#1e1e1e",
            fg="#888888",
            font=small,
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

        text_frame = tk.Frame(right, bg="#2a2a2a", bd=1, relief="flat")
        text_frame.pack(fill="both", expand=True)

        self.text_box = tk.Text(
            text_frame,
            wrap="word",
            bg="#2a2a2a",
            fg="#e8e8e8",
            font=mono,
            relief="flat",
            padx=10,
            pady=10,
            state="disabled",
            cursor="arrow",
        )
        self.text_box.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.text_box.yview)
        self.text_box.configure(yscrollcommand=scrollbar.set)

        # ── Bottom: key hints ────────────────────────────────────────────────
        bottom = tk.Frame(self.root, bg="#2d2d2d", pady=8)
        bottom.pack(fill="x")

        hint_font = tkfont.Font(family="Consolas", size=12, weight="bold")

        tk.Label(
            bottom,
            text="  [W]  Wrong image",
            bg="#2d2d2d",
            fg="#ff5555",
            font=hint_font,
        ).pack(side="left", padx=20)

        tk.Label(
            bottom,
            text="[R]  Correct image  ",
            bg="#2d2d2d",
            fg="#50fa7b",
            font=hint_font,
        ).pack(side="right", padx=20)

        tk.Label(
            bottom,
            text="[Q]  Quit",
            bg="#2d2d2d",
            fg="#888888",
            font=hint_font,
        ).pack(side="right", padx=20)

    def _bind_keys(self):
        self.root.bind("<KeyPress-w>", lambda e: self._record(flagged=1))
        self.root.bind("<KeyPress-W>", lambda e: self._record(flagged=1))
        self.root.bind("<KeyPress-r>", lambda e: self._record(flagged=0))
        self.root.bind("<KeyPress-R>", lambda e: self._record(flagged=0))
        self.root.bind("<KeyPress-q>", lambda e: self._quit())
        self.root.bind("<KeyPress-Q>", lambda e: self._quit())

    def _load_pair(self):
        if self.idx >= self.total:
            self._done()
            return

        subj, stim, n, text = self.queue[self.idx]

        self.progress_var.set(f"{self.idx + 1} / {self.total}")
        self.pair_var.set(f"  {subj}  ×  StimID {stim}")
        self.recall_var.set(f"correct recalls: {n}  ")

        # Load image
        img_path = find_image(stim, self.img_dir)
        if img_path:
            img = self.Image.open(img_path)
            img.thumbnail((self.IMG_W, self.IMG_H), self.Image.LANCZOS)
            self._photo = self.ImageTk.PhotoImage(img)
            self.img_label.configure(image=self._photo, text="")
        else:
            self._photo = None
            self.img_label.configure(
                image="",
                text=f"[image not found:\n{stim}]",
                fg="#ff5555",
            )

        # Load response text
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("end", text)
        self.text_box.configure(state="disabled")
        self.text_box.yview_moveto(0)

        self.root.focus_force()

    def _record(self, flagged: int):
        subj, stim, n, _ = self.queue[self.idx]
        append_flag(self.flags_path, subj, stim, flagged)
        label = "WRONG" if flagged else "correct"
        logger.info(f"  [{subj} × {stim}] marked {label}  (n_recalled={n})")
        self.idx += 1
        self._load_pair()

    def _done(self):
        messagebox.showinfo(
            "Review complete",
            f"All {self.total} pairs reviewed.\nFlags saved to:\n{self.flags_path}",
        )
        self.root.destroy()

    def _quit(self):
        if messagebox.askyesno("Quit", "Quit review? Progress so far is saved."):
            self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Review suspicious recall responses and flag wrong-image pairs."
    )
    parser.add_argument(
        "--scores",
        default=str(DEFAULT_SCORES),
        help="Path to recall_scores.csv",
    )
    parser.add_argument(
        "--behavioral",
        default=str(DEFAULT_BEH_DIR),
        help="Directory containing *_decoding.csv files",
    )
    parser.add_argument(
        "--images",
        default=str(DEFAULT_IMG_DIR),
        help="Directory containing stimulus images",
    )
    parser.add_argument(
        "--flags",
        default=str(DEFAULT_FLAGS),
        help="Path to output wrong_image_flags.csv",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Include pairs with n_correct_recalled <= this value",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing flags and re-review all qualifying pairs",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    beh_dir = Path(args.behavioral)
    img_dir = Path(args.images)
    flags_path = Path(args.flags)

    if not scores_path.exists():
        logger.error(f"recall_scores.csv not found: {scores_path}")
        sys.exit(1)

    logger.info("Loading recall counts ...")
    counts = load_recall_counts(scores_path)
    logger.info(f"  {len(counts)} (SubjectID, StimID) pairs loaded.")

    logger.info("Loading free responses ...")
    responses = load_free_responses(beh_dir)
    logger.info(f"  {len(responses)} non-empty responses loaded.")

    existing = set() if args.reset else load_existing_flags(flags_path)
    logger.info(f"  {len(existing)} pairs already reviewed — skipping.")

    queue = build_review_queue(counts, responses, existing, threshold=args.threshold)

    if not queue:
        logger.info(
            f"No pairs to review at threshold={args.threshold}. "
            "Use --threshold N to include pairs with more recalls, "
            "or --reset to re-review existing flags."
        )
        return

    logger.info(
        f"Queue: {len(queue)} pairs to review "
        f"(n_correct_recalled <= {args.threshold}, sorted worst-first)."
    )

    app = ReviewApp(queue=queue, flags_path=flags_path, img_dir=img_dir)
    app.run()

    logger.info(f"Session ended. Flags at: {flags_path}")


if __name__ == "__main__":
    main()
