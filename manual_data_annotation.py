#!/usr/bin/env python3
"""
Manual Image Annotation UI for COCO Dataset Curation
Allows manual review and categorization of filtered images.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QKeySequence
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QShortcut,
        QVBoxLayout,
        QWidget,
    )

    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("Warning: PyQt5 not installed. Falling back to matplotlib-only interface.")


class ManualAnnotationUI:
    """Manual annotation interface for image curation."""

    def __init__(
        self,
        dataset_dir: str = "./coco_naturalistic_stimuli",
        annotations_file: str = "manual_annotations.json",
    ):
        """
        Initialize the annotation UI.

        Args:
            dataset_dir: Path to processed dataset directory
            annotations_file: Name of file to store manual annotations
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annotations_path = self.dataset_dir / "annotations" / annotations_file
        self.dataset_json = self.dataset_dir / "annotations" / "dataset.json"

        # Load dataset metadata
        if not self.dataset_json.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found: {self.dataset_json}\n"
                f"Please run preprocess_data.py first."
            )

        with open(self.dataset_json, "r") as f:
            self.dataset = json.load(f)

        self.images_metadata = self.dataset["images"]
        self.categories = {cat["id"]: cat for cat in self.dataset["categories"]}

        # Load or initialize manual annotations
        self.manual_annotations = self._load_annotations()

        # Get images to annotate (not yet seen or marked "return to")
        self.images_to_annotate = self._get_images_to_annotate()
        self.current_index = 0

        # Generate colors for visualization
        self.colors = self._generate_colors(len(self.categories))

        print(f"\nManual Annotation UI Initialized")
        print(f"Total images in dataset: {len(self.images_metadata)}")
        print(f"Already annotated: {len(self.manual_annotations)}")
        print(
            f"Images to review: {len(self.images_to_annotate)} (including 'return to' items)"
        )
        print(f"\nAnnotation counts:")
        print(f"  - Selected: {self._count_status('selected')}")
        print(f"  - Eliminated: {self._count_status('eliminated')}")
        print(f"  - Return to: {self._count_status('return_to')}")

    def _generate_colors(self, n: int) -> Dict[int, tuple]:
        """Generate distinguishable colors for categories."""
        np.random.seed(42)
        colors = sns.color_palette("husl", n)
        return {
            cat_id: colors[i % len(colors)]
            for i, cat_id in enumerate(self.categories.keys())
        }

    def _load_annotations(self) -> Dict[str, Dict]:
        """Load existing manual annotations."""
        if self.annotations_path.exists():
            with open(self.annotations_path, "r") as f:
                return json.load(f)
        return {}

    def _save_annotations(self) -> None:
        """Save manual annotations to file."""
        # Ensure directory exists
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.annotations_path, "w") as f:
            json.dump(self.manual_annotations, f, indent=2)

        print(f"\nAnnotations saved: {self.annotations_path}")

    def _get_images_to_annotate(self) -> List[Dict]:
        """Get list of images that need annotation."""
        images_to_review = []

        for img_meta in self.images_metadata:
            img_id = str(img_meta["image_id"])

            # Include if not yet annotated OR marked as "return to"
            if img_id not in self.manual_annotations:
                images_to_review.append(img_meta)
            elif self.manual_annotations[img_id]["status"] == "return_to":
                images_to_review.append(img_meta)

        return images_to_review

    def _count_status(self, status: str) -> int:
        """Count images with a given status."""
        return sum(
            1 for ann in self.manual_annotations.values() if ann.get("status") == status
        )

    def _annotate_image(self, status: str) -> None:
        """
        Annotate current image with given status.

        Args:
            status: One of 'selected', 'eliminated', 'return_to'
        """
        if self.current_index >= len(self.images_to_annotate):
            print("No more images to annotate!")
            return

        current_img = self.images_to_annotate[self.current_index]
        img_id = str(current_img["image_id"])

        # Store annotation
        self.manual_annotations[img_id] = {
            "status": status,
            "file_name": current_img["file_name"],
            "num_aois": current_img["num_aois"],
            "num_categories": current_img["num_categories"],
            "coverage_percent": current_img["coverage_percent"],
            "dominant_category": current_img["dominant_category_name"],
        }

        # Save immediately
        self._save_annotations()

        # Print status
        selected = self._count_status("selected")
        eliminated = self._count_status("eliminated")
        return_to = self._count_status("return_to")
        remaining = len(self.images_to_annotate) - self.current_index - 1

        print(f"\n[{status.upper()}] Image {img_id}")
        print(
            f"Progress: Selected={selected}, Eliminated={eliminated}, "
            f"Return={return_to}, Remaining={remaining}"
        )

        # Move to next image
        self.current_index += 1

    def _draw_image_with_annotations(self, img_meta: Dict, ax: plt.Axes) -> None:
        """Draw image with segmentation overlays."""
        # Load image
        img_path = self.images_dir / img_meta["file_name"]
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            return

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image
        ax.imshow(img)

        # Draw annotations
        for ann in img_meta["annotations"]:
            cat_id = ann["category_id"]
            cat_name = self.categories[cat_id]["name"]
            color = self.colors[cat_id]

            # Draw segmentation if available
            if "segmentation" in ann:
                seg = ann["segmentation"]

                if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                    # Polygon format
                    for s in seg:
                        poly = np.array(s).reshape(-1, 2)
                        polygon = mpatches.Polygon(
                            poly,
                            closed=True,
                            linewidth=2,
                            edgecolor=color,
                            facecolor=(*color, 0.3),
                        )
                        ax.add_patch(polygon)
                else:
                    # Fall back to bbox
                    x, y, w, h = ann["bbox"]
                    rect = mpatches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=(*color, 0.3),
                    )
                    ax.add_patch(rect)
            else:
                # Draw bounding box
                x, y, w, h = ann["bbox"]
                rect = mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=(*color, 0.3),
                )
                ax.add_patch(rect)

            # Label at center
            x, y, w, h = ann["bbox"]
            center_x = x + w / 2
            center_y = y + h / 2

            ax.text(
                center_x,
                center_y,
                cat_name,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                color="white",
                fontweight="bold",
                ha="center",
                va="center",
            )

        # Get caption if available
        caption = ""
        if img_meta.get("captions"):
            caption = f"\n\"{img_meta['captions'][0]}\""

        # Title with metadata
        ax.set_title(
            f"Image {img_meta['image_id']}: {img_meta['original_file']}\n"
            f"AOIs: {img_meta['num_aois']} | Categories: {img_meta['num_categories']} | "
            f"Coverage: {img_meta['coverage_percent']:.1f}% | "
            f"Dominant: {img_meta['dominant_category_name']}"
            f"{caption}",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

        ax.axis("off")

    def run_matplotlib_interface(self) -> None:
        """Run annotation interface using matplotlib (fallback mode)."""
        if self.current_index >= len(self.images_to_annotate):
            print("\nAll images have been annotated!")
            self._print_summary()
            return

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        def show_current_image():
            """Display current image."""
            ax.clear()

            if self.current_index >= len(self.images_to_annotate):
                ax.text(
                    0.5,
                    0.5,
                    "All images annotated!\nClose window to exit.",
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                )
                ax.axis("off")
                plt.draw()
                return

            current_img = self.images_to_annotate[self.current_index]
            self._draw_image_with_annotations(current_img, ax)

            # Add instruction text
            remaining = len(self.images_to_annotate) - self.current_index
            fig.text(
                0.5,
                0.02,
                f"[S]elected | [E]liminated | [R]eturn to | [Q]uit\n"
                f"Progress: {self.current_index + 1}/{len(self.images_to_annotate)} "
                f"(Remaining: {remaining})",
                ha="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

            plt.draw()

        def on_key(event: KeyEvent):
            """Handle keyboard input."""
            if event.key == "s":
                self._annotate_image("selected")
                show_current_image()
            elif event.key == "e":
                self._annotate_image("eliminated")
                show_current_image()
            elif event.key == "r":
                self._annotate_image("return_to")
                show_current_image()
            elif event.key == "q":
                print("\nQuitting...")
                self._print_summary()
                plt.close()

        # Connect event handler
        fig.canvas.mpl_connect("key_press_event", on_key)

        # Show first image
        show_current_image()

        plt.tight_layout()
        plt.show()

    def _print_summary(self) -> None:
        """Print annotation summary."""
        print("\n" + "=" * 50)
        print("ANNOTATION SUMMARY")
        print("=" * 50)
        print(f"Total images in dataset: {len(self.images_metadata)}")
        print(f"Total annotated: {len(self.manual_annotations)}")
        print(f"  - Selected: {self._count_status('selected')}")
        print(f"  - Eliminated: {self._count_status('eliminated')}")
        print(f"  - Return to: {self._count_status('return_to')}")
        print(
            f"Not yet reviewed: {len(self.images_metadata) - len(self.manual_annotations)}"
        )
        print("=" * 50)

    def run(self) -> None:
        """Run the annotation interface."""
        if not HAS_QT:
            print("\nUsing matplotlib interface (keyboard shortcuts only)")
            print(
                "Press 'S' for Selected, 'E' for Eliminated, 'R' for Return to, 'Q' to Quit"
            )
            self.run_matplotlib_interface()
        else:
            print("\nStarting PyQt5 interface (with clickable buttons)")
            self.run_qt_interface()

    def run_qt_interface(self) -> None:
        """Run annotation interface with PyQt5 GUI."""

        class AnnotationWindow(QMainWindow):
            def __init__(self, ui_instance):
                super().__init__()
                self.ui = ui_instance
                self.init_ui()

            def init_ui(self):
                """Initialize the UI components."""
                self.setWindowTitle("Manual Image Annotation")
                self.setGeometry(100, 100, 1600, 1000)

                # Central widget
                central_widget = QWidget()
                self.setCentralWidget(central_widget)

                # Main layout
                layout = QVBoxLayout()

                # Matplotlib canvas
                self.figure = Figure(figsize=(16, 10))
                self.canvas = FigureCanvasQTAgg(self.figure)
                self.ax = self.figure.add_subplot(111)

                layout.addWidget(self.canvas)

                # Status label
                self.status_label = QLabel()
                self.status_label.setAlignment(Qt.AlignCenter)
                self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
                layout.addWidget(self.status_label)

                # Button layout
                button_layout = QHBoxLayout()

                # Selected button
                self.selected_btn = QPushButton("✓ Selected (S)")
                self.selected_btn.setStyleSheet(
                    "background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;"
                )
                self.selected_btn.clicked.connect(lambda: self.annotate("selected"))
                button_layout.addWidget(self.selected_btn)

                # Eliminated button
                self.eliminated_btn = QPushButton("✗ Eliminated (E)")
                self.eliminated_btn.setStyleSheet(
                    "background-color: #f44336; color: white; font-size: 14px; padding: 10px;"
                )
                self.eliminated_btn.clicked.connect(lambda: self.annotate("eliminated"))
                button_layout.addWidget(self.eliminated_btn)

                # Return to button
                self.return_btn = QPushButton("↻ Return To (R)")
                self.return_btn.setStyleSheet(
                    "background-color: #FF9800; color: white; font-size: 14px; padding: 10px;"
                )
                self.return_btn.clicked.connect(lambda: self.annotate("return_to"))
                button_layout.addWidget(self.return_btn)

                layout.addLayout(button_layout)

                central_widget.setLayout(layout)

                # Keyboard shortcuts
                QShortcut(QKeySequence("S"), self, lambda: self.annotate("selected"))
                QShortcut(QKeySequence("E"), self, lambda: self.annotate("eliminated"))
                QShortcut(QKeySequence("R"), self, lambda: self.annotate("return_to"))
                QShortcut(QKeySequence("Q"), self, self.close)

                # Show first image
                self.show_current_image()

            def annotate(self, status: str):
                """Handle annotation action."""
                self.ui._annotate_image(status)
                self.show_current_image()

            def show_current_image(self):
                """Display current image."""
                self.ax.clear()

                if self.ui.current_index >= len(self.ui.images_to_annotate):
                    self.ax.text(
                        0.5,
                        0.5,
                        "All images annotated!\nClose window to exit.",
                        ha="center",
                        va="center",
                        fontsize=20,
                        fontweight="bold",
                    )
                    self.ax.axis("off")
                    self.canvas.draw()

                    # Update status
                    self.status_label.setText("✓ All images annotated!")
                    self.status_label.setStyleSheet("color: green;")

                    # Disable buttons
                    self.selected_btn.setEnabled(False)
                    self.eliminated_btn.setEnabled(False)
                    self.return_btn.setEnabled(False)
                    return

                current_img = self.ui.images_to_annotate[self.ui.current_index]
                self.ui._draw_image_with_annotations(current_img, self.ax)

                self.canvas.draw()

                # Update status label
                remaining = len(self.ui.images_to_annotate) - self.ui.current_index
                selected = self.ui._count_status("selected")
                eliminated = self.ui._count_status("eliminated")
                return_to = self.ui._count_status("return_to")

                status_text = (
                    f"Progress: {self.ui.current_index + 1}/{len(self.ui.images_to_annotate)} | "
                    f"Remaining: {remaining} | "
                    f"Selected: {selected} | Eliminated: {eliminated} | Return: {return_to}"
                )
                self.status_label.setText(status_text)

            def closeEvent(self, event):
                """Handle window close."""
                self.ui._print_summary()
                event.accept()

        # Create and run application
        app = QApplication(sys.argv)
        window = AnnotationWindow(self)
        window.show()
        sys.exit(app.exec_())


def main():
    """Main entry point."""
    # Configuration
    DATASET_DIR = "./coco_naturalistic_stimuli"
    ANNOTATIONS_FILE = "manual_annotations.json"

    # Create UI
    ui = ManualAnnotationUI(
        dataset_dir=DATASET_DIR,
        annotations_file=ANNOTATIONS_FILE,
    )

    # Run interface
    ui.run()


if __name__ == "__main__":
    main()
