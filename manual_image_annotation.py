#!/usr/bin/env python3
"""
Manual Image Annotation UI for SVG Relational Dataset Curation
Allows manual review and categorization of diversity-filtered images.
"""

import json
import shutil
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
    """Manual annotation interface for SVG relational image curation."""

    def __init__(
        self,
        dataset_dir: str = "data/diverse_scored_selection",
        output_dir: str = "data/stimuli",
        annotations_file: str = "manual_annotations.json",
    ):
        """
        Initialize the annotation UI.

        Args:
            dataset_dir: Path to diverse selection dataset directory
            output_dir: Path to output directory for final selected stimuli
            annotations_file: Name of file to store manual annotations
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annotations_path = self.dataset_dir / "annotations" / annotations_file
        self.dataset_json = self.dataset_dir / "annotations" / "selected_dataset.json"

        # Load dataset metadata
        if not self.dataset_json.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found: {self.dataset_json}\n"
                f"Please run select_diverse_scored_stimuli.py first."
            )

        with open(self.dataset_json, "r") as f:
            self.dataset = json.load(f)

        self.images_metadata = self.dataset["images"]

        # Load or initialize manual annotations
        self.manual_annotations = self._load_annotations()

        # Get images to annotate (not yet seen or marked "return to")
        self.images_to_annotate = self._get_images_to_annotate()
        self.current_index = 0

        # Track selected images for display
        self.selected_images_list = self._get_selected_images_list()

        # Generate color palette for visualization
        np.random.seed(42)
        self.colors = np.array(sns.color_palette("husl", 100))

        print(f"\nManual Annotation UI Initialized")
        print(f"Total images in diverse selection: {len(self.images_metadata)}")
        print(f"Already annotated: {len(self.manual_annotations)}")
        print(
            f"Images to review: {len(self.images_to_annotate)} (including 'return to' items)"
        )
        print(f"\nAnnotation counts:")
        print(f"  - Selected: {self._count_status('selected')}")
        print(f"  - Eliminated: {self._count_status('eliminated')}")
        print(f"  - Return to: {self._count_status('return_to')}")

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

    def _get_selected_images_list(self) -> List[str]:
        """Get list of selected image IDs for display."""
        return [
            img_id
            for img_id, ann in self.manual_annotations.items()
            if ann.get("status") == "selected"
        ]

    def _get_image_filename(self, img_meta: Dict) -> str:
        """
        Get image filename from metadata.

        Handles both old format (has 'file_name') and new format (only 'image_id').
        """
        # Try file_name field first (old format)
        if "file_name" in img_meta:
            return img_meta["file_name"]

        # Construct from image_id (new format)
        img_id = str(img_meta.get("image_id", "unknown"))
        return f"{img_id}.jpg"

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

        # Store annotation (use .get() for fields that might not exist)
        self.manual_annotations[img_id] = {
            "status": status,
            "file_name": current_img.get("file_name", f"{img_id}.jpg"),
            "story": current_img.get("story", ""),
            "CIC": current_img.get("CIC", 0),
            "SEP": current_img.get("SEP", 0),
            "DYN": current_img.get("DYN", 0),
            "QLT": current_img.get("QLT", 0),
            "score": current_img.get("score", 0),
        }

        # Save immediately
        self._save_annotations()

        # Update selected images list
        self.selected_images_list = self._get_selected_images_list()

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

        # Show selected list if status is 'selected'
        if status == "selected":
            print(f"\n✓ SELECTED IMAGES ({len(self.selected_images_list)}):")
            for sel_id in self.selected_images_list:
                print(f"  - {sel_id}")

        # Move to next image
        self.current_index += 1

    def _draw_raw_image(self, img_meta: Dict, ax: plt.Axes) -> None:
        """Draw raw image without any annotations."""
        # Load image
        filename = self._get_image_filename(img_meta)
        img_path = self.images_dir / filename
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            # Show placeholder
            ax.text(
                0.5,
                0.5,
                f"Image not found:\n{filename}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Raw Image", fontsize=12, fontweight="bold", pad=10)
            ax.axis("off")
            return

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image only
        ax.imshow(img_rgb)
        ax.set_title("Raw Image", fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    def _draw_image_with_segmentations(self, img_meta: Dict, ax: plt.Axes) -> None:
        """Draw image with segmentations overlay (no relational graph)."""
        # Load image
        filename = self._get_image_filename(img_meta)
        img_path = self.images_dir / filename
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            # Show placeholder
            ax.text(
                0.5,
                0.5,
                f"Image not found:\n{filename}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(
                "Image with Annotations", fontsize=12, fontweight="bold", pad=10
            )
            ax.axis("off")
            return

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image
        ax.imshow(img_rgb)

        # Get objects (may not exist in scored dataset)
        objects = img_meta.get("objects", [])

        if not objects:
            # No segmentation data available - just show the image with story
            story = img_meta.get("story", "")
            if story:
                # Wrap story text
                wrapped_story = "\n".join(
                    [story[i : i + 60] for i in range(0, len(story), 60)]
                )
                ax.text(
                    0.02,
                    0.98,
                    f"Story: {wrapped_story[:150]}...",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    fontsize=8,
                )
            ax.set_title(
                "Image (No segmentation data)", fontsize=12, fontweight="bold", pad=10
            )
            ax.axis("off")
            return

        # Draw segmentations/bboxes only
        for idx, obj in enumerate(objects):
            color = self.colors[idx % len(self.colors)]

            if "polygon" in obj:
                polygon = np.array(obj["polygon"]).reshape(-1, 2)
                patch = mpatches.Polygon(
                    polygon,
                    closed=True,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor=(*color, 0.2),
                )
                ax.add_patch(patch)
            elif "bbox" in obj:
                x, y, w, h = obj["bbox"]
                rect = mpatches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor=(*color, 0.2),
                )
                ax.add_patch(rect)

        # Title with metadata
        ax.set_title(
            f"Segmentations Overlay\n"
            f"Image {img_meta['image_id']} | "
            f"Objects: {img_meta['n_objects']} | Relations: {img_meta['n_relations']} | "
            f"Coverage: {img_meta['coverage_percent']:.1f}% | "
            f"Memorability: {img_meta['memorability']:.3f}",
            fontsize=11,
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

        # Create figure with 2 subplots (side-by-side)
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 10))

        def show_current_image():
            """Display current image."""
            ax_left.clear()
            ax_right.clear()

            if self.current_index >= len(self.images_to_annotate):
                ax_left.text(
                    0.5,
                    0.5,
                    "All images annotated!\nClose window to exit.",
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                )
                ax_left.axis("off")
                ax_right.axis("off")
                plt.draw()
                return

            current_img = self.images_to_annotate[self.current_index]

            # LEFT: Image with segmentations
            self._draw_image_with_segmentations(current_img, ax_left)

            # RIGHT: Raw image
            self._draw_raw_image(current_img, ax_right)

            # Add instruction text
            remaining = len(self.images_to_annotate) - self.current_index
            selected_count = len(self.selected_images_list)

            # Show selected list in instruction text
            selected_text = f"\nSelected: {selected_count} images"
            if selected_count > 0 and selected_count <= 5:
                selected_text += f" [{', '.join(self.selected_images_list)}]"
            elif selected_count > 5:
                selected_text += (
                    f" [showing first 5: {', '.join(self.selected_images_list[:5])}...]"
                )

            fig.text(
                0.5,
                0.02,
                f"[S]elected | [E]liminated | [R]eturn to | [Q]uit\n"
                f"Progress: {self.current_index + 1}/{len(self.images_to_annotate)} "
                f"(Remaining: {remaining}){selected_text}",
                ha="center",
                fontsize=11,
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
        print("\n" + "=" * 60)
        print("ANNOTATION SUMMARY")
        print("=" * 60)
        print(f"Total images in diverse selection: {len(self.images_metadata)}")
        print(f"Total annotated: {len(self.manual_annotations)}")
        print(f"  - Selected: {self._count_status('selected')}")
        print(f"  - Eliminated: {self._count_status('eliminated')}")
        print(f"  - Return to: {self._count_status('return_to')}")
        print(
            f"Not yet reviewed: {len(self.images_metadata) - len(self.manual_annotations)}"
        )
        print("=" * 60)

    def export_selected_images(self) -> None:
        """Export selected images to stimuli directory."""
        selected_count = self._count_status("selected")

        if selected_count == 0:
            print("\nNo images selected for export.")
            return

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

        # Collect selected images
        selected_images = []
        for img_id, annotation in self.manual_annotations.items():
            if annotation["status"] == "selected":
                # Find full metadata
                img_meta = next(
                    (
                        img
                        for img in self.images_metadata
                        if str(img["image_id"]) == img_id
                    ),
                    None,
                )
                if img_meta:
                    selected_images.append(img_meta)

        # Copy images and visualizations
        print(
            f"\nExporting {len(selected_images)} selected images to {self.output_dir}..."
        )

        for img_meta in selected_images:
            img_id = img_meta["image_id"]

            # Copy image
            src_img = self.images_dir / img_meta["file_name"]
            dst_img = self.output_dir / "images" / img_meta["file_name"]
            if src_img.exists():
                shutil.copy(src_img, dst_img)

            # Copy visualization if exists
            src_viz = self.dataset_dir / "visualizations" / f"{img_id}_viz.png"
            dst_viz = self.output_dir / "visualizations" / f"{img_id}_viz.png"
            if src_viz.exists():
                shutil.copy(src_viz, dst_viz)

        # Save metadata
        stimuli_metadata = {
            "info": {
                "description": "Final Curated Stimuli for Relational Gaze Experiment",
                "num_images": len(selected_images),
                "source": "Manual curation from diverse selection",
                "diversity_metrics": self.dataset["info"].get("diversity_metrics", {}),
                "filtering_criteria": self.dataset["info"].get(
                    "filtering_criteria", {}
                ),
            },
            "images": selected_images,
        }

        metadata_path = self.output_dir / "annotations" / "stimuli_dataset.json"
        with open(metadata_path, "w") as f:
            json.dump(stimuli_metadata, f, indent=2)

        print(f"✓ Exported {len(selected_images)} images")
        print(f"✓ Images: {self.output_dir / 'images'}")
        print(f"✓ Visualizations: {self.output_dir / 'visualizations'}")
        print(f"✓ Metadata: {metadata_path}")
        print("=" * 60)

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
                self.setWindowTitle("SVG Relational Image Curation")
                self.setGeometry(100, 100, 1800, 1000)

                # Central widget
                central_widget = QWidget()
                self.setCentralWidget(central_widget)

                # Main layout
                layout = QVBoxLayout()

                # Matplotlib canvas with 2 subplots
                self.figure = Figure(figsize=(18, 9))
                self.canvas = FigureCanvasQTAgg(self.figure)
                self.ax_left = self.figure.add_subplot(121)  # Left panel
                self.ax_right = self.figure.add_subplot(122)  # Right panel

                layout.addWidget(self.canvas)

                # Status label
                self.status_label = QLabel()
                self.status_label.setAlignment(Qt.AlignCenter)
                self.status_label.setFont(QFont("Arial", 11, QFont.Bold))
                layout.addWidget(self.status_label)

                # Selected images list label
                self.selected_list_label = QLabel()
                self.selected_list_label.setAlignment(Qt.AlignLeft)
                self.selected_list_label.setFont(QFont("Arial", 9))
                self.selected_list_label.setStyleSheet(
                    "background-color: #e8f5e9; padding: 5px; border-radius: 3px;"
                )
                layout.addWidget(self.selected_list_label)

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
                self.ax_left.clear()
                self.ax_right.clear()

                if self.ui.current_index >= len(self.ui.images_to_annotate):
                    self.ax_left.text(
                        0.5,
                        0.5,
                        "All images annotated!\nClose window to exit.",
                        ha="center",
                        va="center",
                        fontsize=20,
                        fontweight="bold",
                    )
                    self.ax_left.axis("off")
                    self.ax_right.axis("off")
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

                # LEFT: Image with segmentations
                self.ui._draw_image_with_segmentations(current_img, self.ax_left)

                # RIGHT: Raw image
                self.ui._draw_raw_image(current_img, self.ax_right)

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

                # Update selected images list
                selected_count = len(self.ui.selected_images_list)
                if selected_count == 0:
                    self.selected_list_label.setText("✓ SELECTED IMAGES: None yet")
                else:
                    # Show first 10 selected images
                    display_list = self.ui.selected_images_list[:10]
                    list_text = ", ".join(display_list)
                    if selected_count > 10:
                        list_text += f" ... and {selected_count - 10} more"
                    self.selected_list_label.setText(
                        f"✓ SELECTED IMAGES ({selected_count}): {list_text}"
                    )

            def closeEvent(self, event):
                """Handle window close."""
                self.ui._print_summary()

                # Ask if user wants to export selected images
                selected_count = self.ui._count_status("selected")
                if selected_count > 0:
                    print(f"\n{selected_count} images were selected.")
                    self.ui.export_selected_images()

                event.accept()

        # Create and run application
        app = QApplication(sys.argv)
        window = AnnotationWindow(self)
        window.show()
        sys.exit(app.exec_())


def main():
    """Main entry point."""
    # Configuration
    DATASET_DIR = "data/diverse_scored_selection"
    OUTPUT_DIR = "data/stimuli"
    ANNOTATIONS_FILE = "manual_annotations.json"

    # Create UI
    ui = ManualAnnotationUI(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        annotations_file=ANNOTATIONS_FILE,
    )

    # Run interface
    ui.run()


if __name__ == "__main__":
    main()
