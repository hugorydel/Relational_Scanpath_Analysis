#!/usr/bin/env python3
"""
Image Interaction Scoring with OpenAI Vision API

Usage:
    python score_images_openai.py

    The script will:
    - Scan ./data/processed/images/ for all .jpg files
    - Skip images already in results.jsonl
    - Process up to 200 new/failed images (configurable with --max-images)
    - Prompt you to enter your OpenAI API key securely

Options:
    --image-dir PATH         Directory containing .jpg images (default: ./data/processed/images)
    --output PATH            Output JSONL file
    --errors PATH            Error log JSONL file
    --max-images N           Max number of new images to process (default: 200, 0 = all)
    --max-concurrency N      Max concurrent API requests (default: 5)
    --dry-run                Print plan without making API calls
    --max-dimension N        Max image dimension in pixels (default: 1024)
    --jpeg-quality N         JPEG compression quality 1-100 (default: 85)
    --model NAME             OpenAI model to use
    --force-reprocess        Reprocess all images, ignoring existing results

Requirements:
    pip install openai pillow aiofiles
"""

import argparse
import asyncio
import base64
import getpass
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import aiofiles
from PIL import Image

import config
from openai import AsyncOpenAI
from openAI.response_schema import response_schema
from openAI.scoring_prompt import scoring_prompt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCORING_PROMPT = scoring_prompt


# JSON schema for structured output
RESPONSE_SCHEMA = response_schema


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def preprocess_image(
    image_path: Path, max_dimension: int = 1024, jpeg_quality: int = 85
) -> str:
    """
    Load, resize if needed, and encode image as base64 JPEG.
    Optimized for images that are already 1024x768 or similar sizes.

    Args:
        image_path: Path to image file
        max_dimension: Maximum dimension (width or height) in pixels
        jpeg_quality: JPEG compression quality (1-100)

    Returns:
        Base64-encoded JPEG string
    """
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Only resize if image exceeds max_dimension
    width, height = img.size
    needs_resize = max(width, height) > max_dimension

    if needs_resize:
        scale = max_dimension / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Encode as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)

    # Return base64
    return base64.b64encode(buffer.read()).decode("utf-8")


# ============================================================================
# IMAGE DISCOVERY
# ============================================================================


def discover_images(image_dir: Path) -> List[str]:
    """
    Discover all .jpg images in the directory.

    Args:
        image_dir: Directory containing images

    Returns:
        List of image IDs without extension (e.g., ["1026", "2345", ...])
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Find all .jpg files (case-insensitive)
    # Use set to deduplicate in case filesystem is case-insensitive
    jpg_files = set(image_dir.glob("*.jpg")) | set(image_dir.glob("*.JPG"))
    jpg_files = list(jpg_files)

    if not jpg_files:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")

    # Extract image IDs (filenames without extension)
    image_ids = [f.stem for f in jpg_files]

    # Deduplicate image IDs (in case of case variations)
    image_ids = list(set(image_ids))

    # Sort by numeric ID
    def extract_numeric_id(image_id: str) -> int:
        try:
            return int(image_id)
        except ValueError:
            return 0  # Non-numeric names sort first

    image_ids.sort(key=extract_numeric_id)

    return image_ids


# ============================================================================
# RESULT MANAGEMENT
# ============================================================================


def load_existing_results(output_path: Path) -> Set[str]:
    """
    Load already-processed image IDs from output file.

    Args:
        output_path: Path to results JSONL file

    Returns:
        Set of image IDs that have been successfully processed
    """
    processed_ids = set()

    if not output_path.exists():
        return processed_ids

    try:
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        if "image_id" in result:
                            processed_ids.add(result["image_id"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")

    return processed_ids


def load_errored_images(errors_path: Path) -> Set[str]:
    """
    Load image IDs that previously errored.

    Args:
        errors_path: Path to errors JSONL file

    Returns:
        Set of image IDs that had errors
    """
    errored_ids = set()

    if not errors_path.exists():
        return errored_ids

    try:
        with open(errors_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        error = json.loads(line)
                        if "image_id" in error:
                            errored_ids.add(error["image_id"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load error log: {e}")

    return errored_ids


# ============================================================================
# API CLIENT WITH RETRY LOGIC
# ============================================================================


class ImageScorer:
    """Async OpenAI client with retry logic and rate limiting."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    async def score_image(self, image_id: str, image_base64: str) -> Dict[str, Any]:
        """
        Score a single image with exponential backoff retry.

        Args:
            image_id: Image ID without extension (e.g., "1026")
            image_base64: Base64-encoded image

        Returns:
            Scoring result dict

        Raises:
            Exception: If all retries exhausted
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": SCORING_PROMPT},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            ],
                        }
                    ],
                    text={"format": RESPONSE_SCHEMA},
                    max_output_tokens=1500,
                )

                # Extract JSON from response
                result = json.loads(response.output_text)

                # Validate and sanitize
                result["image_id"] = image_id
                result["story"] = str(result.get("story", "")).strip()
                result["core_interactions"] = result.get("core_interactions", [])[:3]
                result["CIC"] = max(0, min(3, int(result.get("CIC", 0))))
                result["SEP"] = max(0, min(2, int(result.get("SEP", 0))))
                result["DYN"] = max(0, min(2, int(result.get("DYN", 0))))
                result["QLT"] = max(0, min(1, int(result.get("QLT", 0))))

                return result

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate_limit" in error_msg.lower() or "429" in error_msg

                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2**attempt)
                    if is_rate_limit:
                        delay *= 2  # Extra delay for rate limits

                    safe_error = error_msg.encode("ascii", "backslashreplace").decode(
                        "ascii"
                    )
                    print(
                        f"  Retry {attempt + 1}/{self.max_retries} for {image_id} "
                        f"(waiting {delay:.1f}s): {safe_error[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"All retries exhausted: {error_msg}")


# ============================================================================
# BATCH PROCESSOR
# ============================================================================


class BatchProcessor:
    """Process images with concurrency control."""

    def __init__(
        self,
        scorer: ImageScorer,
        image_dir: Path,
        output_path: Path,
        errors_path: Path,
        max_concurrency: int = 5,
        max_dimension: int = 2048,
        jpeg_quality: int = 85,
    ):
        self.scorer = scorer
        self.image_dir = image_dir
        self.output_path = output_path
        self.errors_path = errors_path
        self.max_concurrency = max_concurrency
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.results_lock = asyncio.Lock()
        self.errors_lock = asyncio.Lock()

        self.processed = 0
        self.errors = 0

    async def process_single_image(self, image_id: str) -> None:
        """
        Process a single image with concurrency control.

        Args:
            image_id: Image ID without extension (e.g., "1026")
        """
        async with self.semaphore:
            try:
                # Construct image path (append .jpg extension)
                image_path = self.image_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                # Load and preprocess image
                image_base64 = preprocess_image(
                    image_path,
                    max_dimension=self.max_dimension,
                    jpeg_quality=self.jpeg_quality,
                )

                # Score image
                result = await self.scorer.score_image(image_id, image_base64)

                # Write result
                async with self.results_lock:
                    async with aiofiles.open(self.output_path, "a") as f:
                        await f.write(json.dumps(result) + "\n")

                self.processed += 1
                print(
                    f"✓ [{self.processed}] {image_id}: "
                    f"CIC={result['CIC']} SEP={result['SEP']} DYN={result['DYN']} QLT={result['QLT']}"
                )

            except Exception as e:
                error_record = {"image_id": image_id, "error": str(e)}

                async with self.errors_lock:
                    async with aiofiles.open(self.errors_path, "a") as f:
                        await f.write(json.dumps(error_record) + "\n")

                self.errors += 1
                print(f"✗ [{self.errors}] {image_id}: {str(e)[:100]}")

    async def process_batch(self, image_ids: List[str]) -> None:
        """Process a batch of images concurrently."""
        tasks = [self.process_single_image(img_id) for img_id in image_ids]
        await asyncio.gather(*tasks)


# ============================================================================
# MAIN
# ============================================================================


async def main_async(args):
    """Async main function."""

    # Validate inputs
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    # Discover all images in directory
    try:
        all_images = discover_images(image_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load existing results to avoid reprocessing
    output_path = Path(args.output)
    errors_path = Path(args.errors)

    processed_images = set()
    errored_images = set()

    if not args.force_reprocess:
        print("Checking for existing results...")
        processed_images = load_existing_results(output_path)
        errored_images = load_errored_images(errors_path)

        if processed_images:
            print(f"✓ Found {len(processed_images)} already-processed images")
        if errored_images:
            print(
                f"⚠️  Found {len(errored_images)} previously errored images (will retry)"
            )

    # Filter out already-processed images (but retry errored ones)
    if args.force_reprocess:
        unprocessed = all_images
        skipped_count = 0
    else:
        unprocessed = [img for img in all_images if img not in processed_images]
        skipped_count = len(all_images) - len(unprocessed)

    # Apply max images limit
    max_images = args.max_images
    if max_images > 0:
        image_ids = unprocessed[:max_images]
        remaining_after_limit = len(unprocessed) - max_images
    else:
        image_ids = unprocessed
        remaining_after_limit = 0

    print("=" * 70)
    print("IMAGE INTERACTION SCORING")
    print("=" * 70)
    print(f"Image directory: {image_dir}")
    print(f"Total images found: {len(all_images)}")

    if not args.force_reprocess and skipped_count > 0:
        print(f"Already processed: {skipped_count} images")
        print(f"Unprocessed: {len(unprocessed)} images")
        print(
            f"Retrying: {len([img for img in image_ids if img in errored_images])} previously errored"
        )

    if max_images > 0:
        print(f"Max images per run: {max_images}")
        print(f"Processing this run: {len(image_ids)} images")
        if remaining_after_limit > 0:
            print(f"Remaining for next run: {remaining_after_limit} images")
    else:
        print(f"Processing: {len(image_ids)} images (all unprocessed)")

    print(f"Model: {args.model}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Max dimension: {args.max_dimension}px")
    print(f"JPEG quality: {args.jpeg_quality}")
    print(f"Output: {args.output}")
    print(f"Errors: {args.errors}")
    print("=" * 70)

    if len(image_ids) == 0:
        print("\n✓ All images already processed! Nothing to do.")
        print("Use --force-reprocess to reprocess all images.")
        return

    # Get API key securely (no echo to terminal or logs)
    print("\n" + "=" * 70)
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")

    if not api_key or not api_key.strip():
        print("Error: No API key provided")
        sys.exit(1)

    print("✓ API key received")
    print("=" * 70)

    # Initialize output files (append mode to preserve existing results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    # Create files if they don't exist
    if not output_path.exists():
        output_path.write_text("")
    if not errors_path.exists():
        errors_path.write_text("")

    # Initialize scorer and processor
    scorer = ImageScorer(api_key=api_key.strip(), model=args.model)
    processor = BatchProcessor(
        scorer=scorer,
        image_dir=image_dir,
        output_path=output_path,
        errors_path=errors_path,
        max_concurrency=args.max_concurrency,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    # Process images
    print(f"\nProcessing {len(image_ids)} images...\n")
    await processor.process_batch(image_ids)

    # Summary
    total_processed = len(processed_images) + processor.processed
    total_errors = (
        len(errored_images)
        + processor.errors
        - len([img for img in image_ids if img in errored_images])
    )

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ This run: {processor.processed} successful, {processor.errors} errors")
    print(f"✓ Total in dataset: {total_processed} successful, {total_errors} errors")
    print(
        f"✓ Overall progress: {total_processed}/{len(all_images)} images ({100*total_processed/len(all_images):.1f}%)"
    )

    if remaining_after_limit > 0:
        print(
            f"\n💡 {remaining_after_limit} images remaining. Run again to process next batch."
        )
    elif total_processed < len(all_images):
        unprocessed_remaining = len(all_images) - total_processed - total_errors
        if unprocessed_remaining > 0:
            print(
                f"\n💡 {unprocessed_remaining} images remaining. Run again to continue."
            )

    print(f"\n📄 Results: {output_path}")
    if processor.errors > 0 or total_errors > 0:
        print(f"⚠️  Error log: {errors_path}")
    print("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Score images for interaction analysis using OpenAI Vision API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="./data/processed/images",
        help="Directory containing .jpg images (default: ./data/processed/images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/scored_images/results.jsonl",
        help="Output JSONL file (default: ./data/scored_images/results.jsonl)",
    )
    parser.add_argument(
        "--errors",
        type=str,
        default="./data/scored_images/errors.jsonl",
        help="Error log JSONL file (default: ./data/scored_images/errors.jsonl)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Max number of new images to process (default: 200, 0 = all)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Max image dimension in pixels (default: 1024)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG compression quality 1-100 (default: 85)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.OPENAI_MODEL,
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess all images, ignoring existing results",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
