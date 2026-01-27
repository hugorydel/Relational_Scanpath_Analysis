#!/usr/bin/env python3
"""
Score specific images for model comparison experiment.

Usage:
    python score_images_comparison.py --image-ids image_ids.txt --model gpt-5-mini --output results_gpt5mini.jsonl

This script processes ONLY the images specified in the image IDs file.
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
RESPONSE_SCHEMA = response_schema

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def preprocess_image(
    image_path: Path, max_dimension: int = 1024, jpeg_quality: int = 85
) -> str:
    """Load, resize if needed, and encode image as base64 JPEG."""
    img = Image.open(image_path)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    needs_resize = max(width, height) > max_dimension

    if needs_resize:
        scale = max_dimension / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


# ============================================================================
# LOAD IMAGE IDS
# ============================================================================


def load_image_ids(ids_path: Path) -> List[str]:
    """Load image IDs from text file (one per line)."""
    with open(ids_path, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]
    return image_ids


# ============================================================================
# RESULT MANAGEMENT
# ============================================================================


def load_existing_results(output_path: Path) -> Set[str]:
    """Load already-processed image IDs from output file."""
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


# ============================================================================
# API CLIENT
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
    """Process images in batches with concurrency control."""

    def __init__(
        self,
        scorer: ImageScorer,
        image_dir: Path,
        output_path: Path,
        errors_path: Path,
        max_concurrency: int = 20,
        max_dimension: int = 1024,
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
        self.processed = 0
        self.errors = 0
        self.output_lock = asyncio.Lock()
        self.errors_lock = asyncio.Lock()

    async def process_single_image(self, image_id: str) -> None:
        """Process a single image."""
        async with self.semaphore:
            try:
                # Load and preprocess image
                image_path = self.image_dir / f"{image_id}.jpg"

                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image_base64 = preprocess_image(
                    image_path, self.max_dimension, self.jpeg_quality
                )

                # Score image
                result = await self.scorer.score_image(image_id, image_base64)

                # Write result
                async with self.output_lock:
                    async with aiofiles.open(
                        self.output_path, "a", encoding="utf-8"
                    ) as f:
                        await f.write(json.dumps(result, ensure_ascii=False) + "\n")

                self.processed += 1
                print(
                    f"✓ [{self.processed}] {image_id}: CIC={result['CIC']}, SEP={result['SEP']}, DYN={result['DYN']}, QLT={result['QLT']}"
                )

            except Exception as e:
                error_record = {"image_id": image_id, "error": str(e)}

                async with self.errors_lock:
                    async with aiofiles.open(
                        self.errors_path, "a", encoding="utf-8"
                    ) as f:
                        await f.write(
                            json.dumps(error_record, ensure_ascii=False) + "\n"
                        )

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

    ids_path = Path(args.image_ids)
    if not ids_path.exists():
        print(f"Error: Image IDs file not found: {ids_path}")
        sys.exit(1)

    # Load image IDs to process
    target_image_ids = load_image_ids(ids_path)
    print(f"✓ Loaded {len(target_image_ids)} image IDs from {ids_path}")

    # Load existing results to avoid reprocessing
    output_path = Path(args.output)
    errors_path = Path(args.errors)

    processed_images = load_existing_results(output_path)

    if processed_images:
        print(f"✓ Found {len(processed_images)} already-processed images")

    # Filter out already-processed images
    unprocessed = [img for img in target_image_ids if img not in processed_images]
    skipped_count = len(target_image_ids) - len(unprocessed)

    print("=" * 70)
    print("MODEL COMPARISON SCORING")
    print("=" * 70)
    print(f"Image directory: {image_dir}")
    print(f"Target images: {len(target_image_ids)}")
    print(f"Already processed: {skipped_count}")
    print(f"To process: {len(unprocessed)}")
    print(f"Model: {args.model}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Output: {args.output}")
    print(f"Errors: {args.errors}")
    print("=" * 70)

    if len(unprocessed) == 0:
        print("\n✓ All target images already processed!")
        return

    # Get API key securely
    print("\n" + "=" * 70)
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")

    if not api_key or not api_key.strip():
        print("Error: No API key provided")
        sys.exit(1)

    print("✓ API key received")
    print("=" * 70)

    # Initialize output files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")

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
    print(f"\nProcessing {len(unprocessed)} images...\n")
    await processor.process_batch(unprocessed)

    # Summary
    total_processed = len(processed_images) + processor.processed

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ This run: {processor.processed} successful, {processor.errors} errors")
    print(
        f"✓ Total: {total_processed}/{len(target_image_ids)} images ({100*total_processed/len(target_image_ids):.1f}%)"
    )
    print(f"\n📄 Results: {output_path}")
    if processor.errors > 0:
        print(f"⚠️  Error log: {errors_path}")
    print("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Score specific images for model comparison experiment"
    )

    parser.add_argument(
        "--image-ids",
        type=str,
        required=True,
        help="Text file with image IDs to process (one per line)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./data/processed/images",
        help="Directory containing .jpg images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file (e.g., results_gpt5mini.jsonl)",
    )
    parser.add_argument(
        "--errors", type=str, default="./errors.jsonl", help="Error log JSONL file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenAI model to use (e.g., gpt-5-mini)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=20, help="Max concurrent API requests"
    )
    parser.add_argument(
        "--max-dimension", type=int, default=1024, help="Max image dimension in pixels"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=85, help="JPEG compression quality 1-100"
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
