"""Diagnostic utility to test face detection and composition classification on an image."""

import argparse
import sys
from pathlib import Path

from image_reference_generator.pass1_filter import process_pass1


def debug_image_face(image_path: Path) -> None:
    """Evaluates an image through Pass 1 face detection and reports results."""
    if not image_path.is_file():
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    metrics = process_pass1(image_path)
    if metrics is None:
        print(
            f"❌ Image '{image_path.name}' failed Pass 1 filter "
            "(no face, multiple faces, corrupt, or <1024px)."
        )
    else:
        print(f"✅ Pass 1 Passed for '{image_path.name}':")
        print(f"   Dimensions: {metrics.width}x{metrics.height}")
        print(
            f"   Face box: x={metrics.face_box[0]}, y={metrics.face_box[1]}, "
            f"w={metrics.face_box[2]}, h={metrics.face_box[3]}"
        )
        print(f"   Composition: {metrics.composition}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face detection on a single image")
    parser.add_argument("image_path", type=Path, help="Path to image file")
    args = parser.parse_args()
    debug_image_face(args.image_path)
