"""Diagnostic utility to inspect image resolutions across a directory."""

import argparse
import sys
from pathlib import Path

import cv2


def check_dimensions(directory: Path, limit: int = 20) -> None:
    """Inspects and prints dimensions for image files in the target directory."""
    if not directory.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([f for f in directory.iterdir() if f.suffix.lower() in extensions])[:limit]
    if not files:
        print(f"No image files found in {directory}")
        return

    print(f"Inspecting first {len(files)} images in {directory}:")
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            h, w = img.shape[:2]
            print(f"  {f.name}: {w}x{h} px")
        else:
            print(f"  {f.name}: [Failed to load]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect dimensions of images in a folder")
    parser.add_argument("directory", nargs="?", default=".", type=Path, help="Folder to inspect")
    parser.add_argument("--limit", default=20, type=int, help="Maximum images to check")
    args = parser.parse_args()
    check_dimensions(args.directory, limit=args.limit)
