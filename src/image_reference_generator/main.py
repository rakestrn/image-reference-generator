import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from .ollama_integration import generate_caption
from .pass1_filter import process_pass1
from .pass2_ranking import process_pass2
from .pass3_diversity import process_pass3
from .post_processor import process_and_save_images


def main():
    """CLI Entry Point orchestrating the Curation pipeline.

    Validates arguments, executes the exact sequence of filters/rankings, and bridges the
    Ollama integration. Surfaces detailed terminal progress using tqdm arrays.

    Returns:
        Exit code 0 on success, 1 on explicit failure or invalid directories.
    """
    parser = argparse.ArgumentParser(description="Anti-Gravity Image Curation Tool")
    parser.add_argument("--input", required=True, type=Path, help="Input folder with raw images")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output folder for curated images"
    )
    parser.add_argument(
        "--trigger", required=True, type=str, help="Trigger word for naming and captioning"
    )
    parser.add_argument(
        "--top_n",
        default=100,
        type=int,
        help="Limit for technical ranking before diversity extraction",
    )
    parser.add_argument(
        "--model", default="llama3.2-vision", type=str, help="Ollama vision model to use"
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: Input directory '{args.input}' does not exist.")
        return 1

    print("🚀 Starting Anti-Gravity Image Curation...")
    image_paths = []
    # Exhaustive coverage of typical DSLR/Smartphone extension casings
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp"]:
        image_paths.extend(args.input.glob(ext))

    if not image_paths:
        print(f"Error: No images found in {args.input}")
        return 1

    # --- PASS 1 ---
    print(f"\n[Pass 1]: Initial Liquidity Filter (Processing {len(image_paths)} images)")
    pass1_results = []
    for p in tqdm(image_paths, desc="Filtering invalid images"):
        res = process_pass1(p)
        if res:
            pass1_results.append(res)

    print(f" -> {len(pass1_results)} images passed strict technical formatting.")
    if len(pass1_results) < 25:
        print(" -> Warning: Less than 25 images passed. Portfolio size will be restricted.")

    if not pass1_results:
        return 0

    # --- PASS 2 ---
    print("\n[Pass 2]: Technical Quality Ranking (Optimizing Alpha)")
    pass2_results = process_pass2(pass1_results, top_n=args.top_n)
    print(f" -> Top {len(pass2_results)} technically optimal candidates advanced.")

    # --- PASS 3 ---
    print("\n[Pass 3]: Diversity Optimization (Minimizing Correlation)")
    final_portfolio = process_pass3(pass2_results, target_count=25)
    print(f" -> Generated highly diverse final portfolio of size {len(final_portfolio)}.")

    # --- POST PROCESSING ---
    print("\n[Finalizing]: Adaptive scaling and standardized strict boundary crop")
    process_and_save_images(final_portfolio, args.output, args.trigger)

    # --- CAPTIONING ---
    print(f"\n[AI Labeling]: Generating high-precision descriptions via {args.model}")
    output_files = sorted(args.output.glob(f"{args.trigger}_*.jpg"))
    for file_path in tqdm(output_files, desc="Captioning loop"):
        generate_caption(file_path, args.trigger, args.model)

    print(f"\n✅ Portfolio extraction completed gracefully. Assets archived in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
