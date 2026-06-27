import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from .ollama_integration import check_ollama_status, generate_caption
from .pass1_filter import process_pass1
from .pass2_ranking import process_pass2
from .pass3_diversity import process_pass3
from .pass4_refinement import process_pass4
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
        "--target_count",
        default=30,
        type=int,
        help="Exact number of total images required in the final portfolio",
    )
    parser.add_argument(
        "--top_n",
        default=100,
        type=int,
        help="Limit for technical ranking before diversity extraction",
    )
    parser.add_argument(
        "--model", default="gemma4:26b", type=str, help="Ollama vision model to use"
    )
    parser.add_argument(
        "--format",
        default="preserve",
        choices=["jpg", "png", "webp", "preserve"],
        type=str,
        help="Output image format (jpg, png, webp, or preserve original)",
    )
    parser.add_argument(
        "--preserve_aspect_ratio",
        action="store_true",
        help="Dynamically scale instead of strict square cropping",
    )
    parser.add_argument(
        "--drop_watermarked",
        action="store_true",
        help="Drop images with watermarks entirely instead of blurring",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Tensor batch size for pass 3")

    args = parser.parse_args()

    # Ensure top_n provides enough pool size for the requested target_count
    if args.top_n < args.target_count * 2:
        args.top_n = args.target_count * 2

    print("🚀 Running Pre-Flight Checks...")
    if not check_ollama_status(args.model):
        print("❌ Pre-flight check failed. Exiting.")
        return 1

    if not args.input.is_dir():
        print(f"❌ Error: Input directory '{args.input}' does not exist.")
        return 1

    if not args.output.parent.exists():
        print(f"❌ Error: Parent directory for output '{args.output.parent}' does not exist.")
        return 1

    image_paths = []
    # Exhaustive coverage of typical DSLR/Smartphone extension casings
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"]:
        image_paths.extend(args.input.glob(ext))

    if not image_paths:
        print(f"❌ Error: No images found in {args.input}")
        return 1

    required_buffer = int(args.target_count * 1.5)
    if len(image_paths) < required_buffer:
        print(
            f"❌ Error: Insufficient raw images. Found {len(image_paths)}, "
            f"but require at least {required_buffer} (1.5x target_count) "
            "to ensure a safe buffer for quality and duplicate drops."
        )
        return 1

    print("✅ Pre-Flight Checks Passed!")
    print("🚀 Starting Anti-Gravity Image Curation...")

    try:
        # --- PASS 1 ---
        print(f"\n[Pass 1]: Initial Liquidity Filter (Processing {len(image_paths)} images)")
        pass1_results = []
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed

        max_workers = min(32, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pass1, p): p for p in image_paths}
            completed_futures = as_completed(futures)
            for future in tqdm(
                completed_futures, total=len(futures), desc="Filtering invalid images"
            ):
                try:
                    res = future.result()
                    if res:
                        pass1_results.append(res)
                except Exception as e:
                    print(f"\n⚠️ Error processing image {futures[future]}: {str(e)}")

        print(f" -> {len(pass1_results)} images passed strict technical formatting.")
        if len(pass1_results) < 30:
            print(" -> Warning: Less than 30 images passed. Portfolio size will be restricted.")

        if not pass1_results:
            return 1

        # --- PASS 2 ---
        print("\n[Pass 2]: Technical Quality Ranking (Optimizing Alpha)")
        pass2_results = process_pass2(pass1_results, top_n=args.top_n)
        print(f" -> Top {len(pass2_results)} technically optimal candidates advanced.")

        # --- PASS 3 ---
        print("\n[Pass 3]: Diversity Optimization (Minimizing Correlation)")
        pass3_results = process_pass3(
            pass2_results, target_count=args.target_count, batch_size=args.batch_size
        )
        print(f" -> Generated highly diverse final portfolio of size {len(pass3_results)}.")

        # --- PASS 4 ---
        print("\n[Pass 4]: Holistic Set Refinement (Consensus, Lighting, Aesthetics)")
        final_portfolio = process_pass4(pass3_results, pass2_results, model_name=args.model)
        print(f" -> Completed final portfolio size: {len(final_portfolio)}.")

        # --- POST PROCESSING ---
        print("\n[Finalizing]: Adaptive scaling and standardized strict boundary crop")
        if final_portfolio:
            process_and_save_images(
                final_portfolio,
                args.output,
                args.trigger,
                output_format=args.format,
                preserve_aspect_ratio=args.preserve_aspect_ratio,
                drop_watermarked=args.drop_watermarked,
            )
        else:
            print(" -> No valid images survived pipeline. Exiting.")
            return 1

        # Security check: Verify the folder actually contains the images we just 'processed'
        target_exts = (
            [f".{args.format}"] if args.format != "preserve" else [".jpg", ".jpeg", ".png", ".webp"]
        )

        exported_files = []
        for ext in target_exts:
            exported_files.extend(args.output.glob(f"{args.trigger}_*{ext}"))

        if not exported_files:
            print(f"\n🔥 Critical failure: The output directory {args.output} is empty.")
            print("This usually indicates a volume mounting or file handle issue.")
            return 1

        print(f" -> Successfully exported {len(exported_files)} standardized assets.")

        # --- CAPTIONING ---
        print(f"\n[AI Labeling]: Generating high-precision descriptions via {args.model}")
        output_files = []
        for ext in target_exts:
            output_files.extend(args.output.glob(f"{args.trigger}_*{ext}"))
        output_files = sorted(output_files)

        for file_path, metrics in tqdm(
            zip(output_files, final_portfolio, strict=False),
            total=len(final_portfolio),
            desc="Captioning loop",
        ):
            generate_caption(
                file_path, args.trigger, composition=metrics.composition, model_name=args.model
            )

        print(f"\n✅ Portfolio extraction completed gracefully. Assets archived in {args.output}")

        print(f"\n🧹 Cleaning up: Stopping {args.model} to free system memory...")
        subprocess.run(["ollama", "stop", args.model], check=False)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user. Aborting curation.")
        return 130
    except Exception as e:
        print(f"\n\n🔥 Critical failure in execution pipeline: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
