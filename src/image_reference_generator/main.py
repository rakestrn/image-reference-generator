import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Suppress low-level C++/absl/glog/TFLite logging across parent and child processes
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from tqdm import tqdm

from .ollama_integration import (
    DEFAULT_VISION_MODEL,
    audit_directory,
    check_ollama_status,
    generate_caption,
    recaption_directory,
)
from .pass1_filter import process_pass1
from .pass2_ranking import process_pass2
from .pass3_diversity import process_pass3
from .pass4_refinement import process_pass4
from .post_processor import process_and_save_images
from .replicate_trainer import start_replicate_training


def package_replicate_dataset(
    output_dir: Path, trigger_word: str
) -> tuple[bool, Path | None, str]:
    """Validates and packages curated images and captions into a Replicate-ready zip archive.

    Verifies:
    1. Every exported image has an exact matching .txt caption with the same stem.
    2. Every .txt caption is non-empty and contains the trigger word.
    3. Bundles all pairs into {trigger_word}_dataset.zip at the archive root.

    Args:
        output_dir: Path to directory containing curated images and captions.
        trigger_word: Target identifying sequence.

    Returns:
        tuple of (is_valid, zip_path, summary_message).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([
        f
        for f in output_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts and f.name.startswith(f"{trigger_word}_")
    ])

    if not image_files:
        return (
            False,
            None,
            f"No curated images found in {output_dir} starting with '{trigger_word}_'",
        )

    missing_captions = []
    invalid_captions = []
    pairs_to_pack: list[tuple[Path, Path]] = []

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            missing_captions.append(img_path.name)
            continue

        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            if not content:
                invalid_captions.append((txt_path.name, "Empty caption file"))
            elif trigger_word.lower() not in content.lower():
                invalid_captions.append(
                    (txt_path.name, f"Trigger word '{trigger_word}' missing in caption")
                )
            else:
                pairs_to_pack.append((img_path, txt_path))
        except Exception as e:
            invalid_captions.append((txt_path.name, str(e)))

    if missing_captions:
        return (
            False,
            None,
            f"Missing caption files for {len(missing_captions)} images: {missing_captions[:3]}",
        )

    if invalid_captions:
        return (
            False,
            None,
            f"Found {len(invalid_captions)} invalid captions: {invalid_captions[:3]}",
        )

    zip_path = output_dir / f"{trigger_word}_dataset.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img_path, txt_path in pairs_to_pack:
                zf.write(img_path, arcname=img_path.name)
                zf.write(txt_path, arcname=txt_path.name)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        msg = (
            f"Successfully packaged {len(pairs_to_pack)} image-caption pairs into "
            f"{zip_path.name} ({size_mb:.2f} MB)"
        )
        return True, zip_path, msg
    except Exception as e:
        return False, None, f"Failed to create zip archive: {str(e)}"


def print_replicate_instructions(
    zip_path: Path,
    trigger_word: str,
    steps: int = 3000,
    lora_rank: int = 32,
) -> None:
    """Prints copy-paste ready Replicate training CLI and Python SDK commands."""
    print("\n" + "=" * 80)
    print("🚀 Replicate FLUX.1 LoRA Training Command")
    print("=" * 80)
    print(
        f"replicate train ostris/flux-dev-lora-trainer \\\n"
        f'  --destination "your-username/model-name" \\\n'
        f"  input_images=@{zip_path} \\\n"
        f'  trigger_word="{trigger_word}" \\\n'
        f"  autocaption=false \\\n"
        f"  steps={steps} \\\n"
        f"  lora_rank={lora_rank}"
    )
    print("\nPython SDK snippet:")
    print(
        f"import replicate\n\n"
        f"training = replicate.trainings.create(\n"
        f'    version="ostris/flux-dev-lora-trainer",\n'
        f"    input={{\n"
        f'        "input_images": open("{zip_path}", "rb"),\n'
        f'        "trigger_word": "{trigger_word}",\n'
        f'        "autocaption": False,\n'
        f'        "steps": {steps},\n'
        f'        "lora_rank": {lora_rank},\n'
        f"    }},\n"
        f'    destination="your-username/model-name",\n'
        f")"
    )
    print("=" * 80 + "\n")


def main():
    """CLI Entry Point orchestrating the Curation pipeline.

    Validates arguments, executes the exact sequence of filters/rankings, and bridges the
    Ollama integration. Surfaces detailed terminal progress using tqdm arrays.

    Returns:
        Exit code 0 on success, 1 on explicit failure or invalid directories.
    """
    parser = argparse.ArgumentParser(description="Anti-Gravity Image Curation Tool")
    parser.add_argument(
        "--input", required=False, default=None, type=Path, help="Input folder with raw images"
    )
    parser.add_argument(
        "--output", required=False, default=None, type=Path, help="Output folder for curated images"
    )
    parser.add_argument(
        "--trigger", required=True, type=str, help="Trigger word for naming and captioning"
    )
    parser.add_argument(
        "--audit_only",
        default=None,
        type=Path,
        help="Audit an existing directory of caption text files without running curation",
    )
    parser.add_argument(
        "--recaption_dir",
        default=None,
        type=Path,
        help="Audit and re-caption an existing portfolio directory in place",
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
        "--model",
        default=DEFAULT_VISION_MODEL,
        type=str,
        help=(
            f"Ollama vision model to use (default: {DEFAULT_VISION_MODEL}, "
            "alternatives: qwen2.5vl:7b, llava:13b)"
        ),
    )
    parser.add_argument(
        "--format",
        default="jpg",
        choices=["jpg", "png", "webp", "preserve"],
        type=str,
        help="Output image format (jpg, png, webp, or preserve original; default: jpg)",
    )
    parser.add_argument(
        "--preserve_aspect_ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve native aspect ratio snapped to 64px increments (default: True)",
    )
    parser.add_argument(
        "--crop_square",
        action="store_true",
        default=False,
        help="Override aspect ratio bucketing and force 1024x1024 square crop",
    )
    parser.add_argument(
        "--package_zip",
        action="store_true",
        default=False,
        help="Bundle curated dataset into a Replicate-ready training zip archive",
    )
    parser.add_argument(
        "--drop_watermarked",
        action="store_true",
        help="Drop images with watermarks entirely instead of blurring",
    )
    parser.add_argument(
        "--temporal_balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stratify candidates across chronological eras to avoid recency bias",
    )
    parser.add_argument(
        "--era_window",
        default=None,
        type=str,
        help="Filter photos to an era window (e.g. '2020-2024', '2023', 'all')",
    )
    parser.add_argument(
        "--favor_no_glasses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Favor photos without sunglasses to optimize LoRA facial likeness (default: True)",
    )
    parser.add_argument(
        "--drop_sunglasses",
        action="store_true",
        default=False,
        help="Strictly drop all photos with sunglasses or tinted eyewear",
    )
    parser.add_argument(
        "--min_resolution",
        default=1024,
        type=int,
        help="Minimum allowed shortest edge in pixels for Pass 1 filter (default: 1024)",
    )
    parser.add_argument(
        "--train_replicate",
        action="store_true",
        default=False,
        help="Automatically package dataset and launch FLUX.1 LoRA training on Replicate via API",
    )
    parser.add_argument(
        "--train_only",
        default=None,
        type=Path,
        help="Package and launch Replicate API training on an existing curated directory",
    )
    parser.add_argument(
        "--replicate_destination",
        default=None,
        type=str,
        help="Destination Replicate model ('owner/model-name' or 'model-name')",
    )
    parser.add_argument(
        "--trainer",
        default="ostris",
        choices=["ostris", "fast"],
        type=str,
        help=(
            "Replicate trainer ('ostris' for ostris/flux-dev-lora-trainer [default], "
            "'fast' for replicate/fast-flux-trainer)"
        ),
    )
    parser.add_argument(
        "--steps",
        default=3000,
        type=int,
        help="Number of FLUX.1 LoRA training steps on Replicate (default: 3000)",
    )
    parser.add_argument(
        "--lora_rank",
        default=32,
        type=int,
        help="LoRA rank dimension for ostris trainer (default: 32)",
    )
    parser.add_argument(
        "--wait_for_training",
        action="store_true",
        default=False,
        help="Poll Replicate training status until completion",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Tensor batch size for pass 3")

    args = parser.parse_args()

    # Default destination model name if --train_replicate or --train_only is active
    dest_model = args.replicate_destination or f"flux-{args.trigger.lower().replace('_', '-')}-lora"

    # --- STANDALONE AUDIT MODE ---
    if args.audit_only:
        if not args.audit_only.is_dir():
            print(f"❌ Error: Audit directory '{args.audit_only}' does not exist.")
            return 1
        res = audit_directory(args.audit_only, args.trigger)
        return 0 if res["pass_rate"] >= 80.0 else 1

    # --- STANDALONE TRAIN-ONLY MODE ---
    if args.train_only:
        if not args.train_only.is_dir():
            print(f"❌ Error: Training directory '{args.train_only}' does not exist.")
            return 1
        print(f"\n[Packaging]: Validating and bundling '{args.train_only}' for Replicate...")
        success, zip_path, msg = package_replicate_dataset(args.train_only, args.trigger)
        if not success or not zip_path:
            print(f"❌ Error packaging dataset: {msg}")
            return 1
        print(f" -> ✅ {msg}")
        ok_train, _, train_msg = start_replicate_training(
            zip_path=zip_path,
            destination=dest_model,
            trigger_word=args.trigger,
            trainer=args.trainer,
            steps=args.steps,
            lora_rank=args.lora_rank,
            wait=args.wait_for_training,
        )
        if not ok_train:
            print(f"❌ {train_msg}")
            return 1
        print(f"✅ {train_msg}")
        return 0

    # --- STANDALONE RE-CAPTION MODE ---
    if args.recaption_dir:
        if not args.recaption_dir.is_dir():
            print(f"❌ Error: Recaption directory '{args.recaption_dir}' does not exist.")
            return 1
        print("🚀 Running Pre-Flight Checks for Ollama...")
        if not check_ollama_status(args.model):
            print("❌ Pre-flight check failed. Exiting.")
            return 1

        successful, total = recaption_directory(
            args.recaption_dir, args.trigger, model_name=args.model
        )

        audit_directory(args.recaption_dir, args.trigger)

        if (args.package_zip or args.train_replicate) and successful > 0:
            print("\n[Packaging]: Bundling dataset for Replicate FLUX LoRA training...")
            success, zip_path, msg = package_replicate_dataset(
                args.recaption_dir, args.trigger
            )
            if success and zip_path:
                print(f" -> ✅ {msg}")
                if args.train_replicate:
                    ok_train, _, train_msg = start_replicate_training(
                        zip_path=zip_path,
                        destination=dest_model,
                        trigger_word=args.trigger,
                        trainer=args.trainer,
                        steps=args.steps,
                        lora_rank=args.lora_rank,
                        wait=args.wait_for_training,
                    )
                    if not ok_train:
                        print(f" -> ❌ {train_msg}")
                        return 1
                else:
                    print_replicate_instructions(
                        zip_path, args.trigger, steps=args.steps, lora_rank=args.lora_rank
                    )

        return 0 if successful > 0 else 1

    # Ensure required arguments are provided for regular curation
    if not args.input or not args.output:
        print("❌ Error: Both --input and --output are required for image curation.")
        return 1

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

    seen_paths = set()
    image_paths = []
    # Exhaustive coverage of typical DSLR/Smartphone extension casings with case-insensitive dedup
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"]:
        for p in args.input.glob(ext):
            resolved = p.resolve()
            if resolved not in seen_paths and p.is_file():
                seen_paths.add(resolved)
                image_paths.append(p)
    image_paths.sort()

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
            futures = {
                executor.submit(process_pass1, p, args.min_resolution): p for p in image_paths
            }
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

        # Optional Era Window Filtering
        if args.era_window and args.era_window.lower() != "all":
            try:
                if "-" in args.era_window:
                    start_y, end_y = map(int, args.era_window.split("-"))
                else:
                    start_y = end_y = int(args.era_window)
                filtered_era = [
                    m for m in pass1_results if m.year is not None and start_y <= m.year <= end_y
                ]
                print(
                    f" -> Filtered by era window '{args.era_window}': "
                    f"{len(filtered_era)}/{len(pass1_results)} candidates match."
                )
                if filtered_era:
                    pass1_results = filtered_era
                else:
                    print(
                        f"⚠️ Warning: No candidates matched era window '{args.era_window}'. "
                        "Falling back to full candidate set."
                    )
            except ValueError:
                print(
                    f"⚠️ Warning: Invalid era window format '{args.era_window}'. "
                    "Expected YYYY or YYYY-YYYY. Continuing with full candidate set."
                )

        if len(pass1_results) < 30:
            print(" -> Warning: Less than 30 images passed. Portfolio size will be restricted.")

        if not pass1_results:
            return 1

        # --- PASS 2 ---
        print("\n[Pass 2]: Technical Quality Ranking (Optimizing Alpha)")
        pass2_results = process_pass2(
            pass1_results,
            top_n=args.top_n,
            temporal_balance=args.temporal_balance,
            favor_no_glasses=args.favor_no_glasses,
        )
        print(f" -> Top {len(pass2_results)} technically optimal candidates advanced.")

        # --- PASS 3 ---
        print("\n[Pass 3]: Diversity Optimization (Minimizing Correlation)")
        pass3_results = process_pass3(
            pass2_results,
            target_count=args.target_count,
            batch_size=args.batch_size,
            temporal_balance=args.temporal_balance,
        )
        print(f" -> Generated highly diverse final portfolio of size {len(pass3_results)}.")

        # --- PASS 4 ---
        print("\n[Pass 4]: Holistic Set Refinement (Consensus, Lighting, Aesthetics)")
        final_portfolio = process_pass4(
            pass3_results,
            pass2_results,
            model_name=args.model,
            favor_no_glasses=args.favor_no_glasses,
            drop_sunglasses=args.drop_sunglasses,
        )
        print(f" -> Completed final portfolio size: {len(final_portfolio)}.")

        # --- POST PROCESSING ---
        print("\n[Finalizing]: Adaptive scaling and standardized boundary processing")
        preserve_ar = args.preserve_aspect_ratio and not args.crop_square
        if final_portfolio:
            process_and_save_images(
                final_portfolio,
                args.output,
                args.trigger,
                output_format=args.format,
                preserve_aspect_ratio=preserve_ar,
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

        successful_captions = 0
        for file_path, metrics in tqdm(
            zip(output_files, final_portfolio, strict=False),
            total=len(final_portfolio),
            desc="Captioning loop",
        ):
            hand_status = metrics.hand_status if metrics.hands_truncated else None
            if generate_caption(
                file_path,
                args.trigger,
                composition=metrics.composition,
                model_name=args.model,
                hand_status=hand_status,
                year=metrics.year,
            ):
                successful_captions += 1

        print(
            f" -> Generated {successful_captions}/{len(final_portfolio)} standardized captions."
        )

        # Surface automated quality audit on newly generated captions
        audit_directory(args.output, args.trigger)

        # --- REPLICATE DATASET PACKAGING & API TRAINING ---
        if args.package_zip or args.train_replicate:
            print("\n[Packaging]: Bundling dataset for Replicate FLUX LoRA training...")
            success, zip_path, msg = package_replicate_dataset(args.output, args.trigger)
            if success and zip_path:
                print(f" -> ✅ {msg}")
                if args.train_replicate:
                    print("\n[Replicate API]: Launching remote FLUX.1 LoRA training...")
                    ok_train, _, train_msg = start_replicate_training(
                        zip_path=zip_path,
                        destination=dest_model,
                        trigger_word=args.trigger,
                        trainer=args.trainer,
                        steps=args.steps,
                        lora_rank=args.lora_rank,
                        wait=args.wait_for_training,
                    )
                    if ok_train:
                        print(f" -> ✅ {train_msg}")
                    else:
                        print(f" -> ⚠️ Replicate API training warning: {train_msg}")
                else:
                    print_replicate_instructions(
                        zip_path, args.trigger, steps=args.steps, lora_rank=args.lora_rank
                    )
            else:
                print(f" -> ⚠️ Packaging warning: {msg}")

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
