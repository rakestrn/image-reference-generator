from pathlib import Path

import cv2
from PIL import Image

from .pass1_filter import ImageQualityMetrics, load_image


def process_and_save_images(
    metrics_list: list[ImageQualityMetrics],
    output_dir: Path,
    trigger_word: str,
    output_format: str = "preserve",
    preserve_aspect_ratio: bool = False,
    drop_watermarked: bool = False,
):
    """Adaptive Smart-Scaling and Execution.

    Loops through the highly curated portfolio, utilizing EasyOCR to detect
    and crop any textual watermarks. Scales images, maintaining aspect ratio
    if requested, or square cropping perfectly.
    Uses PIL to encode to pure safetensors-approved JPEG/PNG/WebP formats.

    Args:
        metrics_list: List of the final curated ImageQualityMetrics portfolio.
        output_dir: Absolute path resolving the output target directory.
        trigger_word: Unique identifying string sequentially stamped into the filename.
        output_format: Target output image format ('jpg', 'png', 'webp', or 'preserve').
        preserve_aspect_ratio: If true, bucket resolution matching ~1024^2 area.
        drop_watermarked: If true, drop images containing watermarks entirely.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_idx = 1
    for metrics in metrics_list:
        # Determine output format
        if output_format == "preserve":
            suffix = ""
            if metrics.path:
                suffix = metrics.path.suffix.lower()

            if suffix in [".jpg", ".jpeg"]:
                format_name = "JPEG"
                ext = ".jpg"
            elif suffix == ".png":
                format_name = "PNG"
                ext = ".png"
            elif suffix == ".webp":
                format_name = "WEBP"
                ext = ".webp"
            else:
                format_name = "JPEG"
                ext = ".jpg"
        elif output_format == "jpg":
            format_name = "JPEG"
            ext = ".jpg"
        elif output_format == "png":
            format_name = "PNG"
            ext = ".png"
        elif output_format == "webp":
            format_name = "WEBP"
            ext = ".webp"
        else:
            format_name = "JPEG"
            ext = ".jpg"

        filename = f"{trigger_word}_{valid_idx:03d}{ext}"
        out_path = output_dir / filename

        img = load_image(metrics.path)
        if img is None:
            print(f" ⚠️  Critical: Image {metrics.path.name} is unreadable. Skipping.")
            continue

        h, w = img.shape[:2]

        # --- WATERMARK REMOVAL (TOP & BOTTOM MARGINS) ---
        import easyocr

        if not hasattr(process_and_save_images, "reader"):
            process_and_save_images.reader = easyocr.Reader(["en"])

        margin_h = int(h * 0.15)
        top_roi = img[0:margin_h, 0:w]
        bottom_roi = img[h - margin_h : h, 0:w]

        watermark_boxes_top = []
        for bbox, text, prob in process_and_save_images.reader.readtext(top_roi):
            if prob > 0.3 and len(text.strip()) >= 2:
                watermark_boxes_top.append(bbox)

        watermark_boxes_bottom = []
        for bbox, text, prob in process_and_save_images.reader.readtext(bottom_roi):
            if prob > 0.3 and len(text.strip()) >= 2:
                watermark_boxes_bottom.append(bbox)

        if watermark_boxes_top or watermark_boxes_bottom:
            if drop_watermarked:
                print(f"   -> 💧 Watermark detected in {metrics.path.name}. Dropping entirely.")
                continue

            print(
                f"   -> 💧 Watermark detected in {metrics.path.name}, "
                f"blurring specific text bounding boxes."
            )
            for bbox in watermark_boxes_top:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                min_x = max(0, int(min(xs)))
                max_x = min(w, int(max(xs)))
                min_y = max(0, int(min(ys)))
                max_y = min(margin_h, int(max(ys)))

                pad = 5
                min_x = max(0, min_x - pad)
                max_x = min(w, max_x + pad)
                min_y = max(0, min_y - pad)
                max_y = min(margin_h, max_y + pad)

                roi_to_blur = img[min_y:max_y, min_x:max_x]
                if roi_to_blur.size > 0:
                    img[min_y:max_y, min_x:max_x] = cv2.GaussianBlur(roi_to_blur, (0, 0), 15)

            for bbox in watermark_boxes_bottom:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                min_x = max(0, int(min(xs)))
                max_x = min(w, int(max(xs)))
                min_y = max(0, int(min(ys)))
                max_y = min(margin_h, int(max(ys)))

                abs_min_x = min_x
                abs_max_x = max_x
                abs_min_y = (h - margin_h) + min_y
                abs_max_y = (h - margin_h) + max_y

                pad = 5
                abs_min_x = max(0, abs_min_x - pad)
                abs_max_x = min(w, abs_max_x + pad)
                abs_min_y = max(h - margin_h, abs_min_y - pad)
                abs_max_y = min(h, abs_max_y + pad)

                roi_to_blur = img[abs_min_y:abs_max_y, abs_min_x:abs_max_x]
                if roi_to_blur.size > 0:
                    img[abs_min_y:abs_max_y, abs_min_x:abs_max_x] = cv2.GaussianBlur(
                        roi_to_blur, (0, 0), 15
                    )

        target_area = 1024 * 1024

        if preserve_aspect_ratio:
            # Scale so width * height ~ target_area
            aspect = w / h
            new_h = int((target_area / aspect) ** 0.5)
            new_w = int(new_h * aspect)

            # Snap to nearest multiple of 64
            new_w = max(64, round(new_w / 64) * 64)
            new_h = max(64, round(new_h / 64) * 64)

            img_cropped = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Scale shortest dimension to exactly 1024
            if h < w:
                new_h = 1024
                scale_factor = 1024.0 / h
                new_w = int(round(w * scale_factor))
            else:
                new_w = 1024
                scale_factor = 1024.0 / w
                new_h = int(round(h * scale_factor))

            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Face-centered square crop (1024x1024)
            if metrics.face_box:
                fx, fy, fw, fh = metrics.face_box
                face_center_x = (fx + fw / 2.0) * scale_factor
                face_center_y = (fy + fh / 2.0) * scale_factor
            else:
                face_center_x = new_w / 2.0
                face_center_y = new_h / 2.0

            crop_size = 1024

            if new_w > crop_size:
                crop_start_x = int(face_center_x - (crop_size / 2))
                if crop_start_x < 0:
                    crop_start_x = 0
                elif crop_start_x + crop_size > new_w:
                    crop_start_x = new_w - crop_size

                img_cropped = img_resized[:, crop_start_x : crop_start_x + crop_size]
            elif new_h > crop_size:
                if metrics.face_box:
                    face_top_y = metrics.face_box[1] * scale_factor
                    face_height = metrics.face_box[3] * scale_factor
                    crop_start_y = int(face_top_y - (face_height * 0.5))
                else:
                    crop_start_y = int(face_center_y - (crop_size / 2))

                if crop_start_y < 0:
                    crop_start_y = 0
                elif crop_start_y + crop_size > new_h:
                    crop_start_y = new_h - crop_size

                img_cropped = img_resized[crop_start_y : crop_start_y + crop_size, :]
            else:
                img_cropped = img_resized

            final_h, final_w = img_cropped.shape[:2]
            if final_h != 1024 or final_w != 1024:
                img_cropped = cv2.resize(img_cropped, (1024, 1024), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        if format_name in ("JPEG", "WEBP"):
            pil_img.save(out_path, format_name, quality=95)
        else:
            pil_img.save(out_path, format_name)

        valid_idx += 1
