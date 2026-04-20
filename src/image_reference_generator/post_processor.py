from pathlib import Path

import cv2
from PIL import Image

from .pass1_filter import ImageQualityMetrics


def process_and_save_images(
    metrics_list: list[ImageQualityMetrics], output_dir: Path, trigger_word: str
):
    """Adaptive Smart-Scaling and Execution.

    Loops through the highly curated portfolio, scaling each image so its
    shortest edge is exactly 1024 pixels. It then extracts an absolute
    perfect 1024x1024 square cropped cleanly across the detected facial bounding box.
    Uses PIL to encode to pure safetensors-approved JPEG.

    Args:
        metrics_list: List of the final curated ImageQualityMetrics portfolio.
        output_dir: Absolute path resolving the output target directory.
        trigger_word: Unique identifying string sequentially stamped into the filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, metrics in enumerate(metrics_list, start=1):
        filename = f"{trigger_word}_{idx:03d}.jpg"
        out_path = output_dir / filename

        img = cv2.imread(str(metrics.path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Scale shortest dimension to exactly 1024
        if h < w:
            new_h = 1024
            scale_factor = 1024.0 / h
            new_w = int(round(w * scale_factor))
        else:
            new_w = 1024
            scale_factor = 1024.0 / w
            new_h = int(round(h * scale_factor))

        # INTER_AREA provides the highest quality downsampling
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

        # Absolute mathematical boundaries enforcing strict output dimension sizes
        if new_w > crop_size:
            crop_start_x = int(face_center_x - (crop_size / 2))
            if crop_start_x < 0:
                crop_start_x = 0
            elif crop_start_x + crop_size > new_w:
                crop_start_x = new_w - crop_size

            img_cropped = img_resized[:, crop_start_x : crop_start_x + crop_size]
        elif new_h > crop_size:
            crop_start_y = int(face_center_y - (crop_size / 2))
            if crop_start_y < 0:
                crop_start_y = 0
            elif crop_start_y + crop_size > new_h:
                crop_start_y = new_h - crop_size

            img_cropped = img_resized[crop_start_y : crop_start_y + crop_size, :]
        else:
            # Already exactly 1024x1024
            img_cropped = img_resized

        # Reclaim exact dimensions in case of sub-pixel floating artifacts
        final_h, final_w = img_cropped.shape[:2]
        if final_h != 1024 or final_w != 1024:
            img_cropped = cv2.resize(img_cropped, (1024, 1024), interpolation=cv2.INTER_AREA)

        # Strip BGR channel mapping required by CV2 prior to PIL save pipeline
        img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(out_path, "JPEG", quality=95)
