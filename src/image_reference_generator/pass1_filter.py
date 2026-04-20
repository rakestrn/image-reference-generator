from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, UnidentifiedImageError


@dataclass
class ImageQualityMetrics:
    """Stores the metrics computed across the pipeline for a specific candidate image.

    Attributes:
        path: Absolute path to the original raw image.
        width: Native width of the image in pixels.
        height: Native height of the image in pixels.
        face_box: Bounding box of the detected face (x, y, width, height).
        composition: Geometrical classification ('close-up', 'medium', 'full-body').
        sharpness: Structural detail metric (Laplacian variance).
        exposure_score: Balanced exposure metric punishing high-key and low-key clipping.
        face_sharpness: Structural detail isolated to the facial bounding box.
        total_score: Synthesized score weighting the technical features.
    """

    path: Path
    width: int
    height: int
    face_box: tuple[int, int, int, int] | None  # x, y, w, h
    composition: Literal["close-up", "medium", "full-body"] | None
    sharpness: float = 0.0
    exposure_score: float = 0.0
    face_sharpness: float = 0.0
    total_score: float = 0.0


def _verify_image_integrity(path: Path) -> bool:
    """Verifies that the file is a structurally intact image without loading full RGB into RAM.

    Args:
        path: Path to the suspected image file.

    Returns:
        True if the image header and fundamental structure are valid, False otherwise.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except UnidentifiedImageError, OSError, SyntaxError:
        return False


def process_pass1(image_path: Path) -> ImageQualityMetrics | None:
    """Evaluates an image through the Initial Liquidity Filter constraints.

    Validates structural integrity, enforces minimum resolution bounds, and
    strictly searches for precisely one human face using MediaPipe.

    Args:
        image_path: System path to the candidate image.

    Returns:
        An initialized ImageQualityMetrics instance if the image passes all barriers,
        otherwise None.
    """
    if not _verify_image_integrity(image_path):
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    if min(h, w) < 1024:
        return None

    # OOM Prevention: Downscale massive images (e.g. >4000px) purely for face detection
    max_dim = 4000
    scale_factor = 1.0
    img_for_detection = img

    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img_for_detection = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Modern MediaPipe tasks Face Detection
    model_path = Path(__file__).parent / "blaze_face_short_range.tflite"
    if not model_path.exists():
        # Fallback to hair if local model missing (unlikely but safe)
        return None

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceDetectorOptions(base_options=base_options)
    
    with vision.FaceDetector.create_from_options(options) as detector:
        # Convert CV2 image to MediaPipe image
        img_rgb = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        try:
            results = detector.detect(mp_image)
        except Exception:
            return None

        if not results.detections or len(results.detections) != 1:
            return None

        detection = results.detections[0]
        bbox = detection.bounding_box

        # MediaPipe Tasks returns integer pixels in the detection bbox relative to the input image
        # We need to scale these back up to native resolution using scale_factor
        
        rel_x = bbox.origin_x / img_for_detection.shape[1]
        rel_y = bbox.origin_y / img_for_detection.shape[0]
        rel_w = bbox.width / img_for_detection.shape[1]
        rel_h = bbox.height / img_for_detection.shape[0]

        x = int(rel_x * w)
        y = int(rel_y * h)
        box_w = int(rel_w * w)
        box_h = int(rel_h * h)

        if box_w <= 0 or box_h <= 0:
            return None

        # Calculate bounding box area ratio to determine composition
        img_area = w * h
        face_area = box_w * box_h
        ratio = face_area / img_area

        if ratio > 0.15:
            composition = "close-up"
        elif ratio > 0.05:
            composition = "medium"
        else:
            composition = "full-body"

        return ImageQualityMetrics(
            path=image_path,
            width=w,
            height=h,
            face_box=(x, y, box_w, box_h),
            composition=composition,
        )
