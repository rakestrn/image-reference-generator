import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, UnidentifiedImageError

_face_detector = None
_hand_landmarker = None


@contextmanager
def suppress_c_stderr():
    """Redirects OS-level file descriptor 2 (stderr) to devnull to silence C/C++ backend noise."""
    try:
        stderr_fd = 2
        sys.stderr.flush()
        saved_stderr_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, stderr_fd)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
            os.close(devnull)
    except Exception:
        yield


def get_face_detector() -> vision.FaceDetector | None:
    """Retrieves or initializes the process-cached MediaPipe FaceDetector."""
    global _face_detector
    if _face_detector is not None:
        return _face_detector
    model_path = Path(__file__).parent / "blaze_face_short_range.tflite"
    if not model_path.exists():
        return None
    try:
        with suppress_c_stderr():
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(base_options=base_options)
            _face_detector = vision.FaceDetector.create_from_options(options)
        return _face_detector
    except Exception:
        return None


def get_hand_landmarker() -> vision.HandLandmarker | None:
    """Retrieves or initializes the process-cached MediaPipe HandLandmarker."""
    global _hand_landmarker
    if _hand_landmarker is not None:
        return _hand_landmarker
    model_path = Path(__file__).parent / "hand_landmarker.task"
    if not model_path.exists():
        return None
    try:
        with suppress_c_stderr():
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
            _hand_landmarker = vision.HandLandmarker.create_from_options(options)
        return _hand_landmarker
    except Exception:
        return None


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
        lighting_condition: Categorical lighting condition (e.g., 'Indoor', 'Outdoor').
        timestamp: Extracted capture datetime from EXIF or mtime.
        year: Extracted calendar year for temporal stratification.
        head_pose: Head orientation ('frontal', 'three-quarter', 'profile').
        has_sunglasses: True if dark/tinted eyewear obscures eyes and facial identity.
    """

    path: Path
    width: int
    height: int
    face_box: tuple[int, int, int, int] | None  # x, y, w, h
    composition: Literal["close-up", "medium", "full-body"] | None
    sharpness: float = 0.0
    exposure_score: float = 0.0
    face_sharpness: float = 0.0
    contrast: float = 0.0
    colorfulness: float = 0.0
    noise: float = 0.0
    hand_count: int = 0
    hands_truncated: bool = False
    hand_status: str | None = None
    total_score: float = 0.0
    lighting_condition: str | None = None
    feature_vector: np.ndarray | None = None
    timestamp: datetime | None = None
    year: int | None = None
    head_pose: str | None = None
    has_sunglasses: bool = False
    has_face_occlusion: bool = False


def extract_image_timestamp(image_path: Path) -> tuple[datetime | None, int | None]:
    """Extracts capture datetime and calendar year from image EXIF metadata,
    falling back to filesystem modification time (mtime).

    Args:
        image_path: Path to the image file.

    Returns:
        tuple of (datetime | None, year | None).
    """
    try:
        with Image.open(image_path) as pil_img:
            exif = pil_img.getexif()
            dt_str = None
            if exif:
                # 306 = DateTime in root IFD
                dt_str = exif.get(306)
                # 0x8769 = Exif sub-IFD
                try:
                    exif_ifd = exif.get_ifd(0x8769)
                    if exif_ifd:
                        # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized
                        dt_str = exif_ifd.get(36867) or exif_ifd.get(36868) or dt_str
                except Exception:
                    pass
            if dt_str and isinstance(dt_str, str):
                parts = dt_str.strip().split()
                if parts:
                    d_parts = parts[0].replace("-", ":").split(":")
                    if len(d_parts) == 3:
                        y, m, d = int(d_parts[0]), int(d_parts[1]), int(d_parts[2])
                        if 1970 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                            return datetime(y, m, d), y
    except Exception:
        pass

    try:
        mtime = image_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        return dt, dt.year
    except Exception:
        return None, None


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


def load_image(image_path: Path) -> np.ndarray | None:
    """Loads an image file path robustly using OpenCV, falling back to PIL.

    This ensures support for JPEG, PNG, WebP, and other formats across environments.

    Args:
        image_path: Path to the image file to load.

    Returns:
        The loaded image as a BGR numpy array, or None if loading fails or the file is invalid.
    """
    if image_path is None or not isinstance(image_path, Path):
        return None
    if not image_path.is_file():
        return None

    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            with Image.open(image_path) as pil_raw:
                pil_raw = pil_raw.convert("RGB")
                img = cv2.cvtColor(np.array(pil_raw), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


def check_hand_integrity(img_rgb: np.ndarray) -> tuple[int, bool, str | None]:
    """Detects visible hands and evaluates edge truncation using MediaPipe HandLandmarker.

    Checks whether visible hands have their wrist inside the frame while fingers
    are truncated at the frame boundary, which would otherwise teach the LoRA
    amputated limb artifacts.

    Args:
        img_rgb: RGB image numpy array.

    Returns:
        tuple of (hand_count, hands_truncated, hand_status).
    """
    detector = get_hand_landmarker()
    if detector is None:
        return 0, False, None

    try:
        with suppress_c_stderr():
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)

        if not results.hand_landmarks:
            return 0, False, None

        hand_count = len(results.hand_landmarks)
        hands_truncated = False
        edge_margin = 0.02  # 2% edge boundary

        for hand_landmarks in results.hand_landmarks:
            wrist = hand_landmarks[0]
            tips = [hand_landmarks[idx] for idx in [4, 8, 12, 16, 20]]

            wrist_inside = (
                edge_margin < wrist.x < (1.0 - edge_margin)
                and edge_margin < wrist.y < (1.0 - edge_margin)
            )
            tips_at_edge = any(
                t.x <= edge_margin
                or t.x >= (1.0 - edge_margin)
                or t.y <= edge_margin
                or t.y >= (1.0 - edge_margin)
                for t in tips
            )
            if wrist_inside and tips_at_edge:
                hands_truncated = True
                break

        status = (
            "hands partially cropped at image edge"
            if hands_truncated
            else f"{hand_count} hands visible"
        )
        return hand_count, hands_truncated, status
    except Exception:
        return 0, False, None


def is_letterboxed_screenshot(img: np.ndarray) -> bool:
    """Detects mobile screen captures with solid black UI letterbox or pillarbox bars."""
    h, w = img.shape[:2]
    top_n = max(1, int(h * 0.04))
    bot_n = max(1, int(h * 0.04))
    top_black = np.sum(np.mean(img[:top_n, :], axis=(1, 2)) < 18) / top_n
    bot_black = np.sum(np.mean(img[-bot_n:, :], axis=(1, 2)) < 18) / bot_n
    if top_black > 0.5 and bot_black > 0.5:
        return True

    left_n = max(1, int(w * 0.04))
    right_n = max(1, int(w * 0.04))
    left_black = np.sum(np.mean(img[:, :left_n], axis=(0, 2)) < 18) / left_n
    right_black = np.sum(np.mean(img[:, -right_n:], axis=(0, 2)) < 18) / right_n
    return bool(left_black > 0.5 and right_black > 0.5)


def process_pass1(image_path: Path, min_resolution: int = 1024) -> ImageQualityMetrics | None:
    """Evaluates an image through the Initial Liquidity Filter constraints.

    Validates structural integrity, rejects letterboxed mobile screen recordings,
    enforces minimum resolution bounds, and strictly verifies an upright human face
    with valid anatomical keypoint geometry.

    Args:
        image_path: System path to the candidate image.
        min_resolution: Minimum allowed shortest edge in pixels (default: 1024).

    Returns:
        An initialized ImageQualityMetrics instance if the image passes all barriers,
        otherwise None.
    """
    if not _verify_image_integrity(image_path):
        return None

    img = load_image(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if min(h, w) < min_resolution:
        return None

    # Reject letterboxed screenshots and screen recordings
    if is_letterboxed_screenshot(img):
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

    detector = get_face_detector()
    if detector is None:
        return None

    img_rgb = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    try:
        with suppress_c_stderr():
            results = detector.detect(mp_image)
    except Exception:
        return None

    if not results.detections or len(results.detections) != 1:
        return None

    detection = results.detections[0]
    score = detection.categories[0].score if detection.categories else 0
    if score < 0.65:
        return None

    bbox = detection.bounding_box
    rel_x = bbox.origin_x / img_for_detection.shape[1]
    rel_y = bbox.origin_y / img_for_detection.shape[0]
    rel_w = bbox.width / img_for_detection.shape[1]
    rel_h = bbox.height / img_for_detection.shape[0]

    # In portraits and standing/seated human figures, the head must be in the upper 48% of frame.
    # Rejects false detections of glutes, knees, or feet occurring in the lower half of the image.
    if rel_y > 0.48:
        return None

    # Anatomical Keypoint Validation:
    # BlazeFace keypoints: 0=right_eye, 1=left_eye, 2=nose, 3=mouth, 4=right_ear, 5=left_ear
    kps = detection.keypoints
    if not kps or len(kps) < 4:
        return None

    re, le, nose, mouth = kps[0], kps[1], kps[2], kps[3]
    avg_eye_y = 0.5 * (re.y + le.y)
    eye_dist = math.hypot(le.x - re.x, le.y - re.y)

    # Reject if inter-ocular distance is collapsed
    if eye_dist < 0.040:
        return None

    # In an upright face, the mouth must be strictly below the eyes
    if mouth.y <= avg_eye_y + 0.025:
        return None

    # The nose must be vertically positioned between the eyes and the mouth
    if nose.y < avg_eye_y - 0.03 or nose.y > mouth.y + 0.03:
        return None

    x = int(rel_x * w)
    y = int(rel_y * h)
    box_w = int(rel_w * w)
    box_h = int(rel_h * h)

    if box_w <= 0 or box_h <= 0:
        return None

    # Calibrated bounding box area ratios for FLUX framing:
    # >= 4.0% of image area: close-up headshot
    # 2.5% to 4.0%: medium / waist-up shot
    # < 2.5%: full-body / wide environmental framing
    img_area = w * h
    face_area = box_w * box_h
    ratio = face_area / img_area

    if ratio >= 0.040:
        composition = "close-up"
    elif ratio >= 0.025:
        composition = "medium"
    else:
        composition = "full-body"

    # Check visible hands and limb integrity
    hand_count, hands_truncated, hand_status = check_hand_integrity(img_rgb)

    # Estimate head pose angle / orientation and detect dark eyewear from facial keypoints
    head_pose = "frontal"
    has_sunglasses = False
    if detection.keypoints and len(detection.keypoints) >= 3:
        try:
            kp_r_eye = detection.keypoints[0]
            kp_l_eye = detection.keypoints[1]
            kp_nose = detection.keypoints[2]
            dist_r = abs(kp_nose.x - kp_r_eye.x)
            dist_l = abs(kp_nose.x - kp_l_eye.x)
            total_span = dist_r + dist_l
            if total_span > 1e-5:
                ratio = dist_r / total_span
                if ratio < 0.25 or ratio > 0.75:
                    head_pose = "profile"
                elif ratio < 0.40:
                    head_pose = "three-quarter-right"
                elif ratio > 0.60:
                    head_pose = "three-quarter-left"
                else:
                    head_pose = "frontal"

            # Check for dark sunglasses / tinted eyewear
            ih_det, iw_det = img_for_detection.shape[:2]
            rx, ry = int(kp_r_eye.x * iw_det), int(kp_r_eye.y * ih_det)
            lx, ly = int(kp_l_eye.x * iw_det), int(kp_l_eye.y * ih_det)
            nx, ny = int(kp_nose.x * iw_det), int(kp_nose.y * ih_det)
            eye_dist = math.hypot(lx - rx, ly - ry)
            if eye_dist > 5:
                r = max(2, int(eye_dist * 0.22))
                gray_det = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2GRAY)
                r_patch = gray_det[
                    max(0, ry - r) : min(ih_det, ry + r), max(0, rx - r) : min(iw_det, rx + r)
                ]
                l_patch = gray_det[
                    max(0, ly - r) : min(ih_det, ly + r), max(0, lx - r) : min(iw_det, lx + r)
                ]
                skin_patch = gray_det[
                    max(0, ny) : min(ih_det, ny + int(eye_dist * 0.4)),
                    max(0, nx - r) : min(iw_det, nx + r),
                ]
                if r_patch.size > 0 and l_patch.size > 0 and skin_patch.size > 0:
                    eye_mean = 0.5 * (float(np.mean(r_patch)) + float(np.mean(l_patch)))
                    skin_mean = float(np.mean(skin_patch))
                    if skin_mean > 50 and eye_mean < 70 and (eye_mean / skin_mean) < 0.55:
                        has_sunglasses = True
        except Exception:
            head_pose = "frontal"
            has_sunglasses = False

    timestamp, year = extract_image_timestamp(image_path)

    return ImageQualityMetrics(
        path=image_path,
        width=w,
        height=h,
        face_box=(x, y, box_w, box_h),
        composition=composition,
        hand_count=hand_count,
        hands_truncated=hands_truncated,
        hand_status=hand_status,
        timestamp=timestamp,
        year=year,
        head_pose=head_pose,
        has_sunglasses=has_sunglasses,
    )
