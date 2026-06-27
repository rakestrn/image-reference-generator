import math

import cv2
import numpy as np

from .pass1_filter import ImageQualityMetrics, load_image


def compute_contrast(gray: np.ndarray) -> float:
    """Computes RMS Contrast (standard deviation of pixel intensities)."""
    return float(np.std(gray))


def compute_colorfulness(img: np.ndarray) -> float:
    """Computes Hasler and Suesstrunk colorfulness metric."""
    R = img[:, :, 2].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 0].astype(float)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    mean_rg = np.mean(rg)
    std_yb = np.std(yb)
    mean_yb = np.mean(yb)
    std_root = np.sqrt((std_rg**2) + (std_yb**2))
    mean_root = np.sqrt((mean_rg**2) + (mean_yb**2))
    return float(std_root + (0.3 * mean_root))


def compute_noise(gray: np.ndarray) -> float:
    """Estimates high-frequency noise using median filter absolute difference."""
    blurred = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, blurred)
    return float(np.mean(diff))


def process_pass2_single(metrics: ImageQualityMetrics) -> ImageQualityMetrics:
    """Computes technical quality metrics for a single image.

    Calculates sharpness, exposure safety, and facial clarity.
    """
    img = load_image(metrics.path)
    if img is None:
        return metrics

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Metric 1: Sharpness (Variance of Laplacian)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics.sharpness = float(sharpness) if not math.isnan(sharpness) else 0.0

    # Metric 2: Exposure Balance
    # Penalize overexposed and underexposed
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    under = np.sum(hist[0:15])
    over = np.sum(hist[240:256])
    total = gray.size

    # Prevent zero-division if an image is somehow 0x0
    exposure_score = max(0.0, 1.0 - ((under + over) / total)) if total > 0 else 0.0
    metrics.exposure_score = float(exposure_score) if not math.isnan(exposure_score) else 0.0

    # Metric 3: Subject Clarity
    if metrics.face_box:
        x, y, w, h = metrics.face_box
        # Ensure safe bounds
        x_start = max(0, x)
        x_end = min(gray.shape[1], x + w)
        y_start = max(0, y)
        y_end = min(gray.shape[0], y + h)

        face_roi = gray[y_start:y_end, x_start:x_end]

        if face_roi.size > 0:
            face_sharpness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            metrics.face_sharpness = (
                float(face_sharpness) if not math.isnan(face_sharpness) else 0.0
            )
        else:
            metrics.face_sharpness = 0.0
    else:
        metrics.face_sharpness = 0.0

    # Metric 4: Contrast
    metrics.contrast = compute_contrast(gray)

    # Metric 5: Colorfulness
    metrics.colorfulness = compute_colorfulness(img)

    # Metric 6: High-Frequency Noise
    metrics.noise = compute_noise(gray)

    return metrics


def process_pass2(
    metrics_list: list[ImageQualityMetrics], top_n: int = 60
) -> list[ImageQualityMetrics]:
    """Pass 2: Technical Quality Ranking

    Evaluates an array of ImageQualityMetrics candidates, processing the original images
    to compute their technical execution scores (sharpness, exposure safety, facial clarity).
    Provides robust NaN protection during variance extraction.

    Args:
        metrics_list: A list of previously filtered ImageQualityMetrics.
        top_n: The maximum number of highest-scoring images to return.

    Returns:
        A sorted list (descending order of total_score) of the top N candidates.
    """
    if not metrics_list:
        return []

    # For small sets, run sequentially to avoid process overhead and allow testing/mocking
    if len(metrics_list) < 5:
        processed_list = [process_pass2_single(m) for m in metrics_list]
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor

        max_workers = min(32, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            processed_list = list(executor.map(process_pass2_single, metrics_list))

    # Normalize scores across the dataset to fairly combine them
    max_sharpness = max((m.sharpness for m in processed_list), default=1.0)
    if max_sharpness == 0.0:
        max_sharpness = 1.0

    max_face_sharpness = max((m.face_sharpness for m in processed_list), default=1.0)
    if max_face_sharpness == 0.0:
        max_face_sharpness = 1.0

    max_contrast = max((m.contrast for m in processed_list), default=1.0)
    if max_contrast == 0.0:
        max_contrast = 1.0

    max_colorfulness = max((m.colorfulness for m in processed_list), default=1.0)
    if max_colorfulness == 0.0:
        max_colorfulness = 1.0

    max_noise = max((m.noise for m in processed_list), default=1.0)
    if max_noise == 0.0:
        max_noise = 1.0

    for m in processed_list:
        norm_sharpness = m.sharpness / max_sharpness
        norm_face_sharpness = m.face_sharpness / max_face_sharpness
        norm_contrast = m.contrast / max_contrast
        norm_colorfulness = m.colorfulness / max_colorfulness
        norm_noise = m.noise / max_noise

        # Weighted composite score:
        # 0.2 sharpness, 0.3 face_sharpness, 0.15 exposure, 0.15 contrast,
        # 0.1 colorfulness, 0.1 (1.0 - noise penalty)
        m.total_score = (
            0.2 * norm_sharpness
            + 0.3 * norm_face_sharpness
            + 0.15 * m.exposure_score
            + 0.15 * norm_contrast
            + 0.1 * norm_colorfulness
            + 0.1 * (1.0 - norm_noise)
        )

    sorted_metrics = sorted(processed_list, key=lambda m: m.total_score, reverse=True)
    return sorted_metrics[:top_n]
