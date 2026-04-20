import math

import cv2
import numpy as np

from .pass1_filter import ImageQualityMetrics


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

    for metrics in metrics_list:
        img = cv2.imread(str(metrics.path), cv2.IMREAD_COLOR)
        if img is None:
            continue

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
        exposure_score = (
            max(0.0, 1.0 - ((under + over) / total)) if total > 0 else 0.0
        )

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

    # Normalize scores across the dataset to fairly combine them
    max_sharpness = max((m.sharpness for m in metrics_list), default=1.0)
    if max_sharpness == 0.0:
        max_sharpness = 1.0

    max_face_sharpness = max((m.face_sharpness for m in metrics_list), default=1.0)
    if max_face_sharpness == 0.0:
        max_face_sharpness = 1.0

    for m in metrics_list:
        norm_sharpness = m.sharpness / max_sharpness
        norm_face_sharpness = m.face_sharpness / max_face_sharpness
        # Weighted composite score
        m.total_score = 0.3 * norm_sharpness + 0.3 * m.exposure_score + 0.4 * norm_face_sharpness

    sorted_metrics = sorted(metrics_list, key=lambda m: m.total_score, reverse=True)
    return sorted_metrics[:top_n]
