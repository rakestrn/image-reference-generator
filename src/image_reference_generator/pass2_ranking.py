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

    Downscales images to a canonical reference dimension (max 1024px) before
    computing metrics to eliminate the megapixel resolution bias where modern
    48MP smartphones artificially crush older 8-12MP cameras.
    """
    img = load_image(metrics.path)
    if img is None:
        return metrics

    # Canonical Resolution Normalization (1024px)
    CANONICAL_DIM = 1024
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > CANONICAL_DIM:
        scale = CANONICAL_DIM / max_dim
        norm_w = max(1, int(round(w * scale)))
        norm_h = max(1, int(round(h * scale)))
        img_eval = cv2.resize(img, (norm_w, norm_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        img_eval = img

    gray = cv2.cvtColor(img_eval, cv2.COLOR_BGR2GRAY)

    # Metric 1: Sharpness (Variance of Laplacian at canonical scale)
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

    # Metric 3: Subject Clarity at canonical scale
    if metrics.face_box:
        x, y, w_box, h_box = metrics.face_box
        x_s = max(0, int(round(x * scale)))
        y_s = max(0, int(round(y * scale)))
        bw_s = int(round(w_box * scale))
        bh_s = int(round(h_box * scale))

        x_start = max(0, x_s)
        x_end = min(gray.shape[1], x_s + bw_s)
        y_start = max(0, y_s)
        y_end = min(gray.shape[0], y_s + bh_s)

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
    metrics.colorfulness = compute_colorfulness(img_eval)

    # Metric 6: High-Frequency Noise
    metrics.noise = compute_noise(gray)

    return metrics


def process_pass2(
    metrics_list: list[ImageQualityMetrics],
    top_n: int = 60,
    temporal_balance: bool = True,
    favor_no_glasses: bool = True,
) -> list[ImageQualityMetrics]:
    """Pass 2: Technical Quality Ranking with Scale-Invariance & Temporal Stratification

    Evaluates an array of ImageQualityMetrics candidates at a standardized canonical resolution,
    computes robust normalized composite scores, and applies stratified temporal sampling
    to prevent modern photos from crowding out earlier chronological eras.

    Args:
        metrics_list: A list of previously filtered ImageQualityMetrics.
        top_n: The maximum number of highest-scoring images to return.
        temporal_balance: If True, stratifies candidates across chronological years.
        favor_no_glasses: If True, penalizes photos with sunglasses to prioritize clear
            facial features.

    Returns:
        A list of the top N candidates, balanced across temporal bins.
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

    # Helper for robust 95th-percentile ceiling normalization
    def _calc_ceiling(values: list[float]) -> float:
        valid = [v for v in values if v > 0 and not math.isnan(v)]
        if not valid:
            return 1.0
        p95 = float(np.percentile(valid, 95))
        return p95 if p95 > 0.0 else max(valid)

    ceil_sharpness = _calc_ceiling([m.sharpness for m in processed_list])
    ceil_face_sharpness = _calc_ceiling([m.face_sharpness for m in processed_list])
    ceil_contrast = _calc_ceiling([m.contrast for m in processed_list])
    ceil_colorfulness = _calc_ceiling([m.colorfulness for m in processed_list])
    ceil_noise = _calc_ceiling([m.noise for m in processed_list])

    for m in processed_list:
        norm_sharpness = min(1.0, m.sharpness / ceil_sharpness)
        norm_face_sharpness = min(1.0, m.face_sharpness / ceil_face_sharpness)
        norm_contrast = min(1.0, m.contrast / ceil_contrast)
        norm_colorfulness = min(1.0, m.colorfulness / ceil_colorfulness)
        norm_noise = min(1.0, m.noise / ceil_noise)

        # Weighted composite score:
        # 0.15 sharpness, 0.35 face_sharpness, 0.15 exposure, 0.15 contrast,
        # 0.1 colorfulness, 0.1 (1.0 - noise penalty)
        m.total_score = (
            0.15 * norm_sharpness
            + 0.35 * norm_face_sharpness
            + 0.15 * m.exposure_score
            + 0.15 * norm_contrast
            + 0.1 * norm_colorfulness
            + 0.1 * (1.0 - norm_noise)
        )
        # Head pose prioritization: reward frontal and 3/4 face angles for direct gaze lock
        if m.head_pose in ("frontal", "three-quarter-left", "three-quarter-right"):
            m.total_score += 0.05
        elif m.head_pose == "profile":
            m.total_score -= 0.05

        # Limb integrity barrier: penalize truncated hands at image boundaries
        if m.hands_truncated:
            m.total_score -= 0.05

        # Eyewear barrier: penalize sunglasses to favor clear uncovered facial likeness
        if favor_no_glasses and getattr(m, "has_sunglasses", False):
            m.total_score -= 0.15

    sorted_metrics = sorted(processed_list, key=lambda m: m.total_score, reverse=True)

    # --- COMPOSITION & TEMPORAL STRATIFICATION ---
    # Ensure top_n pool has ample candidates across all 3 compositions (close-up, medium, full-body)
    # so Pass 3 diversity optimization is never starved of facial close-ups.
    comp_bins: dict[str, list[ImageQualityMetrics]] = {
        "close-up": [],
        "medium": [],
        "full-body": [],
    }
    for m in sorted_metrics:
        if m.composition in comp_bins:
            comp_bins[m.composition].append(m)
        else:
            comp_bins["medium"].append(m)

    selected: list[ImageQualityMetrics] = []
    selected_paths = set()

    # If compositions are present, seed top_n with target quotas (35% close, 45% med, 20% full)
    if any(len(comp_bins[c]) > 0 for c in comp_bins) and top_n >= 10:
        targets = {
            "close-up": max(1, int(top_n * 0.35)),
            "medium": max(1, int(top_n * 0.45)),
            "full-body": max(1, int(top_n * 0.20)),
        }
        for comp, target_count in targets.items():
            for m in comp_bins[comp][:target_count]:
                if len(selected) < top_n and m.path not in selected_paths:
                    selected.append(m)
                    selected_paths.add(m.path)

    # Fill remaining slots using temporal balance or highest total_score
    remaining = [m for m in sorted_metrics if m.path not in selected_paths]

    if temporal_balance and remaining:
        year_bins: dict[int | None, list[ImageQualityMetrics]] = {}
        for m in remaining:
            year_bins.setdefault(m.year, []).append(m)

        if len(year_bins) > 1:
            for y in year_bins:
                year_bins[y].sort(key=lambda m: m.total_score, reverse=True)

            slots_needed = top_n - len(selected)
            target_per_bin = max(1, slots_needed // len(year_bins))

            for b_metrics in year_bins.values():
                for m in b_metrics[:target_per_bin]:
                    if len(selected) < top_n and m.path not in selected_paths:
                        selected.append(m)
                        selected_paths.add(m.path)

    # Fill any remaining slots by absolute highest total_score
    remaining_final = [m for m in sorted_metrics if m.path not in selected_paths]
    slots_still_needed = top_n - len(selected)
    if slots_still_needed > 0:
        selected.extend(remaining_final[:slots_still_needed])

    return sorted(selected, key=lambda m: m.total_score, reverse=True)
