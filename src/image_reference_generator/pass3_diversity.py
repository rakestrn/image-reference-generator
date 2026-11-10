import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_distances
from transformers import CLIPModel, CLIPProcessor

from .pass1_filter import ImageQualityMetrics

_MODEL_CACHE = None
_DEVICE_CACHE = None


def get_clip_model():
    """Initializes and caches the CLIP model and processor."""
    global _MODEL_CACHE, _DEVICE_CACHE
    if _MODEL_CACHE is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        # Auto-detect Apple Silicon MPS or CUDA
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        _MODEL_CACHE = (model, processor)
        _DEVICE_CACHE = device
    return _MODEL_CACHE[0], _MODEL_CACHE[1], _DEVICE_CACHE


def extract_features(
    metrics_list: list[ImageQualityMetrics], batch_size: int = 32
) -> tuple[np.ndarray, list[ImageQualityMetrics]]:
    """Extracts high-dimensional visual feature vectors from candidate images using CLIP.

    Processes images in batched tensors to maximize hardware utilization (GPU/MPS).
    Caches the extracted feature vector directly on each ImageQualityMetrics instance
    so downstream passes (e.g., Pass 4) don't repeat expensive neural network inferences.
    Filters out any images that mysteriously corrupted between passes.

    Args:
        metrics_list: List of top N technically ranked image candidates.
        batch_size: Number of images to process simultaneously through the neural network.

    Returns:
        A tuple containing:
            - A 2D numpy array of feature vectors.
            - A synchronized list of ImageQualityMetrics that successfully yielded a vector.
    """
    # Check if all images already have cached feature vectors
    if metrics_list and all(m.feature_vector is not None for m in metrics_list):
        return np.vstack([m.feature_vector for m in metrics_list]), list(metrics_list)

    model, processor, device = get_clip_model()

    valid_metrics: list[ImageQualityMetrics] = []
    uncached_metrics: list[ImageQualityMetrics] = []
    uncached_images: list[Image.Image] = []

    for m in metrics_list:
        if m.feature_vector is not None:
            valid_metrics.append(m)
            continue

        try:
            img = Image.open(m.path).convert("RGB")
            uncached_images.append(img)
            uncached_metrics.append(m)
            valid_metrics.append(m)
        except (UnidentifiedImageError, OSError, SyntaxError):
            continue

    if not valid_metrics:
        return np.array([]), []

    # Batch Process uncached images
    if uncached_images:
        with torch.no_grad():
            for i in range(0, len(uncached_images), batch_size):
                batch_imgs = uncached_images[i : i + batch_size]
                batch_m = uncached_metrics[i : i + batch_size]
                inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
                output = model.get_image_features(**inputs)
                # In transformers >= 4.48, get_image_features returns BaseModelOutputWithPooling
                # where the pooler_output has the projected features.
                image_features = (
                    output.pooler_output if hasattr(output, "pooler_output") else output
                )

                # Normalize features for cosine distance
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                np_features = image_features.cpu().numpy()

                for m, feat in zip(batch_m, np_features, strict=False):
                    m.feature_vector = feat.reshape(1, -1)

    features = [m.feature_vector for m in valid_metrics if m.feature_vector is not None]
    if not features:
        return np.array([]), []

    return np.vstack(features), valid_metrics


def process_pass3(
    metrics_list: list[ImageQualityMetrics],
    target_count: int = 30,
    batch_size: int = 32,
    temporal_balance: bool = True,
) -> list[ImageQualityMetrics]:
    """Pass 3: Diversity Optimization

    Extracts features and greedily selects images that maximize visual diversity
    (calculating average cosine distance). Enforces compositional quotas
    (35% close-up, 45% medium, 20% full-body), incorporates temporal stratification
    across years when available, and balances facial orientations.

    Args:
        metrics_list: List of top N technically ranked image candidates.
        target_count: Exact number of total images required in the final portfolio.
        batch_size: Batch size for neural network inference.
        temporal_balance: If True, balances representation across chronological years.

    Returns:
        A list containing precisely `target_count` highly diverse ImageQualityMetrics.
    """
    if not metrics_list:
        return []

    features, robust_metrics_list = extract_features(metrics_list, batch_size=batch_size)

    if len(robust_metrics_list) <= target_count:
        return robust_metrics_list

    distances = cosine_distances(features)

    # Pre-filter: Greedily deduplicate very similar images (distance < 0.25)
    # The metrics_list is already sorted by technical quality, so this keeps the best versions
    unique_indices = []
    for i in range(len(robust_metrics_list)):
        is_duplicate = False
        for j in unique_indices:
            # CLIP features are much more clustered than MobileNet;
            # < 0.05 targets near-exact duplicates
            if distances[i, j] < 0.05:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    # Apply duplicate filter
    robust_metrics_list = [robust_metrics_list[i] for i in unique_indices]
    features = features[unique_indices]

    # If the unique list shrank below our target, return whatever we safely have
    if len(robust_metrics_list) <= target_count:
        return robust_metrics_list

    # Recompute distances for the remaining unique pool
    distances = cosine_distances(features)

    # Balanced distribution for FLUX.1 LoRA: 35% close-up / 45% medium / 20% full-body
    close_up_target = int(round(target_count * 0.35))
    medium_target = int(round(target_count * 0.45))
    full_body_target = target_count - close_up_target - medium_target
    target_comp = {
        "close-up": close_up_target,
        "medium": medium_target,
        "full-body": full_body_target,
    }
    current_comp = {"close-up": 0, "medium": 0, "full-body": 0}

    # Chronological Era Stratification quotas
    distinct_years = {m.year for m in robust_metrics_list if m.year is not None}
    if len(distinct_years) > 1 and temporal_balance:
        base_per_year = target_count // len(distinct_years)
        target_per_year = {y: base_per_year for y in distinct_years}
        rem = target_count % len(distinct_years)
        for y in list(distinct_years)[:rem]:
            target_per_year[y] += 1
    else:
        target_per_year = {}

    current_years: dict[int, int] = {}
    current_poses: dict[str, int] = {}

    # Initialize with the absolute highest technical score
    selected_indices = [0]
    unselected_indices = list(range(1, len(robust_metrics_list)))

    first_m = robust_metrics_list[0]
    if first_m.composition in current_comp:
        current_comp[first_m.composition] += 1
    if first_m.year is not None:
        current_years[first_m.year] = current_years.get(first_m.year, 0) + 1
    if first_m.head_pose:
        current_poses[first_m.head_pose] = current_poses.get(first_m.head_pose, 0) + 1

    while len(selected_indices) < target_count and unselected_indices:
        # Tiered Candidate Selection
        valid_comp_and_year = []
        valid_comp_only = []
        valid_year_only = []

        for idx in unselected_indices:
            m = robust_metrics_list[idx]
            comp = m.composition
            has_comp_quota = comp and current_comp.get(comp, 0) < target_comp.get(comp, 0)
            has_year_quota = (
                (m.year is None)
                or (not target_per_year)
                or (current_years.get(m.year, 0) < target_per_year.get(m.year, 0))
            )

            if has_comp_quota and has_year_quota:
                valid_comp_and_year.append(idx)
            elif has_comp_quota:
                valid_comp_only.append(idx)
            elif has_year_quota:
                valid_year_only.append(idx)

        # Escape Hatch Progression:
        # 1. Meets composition quota AND within year quota
        # 2. Meets composition quota only (composition is primary FLUX constraint)
        # 3. Meets year quota only
        # 4. Any remaining candidate
        if valid_comp_and_year:
            candidates_to_consider = valid_comp_and_year
        elif valid_comp_only:
            candidates_to_consider = valid_comp_only
        elif valid_year_only:
            candidates_to_consider = valid_year_only
        else:
            candidates_to_consider = unselected_indices

        best_candidate_idx = -1
        max_min_dist = -1.0

        for idx in candidates_to_consider:
            # Maximizing the minimum distance to any already-selected portfolio image
            min_dist_to_selected = float(np.min(distances[idx, selected_indices]))

            # Slight boost for unrepresented head poses to promote 3D rotational coverage
            cand_pose = robust_metrics_list[idx].head_pose
            if cand_pose and current_poses.get(cand_pose, 0) == 0:
                min_dist_to_selected *= 1.05

            if min_dist_to_selected > max_min_dist:
                max_min_dist = min_dist_to_selected
                best_candidate_idx = idx

        selected_indices.append(best_candidate_idx)
        unselected_indices.remove(best_candidate_idx)

        best_m = robust_metrics_list[best_candidate_idx]
        if best_m.composition in current_comp:
            current_comp[best_m.composition] += 1
        if best_m.year is not None:
            current_years[best_m.year] = current_years.get(best_m.year, 0) + 1
        if best_m.head_pose:
            current_poses[best_m.head_pose] = current_poses.get(best_m.head_pose, 0) + 1

    return [robust_metrics_list[i] for i in selected_indices]
