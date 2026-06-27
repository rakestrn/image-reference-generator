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
    Filters out any images that mysteriously corrupted between passes.

    Args:
        metrics_list: List of top N technically ranked image candidates.
        batch_size: Number of images to process simultaneously through the neural network.

    Returns:
        A tuple containing:
            - A 2D numpy array of feature vectors.
            - A synchronized list of ImageQualityMetrics that successfully yielded a vector.
    """
    model, processor, device = get_clip_model()

    valid_metrics = []
    image_list = []

    # Attempt to load and preprocess all candidates
    for m in metrics_list:
        try:
            img = Image.open(m.path).convert("RGB")
            image_list.append(img)
            valid_metrics.append(m)
        except UnidentifiedImageError, OSError, SyntaxError:
            continue

    if not image_list:
        return np.array([]), []

    # Batch Process
    features = []
    with torch.no_grad():
        for i in range(0, len(image_list), batch_size):
            batch_imgs = image_list[i : i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            output = model.get_image_features(**inputs)
            # In transformers >= 4.48, get_image_features returns BaseModelOutputWithPooling
            # where the pooler_output has the projected features.
            image_features = output.pooler_output if hasattr(output, "pooler_output") else output

            # Normalize features for cosine distance
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            features.append(image_features.cpu().numpy())

    return np.vstack(features), valid_metrics


def process_pass3(
    metrics_list: list[ImageQualityMetrics], target_count: int = 30, batch_size: int = 32
) -> list[ImageQualityMetrics]:
    """Pass 3: Diversity Optimization

    Extracts features and greedily selects images that maximize visual diversity
    (calculating average cosine distance). Attempts to satisfy compositional quotas
    that dynamically scale based on the target_count (e.g., ~67% close-ups, ~13% full-body),
    maintaining a fallback escape-hatch if candidates are deficient.

    Args:
        metrics_list: List of top N technically ranked image candidates.
        target_count: Exact number of total images required in the final portfolio.

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

    close_up_target = int(target_count * (20 / 30))
    medium_target = int(target_count * (6 / 30))
    full_body_target = target_count - close_up_target - medium_target
    target_comp = {
        "close-up": close_up_target,
        "medium": medium_target,
        "full-body": full_body_target,
    }
    current_comp = {"close-up": 0, "medium": 0, "full-body": 0}

    # Initialize with the absolute highest technical score
    selected_indices = [0]
    unselected_indices = list(range(1, len(robust_metrics_list)))

    first_comp = robust_metrics_list[0].composition
    if first_comp in current_comp:
        current_comp[first_comp] += 1

    while len(selected_indices) < target_count and unselected_indices:
        # Filter candidates adhering to strict remaining quota
        valid_candidates = []
        for idx in unselected_indices:
            comp = robust_metrics_list[idx].composition
            if comp and current_comp.get(comp, 0) < target_comp.get(comp, 0):
                valid_candidates.append(idx)

        # Escape Hatch: Relax constraints if quota is naturally unachievable
        candidates_to_consider = valid_candidates if valid_candidates else unselected_indices

        best_candidate_idx = -1
        max_min_dist = -1.0

        for idx in candidates_to_consider:
            # Maximizing the minimum distance to any already-selected portfolio image
            min_dist_to_selected = np.min(distances[idx, selected_indices])
            if min_dist_to_selected > max_min_dist:
                max_min_dist = min_dist_to_selected
                best_candidate_idx = idx

        selected_indices.append(best_candidate_idx)
        unselected_indices.remove(best_candidate_idx)

        comp = robust_metrics_list[best_candidate_idx].composition
        if comp in current_comp:
            current_comp[comp] += 1

    return [robust_metrics_list[i] for i in selected_indices]
