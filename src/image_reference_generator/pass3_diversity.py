
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_distances

from .pass1_filter import ImageQualityMetrics


def extract_features(
    metrics_list: list[ImageQualityMetrics], batch_size: int = 32
) -> tuple[np.ndarray, list[ImageQualityMetrics]]:
    """Extracts high-dimensional visual feature vectors from candidate images using MobileNet.

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
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()

    # Auto-detect Apple Silicon MPS or CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_metrics = []
    tensor_list = []

    # Attempt to load and preprocess all candidates
    for m in metrics_list:
        try:
            img = Image.open(m.path).convert("RGB")
            input_tensor = preprocess(img)
            tensor_list.append(input_tensor)
            valid_metrics.append(m)
        except UnidentifiedImageError, OSError, SyntaxError:
            continue

    if not tensor_list:
        return np.array([]), []

    # Batch Process
    features = []
    with torch.no_grad():
        for i in range(0, len(tensor_list), batch_size):
            batch = torch.stack(tensor_list[i : i + batch_size]).to(device)
            output = model(batch)
            features.append(output.cpu().numpy())

    return np.vstack(features), valid_metrics


def process_pass3(
    metrics_list: list[ImageQualityMetrics], target_count: int = 25
) -> list[ImageQualityMetrics]:
    """Pass 3: Diversity Optimization

    Extracts features and greedily selects images that maximize visual diversity
    (calculating average cosine distance). Attempts to satisfy compositional quotas
    (e.g., 6 full-body shots), maintaining a fallback escape-hatch if candidates are deficient.

    Args:
        metrics_list: List of top N technically ranked image candidates.
        target_count: Exact number of total images required in the final portfolio.

    Returns:
        A list containing precisely `target_count` highly diverse ImageQualityMetrics.
    """
    if not metrics_list:
        return []

    features, robust_metrics_list = extract_features(metrics_list)

    # If the robust list shrank below our target, return whatever we safely have
    if len(robust_metrics_list) <= target_count:
        return robust_metrics_list

    distances = cosine_distances(features)

    target_comp = {"close-up": 10, "medium": 9, "full-body": 6}
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
