import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from .ollama_integration import analyze_lighting
from .pass1_filter import ImageQualityMetrics
from .pass3_diversity import extract_features


def process_pass4(
    selected_metrics: list[ImageQualityMetrics],
    full_pass2_metrics: list[ImageQualityMetrics],
    model_name: str = "gemma4:26b",
) -> list[ImageQualityMetrics]:
    """Pass 4: Holistic Set Refinement

    - Subject Consensus Check: Removes outliers furthest from the median feature vector.
    - Lighting Balancing: Uses Ollama Vision to analyze lighting and enforce a 70/30 split.
    - Aesthetic 'Lift': Replaces the bottom 10% (lowest pass2 total_score) with better ones.
    """
    if not selected_metrics:
        return []

    print("\n   -> [Pass 4.1] Subject Consensus Check & Global Deduplication")

    # Pre-extract MobileNet tensors for the ENTIRE Pass 2 pool
    # so we can mathematically guard substitutions
    all_features, valid_all_metrics = extract_features(full_pass2_metrics)
    feature_map = {
        m.path: f.reshape(1, -1) for m, f in zip(valid_all_metrics, all_features, strict=False)
    }

    robust_metrics = [m for m in selected_metrics if m.path in feature_map]
    if not robust_metrics:
        return selected_metrics

    # Reconstruct features strictly for the portfolio we are dealing with right now
    selected_features = np.vstack([feature_map[m.path] for m in robust_metrics])

    # Calculate median feature vector
    median_vector = np.median(selected_features, axis=0).reshape(1, -1)
    distances = cosine_distances(selected_features, median_vector).flatten()

    # Remove top 5% outliers (e.g., 1-2 images)
    num_outliers = max(1, int(len(robust_metrics) * 0.05))
    outlier_indices = distances.argsort()[-num_outliers:]

    # Create the refined list without outliers
    refined = [m for i, m in enumerate(robust_metrics) if i not in outlier_indices]

    # Get remaining pool to use as replacements
    selected_paths = {m.path for m in refined}
    remaining_pool = [
        m for m in full_pass2_metrics if m.path not in selected_paths and m.path in feature_map
    ]

    def is_duplicate(candidate_path, current_portfolio, threshold=0.05) -> bool:
        """Mathematically prevents grabbing a visually identical image from the trash pile."""
        if candidate_path not in feature_map:
            return True  # fail-safe block

        c_feat = feature_map[candidate_path]
        for p_m in current_portfolio:
            if p_m.path in feature_map:
                dist = cosine_distances(c_feat, feature_map[p_m.path])[0][0]
                if dist < threshold:
                    return True
        return False

    def replace_preserving_comp(
        composition: str,
        pool: list[ImageQualityMetrics],
        current_portfolio: list[ImageQualityMetrics],
    ) -> ImageQualityMetrics | None:
        for i, m in enumerate(pool):
            if m.composition == composition and not is_duplicate(m.path, current_portfolio):
                return pool.pop(i)
        return None

    # Replace removed outliers to maintain count
    for idx in outlier_indices:
        comp = robust_metrics[idx].composition
        replacement = replace_preserving_comp(comp, remaining_pool, refined)
        if replacement:
            refined.append(replacement)
            selected_paths.add(replacement.path)
        else:
            # Fallback to absolute best unused image if quota cannot be fulfilled
            for i, rm in enumerate(remaining_pool):
                if not is_duplicate(rm.path, refined):
                    replacement = remaining_pool.pop(i)
                    refined.append(replacement)
                    selected_paths.add(replacement.path)
                    break

    print("\n   -> [Pass 4.2] Lighting Balancing (70/30 Split via Ollama)")
    indoor_count = 0
    outdoor_count = 0

    for m in tqdm(refined, desc="Classifying lighting conditions"):
        if not m.lighting_condition:
            m.lighting_condition = analyze_lighting(m.path, model_name=model_name)
        if m.lighting_condition == "Outdoor":
            outdoor_count += 1
        else:
            indoor_count += 1

    total = len(refined)
    target_primary = int(total * 0.7)

    primary_state = "Indoor" if indoor_count >= outdoor_count else "Outdoor"
    secondary_state = "Outdoor" if primary_state == "Indoor" else "Indoor"
    primary_count = indoor_count if primary_state == "Indoor" else outdoor_count

    if primary_count > target_primary:
        excess = primary_count - target_primary
        refined.sort(key=lambda x: x.total_score)

        swaps_done = 0
        images_to_remove = []
        for m in refined:
            if m.lighting_condition == primary_state and swaps_done < excess:
                replacement_idx = -1
                for i, rm in enumerate(remaining_pool):
                    # Fast checks first: don't waste 15 seconds on a VLM call if the candidate
                    # is the wrong composition or a mathematical duplicate!
                    if rm.composition != m.composition or is_duplicate(rm.path, refined):
                        continue

                    if not rm.lighting_condition:
                        print(f"      [VLM] Searching fallback pool: analyzing {rm.path.name}...")
                        rm.lighting_condition = analyze_lighting(rm.path, model_name=model_name)

                    if rm.lighting_condition == secondary_state:
                        replacement_idx = i
                        break

                if replacement_idx != -1:
                    replacement = remaining_pool.pop(replacement_idx)
                    images_to_remove.append(m)
                    refined.append(replacement)
                    swaps_done += 1

        for m in images_to_remove:
            refined.remove(m)

    print("\n   -> [Pass 4.3] Aesthetic 'Lift'")
    lift_count = max(1, int(len(refined) * 0.10))
    refined.sort(key=lambda x: x.total_score)

    lift_removals = []
    for i in range(lift_count):
        target = refined[i]
        for j, rm in enumerate(remaining_pool):
            if (
                rm.composition == target.composition
                and rm.total_score > target.total_score
                and not is_duplicate(rm.path, refined)
            ):
                replacement = remaining_pool.pop(j)
                lift_removals.append((target, replacement))
                break

    for target, replacement in lift_removals:
        refined.remove(target)
        refined.append(replacement)

    return refined
