import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from .ollama_integration import (
    analyze_lighting,
    check_face_occlusions,
    check_sunglasses,
    verify_single_person,
)
from .pass1_filter import ImageQualityMetrics
from .pass3_diversity import extract_features


def process_pass4(
    selected_metrics: list[ImageQualityMetrics],
    full_pass2_metrics: list[ImageQualityMetrics],
    model_name: str = "llava:13b",
    favor_no_glasses: bool = True,
    drop_sunglasses: bool = False,
) -> list[ImageQualityMetrics]:
    """Pass 4: Holistic Set Refinement

    - Subject Consensus Check: Removes outliers furthest from the median feature vector.
    - Single-Subject Verification: Removes and replaces images with companions or multiple people.
    - Eyewear Optimization: Actively replaces sunglasses with bare-eyed/clear-faced photos.
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
        """Mathematically prevents grabbing a visually identical image from the fallback pool."""
        if candidate_path not in feature_map:
            return True  # fail-safe block

        c_feat = feature_map[candidate_path]
        portfolio_feats = [
            feature_map[p_m.path] for p_m in current_portfolio if p_m.path in feature_map
        ]
        if not portfolio_feats:
            return False

        stacked = np.vstack(portfolio_feats)
        dists = cosine_distances(c_feat, stacked)
        return bool(np.any(dists < threshold))

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

    print("\n   -> [Pass 4.2] Single-Subject Verification & Lighting Balancing (via Ollama)")
    indoor_count = 0
    outdoor_count = 0
    multi_subject_removals = []

    for m in tqdm(list(refined), desc="Verifying subjects & lighting"):
        is_single = verify_single_person(m.path, model_name=model_name)
        lighting = analyze_lighting(m.path, model_name=model_name)
        has_sunglasses = check_sunglasses(m.path, model_name=model_name)
        has_occlusion = check_face_occlusions(m.path, model_name=model_name)
        m.lighting_condition = lighting
        m.has_sunglasses = has_sunglasses
        m.has_face_occlusion = has_occlusion
        if not is_single:
            print(
                f"\n      ⚠️ [VLM Filter] Multiple people/companion detected in {m.path.name}. "
                "Flagged for replacement."
            )
            multi_subject_removals.append(m)
        else:
            if lighting == "Outdoor":
                outdoor_count += 1
            else:
                indoor_count += 1

    # Replace multiple-subject images with validated single-person candidates from remaining pool
    for multi_m in multi_subject_removals:
        refined.remove(multi_m)
        for i, rm in enumerate(remaining_pool):
            if not is_duplicate(rm.path, refined):
                is_single = verify_single_person(rm.path, model_name=model_name)
                lighting = analyze_lighting(rm.path, model_name=model_name)
                rm.lighting_condition = lighting
                rm.has_sunglasses = check_sunglasses(rm.path, model_name=model_name)
                rm.has_face_occlusion = check_face_occlusions(rm.path, model_name)
                if is_single:
                    replacement = remaining_pool.pop(i)
                    print(
                        f"      ✅ Replaced {multi_m.path.name} with verified single-person image "
                        f"{replacement.path.name}"
                    )
                    refined.append(replacement)
                    selected_paths.add(replacement.path)
                    if lighting == "Outdoor":
                        outdoor_count += 1
                    else:
                        indoor_count += 1
                    break

    # --- PASS 4.3: FACE INTEGRITY & EYEWEAR OPTIMIZATION ---
    # Prioritizes clear, unobstructed faces over sunglasses, caps, headphones, or turned-away poses
    problem_images = [
        m
        for m in list(refined)
        if (favor_no_glasses and getattr(m, "has_sunglasses", False))
        or getattr(m, "has_face_occlusion", False)
    ]
    if problem_images:
        print("\n   -> [Pass 4.3] Face Integrity Optimization (Favoring Unobstructed Faces)")
        for prob_m in problem_images:
            reason = (
                "sunglasses"
                if getattr(prob_m, "has_sunglasses", False)
                else "face occlusion/turned away"
            )
            replaced = False
            candidates = sorted(
                remaining_pool,
                key=lambda cand: (cand.composition != prob_m.composition, -cand.total_score),
            )
            for rm in candidates:
                if not is_duplicate(rm.path, refined):
                    rm_single = verify_single_person(rm.path, model_name=model_name)
                    rm_lighting = analyze_lighting(rm.path, model_name=model_name)
                    rm_sunglasses = check_sunglasses(rm.path, model_name=model_name)
                    rm_occlusion = check_face_occlusions(rm.path, model_name=model_name)
                    rm.lighting_condition = rm_lighting
                    rm.has_sunglasses = rm_sunglasses
                    rm.has_face_occlusion = rm_occlusion
                    if rm_single and not rm_sunglasses and not rm_occlusion:
                        remaining_pool.remove(rm)
                        refined.remove(prob_m)
                        refined.append(rm)
                        selected_paths.add(rm.path)
                        print(
                            f"      ✅ Replaced {reason} photo {prob_m.path.name} with "
                            f"clean-faced {rm.composition} shot {rm.path.name}"
                        )
                        replaced = True
                        break
            if not replaced and drop_sunglasses and getattr(prob_m, "has_sunglasses", False):
                print(f"      🗑️ Dropped {prob_m.path.name} (strictly dropping sunglasses).")
                refined.remove(prob_m)

    total = len(refined)
    target_primary = int(total * 0.7)

    primary_state = "Indoor" if indoor_count >= outdoor_count else "Outdoor"
    secondary_state = "Outdoor" if primary_state == "Indoor" else "Indoor"
    primary_count = indoor_count if primary_state == "Indoor" else outdoor_count

    if primary_count > target_primary:
        excess = primary_count - target_primary
        refined.sort(key=lambda x: x.total_score)
        # Ensure remaining pool is prioritized by best technical quality
        remaining_pool.sort(key=lambda x: x.total_score, reverse=True)

        swaps_done = 0
        images_to_remove = []
        max_candidates_per_swap = 6

        for m in refined:
            if m.lighting_condition == primary_state and swaps_done < excess:
                replacement_idx = -1
                candidates_checked = 0

                for i, rm in enumerate(remaining_pool):
                    # Fast checks first: skip if wrong composition or mathematical duplicate
                    if rm.composition != m.composition or is_duplicate(rm.path, refined):
                        continue

                    if not rm.lighting_condition:
                        if candidates_checked >= max_candidates_per_swap:
                            # Reached search budget for this swap
                            break
                        candidates_checked += 1
                        print(f"      [VLM] Checking fallback candidate: {rm.path.name}...")
                        is_single = verify_single_person(rm.path, model_name=model_name)
                        lighting = analyze_lighting(rm.path, model_name=model_name)
                        rm.lighting_condition = lighting
                        if not is_single:
                            continue

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

    if not refined:
        return []

    print("\n   -> [Pass 4.4] Aesthetic 'Lift'")
    lift_count = min(len(refined), max(1, int(len(refined) * 0.10)))
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
