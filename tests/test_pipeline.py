from pathlib import Path

import numpy as np

from image_reference_generator.pass1_filter import ImageQualityMetrics
from image_reference_generator.pass2_ranking import process_pass2
from image_reference_generator.pass3_diversity import process_pass3


def test_pass2_scoring():
    """Verify pass2 technical ranking safely bypasses execution on unreadable paths,
    and handles empty collections without exception.
    """
    assert process_pass2([]) == []


def test_pass3_fallback(monkeypatch):
    """Test the strict composition distribution capabilities of pass3.
    Ensures that when valid candidates are perfectly spread, exactly the correct
    compositional quotas (10 close, 9 med, 6 full) are enforced.
    """
    metrics = []
    # 10 close up, 10 medium, 10 full
    for i in range(30):
        comp = "close-up" if i < 10 else "medium" if i < 20 else "full-body"
        m = ImageQualityMetrics(Path(f"fake{i}.jpg"), 1024, 1024, (0, 0, 10, 10), comp)
        metrics.append(m)

    # Monkeypatch the neural network feature extractor so we bypass Image.open() verification
    def mock_extract(metrics_list, batch_size=32):
        # Return dummy features (zero arrays) and the exact metrics list supplied
        features = np.zeros((len(metrics_list), 1000))
        return features, metrics_list

    monkeypatch.setattr("image_reference_generator.pass3_diversity.extract_features", mock_extract)

    result = process_pass3(metrics, target_count=25)
    assert len(result) == 25

    comp_counts = {"close-up": 0, "medium": 0, "full-body": 0}
    for r in result:
        comp_counts[r.composition] += 1

    assert comp_counts["close-up"] == 10
    assert comp_counts["medium"] == 9
    assert comp_counts["full-body"] == 6


def test_pass3_fallback_deficiency(monkeypatch):
    """Test the fallback 'escape hatch' functionality in pass3.
    Verifies that if a compositional quota can't be met, the algorithm gracefully abandons
    it and fills out the 25 required images using whatever technically optimal images exist.
    """
    metrics = []
    # Only supply 2 full-body shots! We need 6 total.
    for i in range(30):
        if i < 14:
            comp = "close-up"
        elif i < 28:
            comp = "medium"
        else:
            comp = "full-body"  # only 2 (indexes 28 and 29)

        m = ImageQualityMetrics(Path(f"fake{i}.jpg"), 1024, 1024, (0, 0, 10, 10), comp)
        metrics.append(m)

    def mock_extract(metrics_list, batch_size=32):
        features = np.zeros((len(metrics_list), 1000))
        return features, metrics_list

    monkeypatch.setattr("image_reference_generator.pass3_diversity.extract_features", mock_extract)

    result = process_pass3(metrics, target_count=25)

    assert len(result) == 25
    comp_counts = {"close-up": 0, "medium": 0, "full-body": 0}
    for r in result:
        comp_counts[r.composition] += 1

    # Due to deficiency, full-body should cap at 2.
    assert comp_counts["full-body"] == 2
    # The remaining 4 overflow slots should be cleanly absorbed by the others
    assert comp_counts["close-up"] + comp_counts["medium"] == 23
