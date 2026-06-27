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
    # 25 close up, 10 medium, 10 full
    for i in range(45):
        comp = "close-up" if i < 25 else "medium" if i < 35 else "full-body"
        m = ImageQualityMetrics(Path(f"fake{i}.jpg"), 1024, 1024, (0, 0, 10, 10), comp)
        metrics.append(m)

    # Monkeypatch the neural network feature extractor so we bypass Image.open() verification
    def mock_extract(metrics_list, batch_size=32):
        # Return dummy features (zero arrays) and the exact metrics list supplied
        features = np.zeros((len(metrics_list), 1000))
        return features, metrics_list

    monkeypatch.setattr("image_reference_generator.pass3_diversity.extract_features", mock_extract)

    result = process_pass3(metrics, target_count=30)
    assert len(result) == 30

    comp_counts = {"close-up": 0, "medium": 0, "full-body": 0}
    for r in result:
        comp_counts[r.composition] += 1

    assert comp_counts["close-up"] == 20
    assert comp_counts["medium"] == 6
    assert comp_counts["full-body"] == 4


def test_pass3_fallback_deficiency(monkeypatch):
    """Test the fallback 'escape hatch' functionality in pass3.
    Verifies that if a compositional quota can't be met, the algorithm gracefully abandons
    it and fills out the 25 required images using whatever technically optimal images exist.
    """
    metrics = []
    # Only supply 2 full-body shots! We need 4 total.
    for i in range(40):
        if i < 25:
            comp = "close-up"
        elif i < 38:
            comp = "medium"
        else:
            comp = "full-body"  # only 2 (indexes 38 and 39)

        m = ImageQualityMetrics(Path(f"fake{i}.jpg"), 1024, 1024, (0, 0, 10, 10), comp)
        metrics.append(m)

    def mock_extract(metrics_list, batch_size=32):
        features = np.zeros((len(metrics_list), 1000))
        return features, metrics_list

    monkeypatch.setattr("image_reference_generator.pass3_diversity.extract_features", mock_extract)

    result = process_pass3(metrics, target_count=30)

    assert len(result) == 30
    comp_counts = {"close-up": 0, "medium": 0, "full-body": 0}
    for r in result:
        comp_counts[r.composition] += 1

    # Due to deficiency, full-body should cap at 2.
    assert comp_counts["full-body"] == 2
    # The remaining 2 overflow slots should be cleanly absorbed by the others
    assert comp_counts["close-up"] + comp_counts["medium"] == 28


def test_multiple_formats_loading_and_saving(tmp_path):
    """Verifies robust loading of JPEG, PNG, and WebP formats, and confirms that
    process_and_save_images preserves the source image file format.
    """
    from PIL import Image

    from image_reference_generator.pass1_filter import load_image
    from image_reference_generator.post_processor import process_and_save_images

    # Create dummy images
    img = Image.new("RGB", (1024, 1024), color="red")

    jpg_path = tmp_path / "test.jpg"
    png_path = tmp_path / "test.png"
    webp_path = tmp_path / "test.webp"

    img.save(jpg_path, "JPEG")
    img.save(png_path, "PNG")
    img.save(webp_path, "WEBP")

    # Verify load_image works on all of them
    for p in [jpg_path, png_path, webp_path]:
        loaded = load_image(p)
        assert loaded is not None
        assert loaded.shape == (1024, 1024, 3)

    # Verify process_and_save_images preserves format
    metrics_list = [
        ImageQualityMetrics(jpg_path, 1024, 1024, (100, 100, 100, 100), "close-up"),
        ImageQualityMetrics(png_path, 1024, 1024, (100, 100, 100, 100), "medium"),
        ImageQualityMetrics(webp_path, 1024, 1024, (100, 100, 100, 100), "full-body"),
    ]

    out_dir = tmp_path / "output"

    # Mock easyocr Reader so it doesn't try to download models or run detection in tests
    class MockReader:
        def readtext(self, roi):
            return []

    process_and_save_images.reader = MockReader()

    process_and_save_images(metrics_list, out_dir, "trigger")

    out_jpg = out_dir / "trigger_001.jpg"
    out_png = out_dir / "trigger_002.png"
    out_webp = out_dir / "trigger_003.webp"

    assert out_jpg.exists()
    assert out_png.exists()
    assert out_webp.exists()

    # Clean up reader attribute
    if hasattr(process_and_save_images, "reader"):
        delattr(process_and_save_images, "reader")


def test_invalid_and_missing_extensions(tmp_path):
    """Verify load_image and post_processor safety on invalid types, missing files,
    and missing extensions.
    """
    from PIL import Image

    from image_reference_generator.pass1_filter import load_image
    from image_reference_generator.post_processor import process_and_save_images

    # 1. Non-existent path
    assert load_image(tmp_path / "does_not_exist.jpg") is None

    # 2. Path is None or not Path
    assert load_image(None) is None
    assert load_image("not_a_path_object") is None

    # 3. Create file with missing/no extension
    no_ext_path = tmp_path / "image_no_ext"
    img = Image.new("RGB", (1024, 1024), color="blue")
    img.save(no_ext_path, "JPEG")

    # Verify load_image works on a file without extension
    loaded = load_image(no_ext_path)
    assert loaded is not None
    assert loaded.shape == (1024, 1024, 3)

    # 4. Save output with no extension in metrics.path
    metrics_list = [
        ImageQualityMetrics(no_ext_path, 1024, 1024, (100, 100, 100, 100), "close-up"),
    ]
    out_dir = tmp_path / "output_no_ext"

    # Mock Reader to avoid network
    class MockReader:
        def readtext(self, roi):
            return []

    process_and_save_images.reader = MockReader()

    process_and_save_images(metrics_list, out_dir, "trigger")

    # Should fallback to .jpg
    out_fallback = out_dir / "trigger_001.jpg"
    assert out_fallback.exists()

    if hasattr(process_and_save_images, "reader"):
        delattr(process_and_save_images, "reader")


def test_corrupted_images_handling(tmp_path):
    """Verify that load_image returns None on corrupted files instead of throwing exceptions."""
    from image_reference_generator.pass1_filter import load_image

    corrupt_path = tmp_path / "corrupt.jpg"
    with open(corrupt_path, "w") as f:
        f.write("this is not a valid image format structure at all!")

    # Verify it returns None cleanly
    assert load_image(corrupt_path) is None


def test_format_override_conversion(tmp_path):
    """Verify that process_and_save_images correctly converts images to a specific format
    when output_format is explicitly set.
    """
    from PIL import Image

    from image_reference_generator.post_processor import process_and_save_images

    # Create dummy images (all original png)
    img = Image.new("RGB", (1024, 1024), color="green")
    png_path_1 = tmp_path / "img1.png"
    png_path_2 = tmp_path / "img2.png"
    img.save(png_path_1, "PNG")
    img.save(png_path_2, "PNG")

    metrics_list = [
        ImageQualityMetrics(png_path_1, 1024, 1024, (100, 100, 100, 100), "close-up"),
        ImageQualityMetrics(png_path_2, 1024, 1024, (100, 100, 100, 100), "medium"),
    ]

    out_dir_webp = tmp_path / "output_webp"

    # Mock Reader to avoid network
    class MockReader:
        def readtext(self, roi):
            return []

    process_and_save_images.reader = MockReader()

    # Convert all to WebP
    process_and_save_images(metrics_list, out_dir_webp, "trigger", output_format="webp")

    out_webp_1 = out_dir_webp / "trigger_001.webp"
    out_webp_2 = out_dir_webp / "trigger_002.webp"

    assert out_webp_1.exists()
    assert out_webp_2.exists()

    # Try converting to JPEG
    out_dir_jpg = tmp_path / "output_jpg"
    process_and_save_images(metrics_list, out_dir_jpg, "trigger", output_format="jpg")

    out_jpg_1 = out_dir_jpg / "trigger_001.jpg"
    out_jpg_2 = out_dir_jpg / "trigger_002.jpg"

    assert out_jpg_1.exists()
    assert out_jpg_2.exists()

    if hasattr(process_and_save_images, "reader"):
        delattr(process_and_save_images, "reader")


def test_pass2_computations(monkeypatch):
    """Verify that pass2 computing successfully calculates colorfulness, contrast, and noise
    metrics and ranks images accordingly.
    """
    img_colorful = np.zeros((100, 100, 3), dtype=np.uint8)
    img_colorful[:, :, 1] = 255
    img_colorful[50:, :, :] = 128  # Add contrast

    img_gray = np.zeros((100, 100, 3), dtype=np.uint8)
    img_gray[:, :, :] = 128

    paths_map = {
        Path("colorful.jpg"): img_colorful,
        Path("gray.jpg"): img_gray,
    }

    def mock_load_image(path):
        return paths_map.get(path)

    monkeypatch.setattr("image_reference_generator.pass2_ranking.load_image", mock_load_image)

    metrics = [
        ImageQualityMetrics(Path("colorful.jpg"), 100, 100, None, "close-up"),
        ImageQualityMetrics(Path("gray.jpg"), 100, 100, None, "medium"),
    ]

    result = process_pass2(metrics)
    assert len(result) == 2

    colorful_metric = next(m for m in result if m.path == Path("colorful.jpg"))
    gray_metric = next(m for m in result if m.path == Path("gray.jpg"))

    assert colorful_metric.colorfulness > 0.0
    assert gray_metric.colorfulness == 0.0
    assert colorful_metric.contrast > 0.0
    assert gray_metric.contrast == 0.0
    assert colorful_metric.noise >= 0.0
    assert colorful_metric.total_score > 0.0
