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
    FLUX composition quotas (35% close-up, 45% medium, 20% full-body) are enforced.
    """
    metrics = []
    # 20 close-up, 20 medium, 10 full-body (total 50)
    for i in range(50):
        comp = "close-up" if i < 20 else "medium" if i < 40 else "full-body"
        m = ImageQualityMetrics(Path(f"fake{i}.jpg"), 1024, 1024, (0, 0, 10, 10), comp)
        metrics.append(m)

    # Monkeypatch the neural network feature extractor so we bypass Image.open() verification
    def mock_extract(metrics_list, batch_size=32):
        features = np.zeros((len(metrics_list), 1000))
        return features, metrics_list

    monkeypatch.setattr("image_reference_generator.pass3_diversity.extract_features", mock_extract)

    result = process_pass3(metrics, target_count=30)
    assert len(result) == 30

    comp_counts = {"close-up": 0, "medium": 0, "full-body": 0}
    for r in result:
        comp_counts[r.composition] += 1

    assert comp_counts["close-up"] == 10
    assert comp_counts["medium"] == 14
    assert comp_counts["full-body"] == 6


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

    # Verify process_and_save_images defaults to jpg
    metrics_list = [
        ImageQualityMetrics(jpg_path, 1024, 1024, (100, 100, 100, 100), "close-up"),
        ImageQualityMetrics(png_path, 1024, 1024, (100, 100, 100, 100), "medium"),
        ImageQualityMetrics(webp_path, 1024, 1024, (100, 100, 100, 100), "full-body"),
    ]

    out_dir_default = tmp_path / "output_default"

    # Mock easyocr Reader so it doesn't try to download models or run detection in tests
    class MockReader:
        def readtext(self, roi):
            return []

    process_and_save_images.reader = MockReader()

    # Default should convert everything to .jpg
    process_and_save_images(metrics_list, out_dir_default, "trigger")

    assert (out_dir_default / "trigger_001.jpg").exists()
    assert (out_dir_default / "trigger_002.jpg").exists()
    assert (out_dir_default / "trigger_003.jpg").exists()

    # Verify output_format='preserve' preserves original formats
    out_dir_preserve = tmp_path / "output_preserve"
    process_and_save_images(metrics_list, out_dir_preserve, "trigger", output_format="preserve")

    assert (out_dir_preserve / "trigger_001.jpg").exists()
    assert (out_dir_preserve / "trigger_002.png").exists()
    assert (out_dir_preserve / "trigger_003.webp").exists()

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


def test_caption_cleaning_and_formatting():
    """Verify clean_and_format_caption normalizes model outputs to tagless FLUX prose."""
    from image_reference_generator.ollama_integration import clean_and_format_caption

    lead = "ohwx woman, a close-up shot"
    trigger = "ohwx woman"

    # 1. Output containing unwanted Tags: line and bullet points
    input_with_tags = (
        "ohwx woman, a close-up shot looking directly at the camera with a subtle calm expression. "
        "She is wearing a high-neck dark wool sweater against a soft studio backdrop. "
        "Hands are resting relaxed out of frame with soft shallow depth of field.\n"
        "Tags: close up, wool sweater, studio, soft lighting, calm"
    )
    ok, formatted, reason = clean_and_format_caption(input_with_tags, trigger, lead)
    assert ok is True
    assert formatted.startswith(lead)
    assert "Tags:" not in formatted
    assert "wool sweater" in formatted

    # 2. Conversational preamble, code fences, and bold text
    messy_text = (
        "Sure, here is the caption:\n```\n"
        "A waist-up photo of a woman in a **blue silk blouse** outdoors in a sunlit garden. "
        "Hands are resting gently on the wooden fence. The camera captures sharp focus "
        "with creamy background bokeh.\n```"
    )
    ok, formatted, _ = clean_and_format_caption(messy_text, trigger, lead)
    assert ok is True
    assert formatted.startswith(lead)
    assert "**" not in formatted
    assert "Tags:" not in formatted

    # 3. Too short
    too_short = "A photo."
    ok, _, reason = clean_and_format_caption(too_short, trigger, lead)
    assert ok is False
    assert "short" in reason.lower()

    # 4. Empty
    ok, _, reason = clean_and_format_caption("", trigger, lead)
    assert ok is False
    assert "empty" in reason.lower()


def test_generate_caption_with_validation_and_retry(monkeypatch, tmp_path):
    """Verify generate_caption retries on malformed output and writes formatted natural prose."""
    from PIL import Image

    from image_reference_generator.ollama_integration import generate_caption

    img = Image.new("RGB", (1024, 1024), color="blue")
    img_path = tmp_path / "test_img.jpg"
    img.save(img_path, "JPEG")

    # Mock Ollama API responses: first attempt returns malformed output, second attempt succeeds
    call_count = 0

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    def mock_post(url, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Malformed (too short)
            return MockResponse({"response": "Too short"})
        else:
            # Valid FLUX natural prose
            return MockResponse({
                "response": (
                    "ohwx woman, a close-up shot looking slightly away with a warm smile. "
                    "She is wearing a charcoal blazer over a white crewneck shirt. "
                    "Hands are resting gently out of frame under soft diffused lighting "
                    "with shallow depth of field."
                )
            })

    monkeypatch.setattr("requests.post", mock_post)

    success = generate_caption(img_path, "ohwx woman", composition="close-up")
    assert success is True
    assert call_count == 2

    # Check written file
    txt_path = img_path.with_suffix(".txt")
    assert txt_path.exists()
    content = txt_path.read_text(encoding="utf-8")
    assert content.startswith("ohwx woman, a close-up shot")
    assert "Tags:" not in content


def test_pass4_feature_caching_and_vectorized_dedup(monkeypatch):
    """Verify pass4 reuses cached feature vectors and optimizes lighting balancing."""
    from image_reference_generator.pass4_refinement import process_pass4

    # Create dummy metrics with pre-cached feature vectors
    metrics = []
    for i in range(10):
        m = ImageQualityMetrics(
            Path(f"img_{i}.jpg"),
            1024,
            1024,
            (10, 10, 100, 100),
            "close-up",
            total_score=float(i + 1),
            lighting_condition="Indoor" if i < 8 else "Outdoor",
        )
        # Assign distinct feature vectors
        vec = np.zeros((1, 10))
        vec[0, i % 10] = 1.0
        m.feature_vector = vec
        metrics.append(m)

    selected = metrics[:6]
    full_pool = metrics

    # Mock analyze_lighting so it's fast
    def mock_analyze(path, model_name="llava:13b"):
        return "Outdoor"

    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.analyze_lighting", mock_analyze
    )

    refined = process_pass4(selected, full_pool, model_name="llava:13b")
    assert len(refined) == len(selected)


def test_aspect_ratio_bucketing_snapped_to_64(tmp_path):
    """Verify that process_and_save_images snaps both dimensions to multiples of 64
    by default under multi-aspect ratio bucketing.
    """
    from PIL import Image

    from image_reference_generator.post_processor import process_and_save_images

    # 1600x900 landscape image
    img_landscape = Image.new("RGB", (1600, 900), color="blue")
    img_path = tmp_path / "landscape.jpg"
    img_landscape.save(img_path, "JPEG")

    metrics_list = [
        ImageQualityMetrics(img_path, 1600, 900, (200, 200, 100, 100), "medium"),
    ]
    out_dir = tmp_path / "output_arb"

    class MockReader:
        def readtext(self, roi):
            return []

    process_and_save_images.reader = MockReader()

    # Default preserve_aspect_ratio is True
    process_and_save_images(metrics_list, out_dir, "tok")

    exported = out_dir / "tok_001.jpg"
    assert exported.exists()

    with Image.open(exported) as result_img:
        ew, eh = result_img.size
        # Must both snap to multiples of 64
        assert ew % 64 == 0, f"Width {ew} is not a multiple of 64"
        assert eh % 64 == 0, f"Height {eh} is not a multiple of 64"
        # Total area should be approximately 1024*1024 (~1M pixels)
        area = ew * eh
        assert 800000 <= area <= 1300000, f"Area {area} is far from target 1024^2"

    if hasattr(process_and_save_images, "reader"):
        delattr(process_and_save_images, "reader")


def test_replicate_package_dataset(tmp_path):
    """Verify package_replicate_dataset validates image/caption pairs and creates a zip archive."""
    import zipfile

    from PIL import Image

    from image_reference_generator.main import package_replicate_dataset

    # 1. Test failure on empty folder
    ok, zip_path, msg = package_replicate_dataset(tmp_path, "tok")
    assert ok is False
    assert "No curated images found" in msg

    # Create dummy images
    img = Image.new("RGB", (100, 100), color="white")
    img1_path = tmp_path / "tok_001.jpg"
    img2_path = tmp_path / "tok_002.jpg"
    img.save(img1_path, "JPEG")
    img.save(img2_path, "JPEG")

    # 2. Test failure when caption files are missing
    ok, zip_path, msg = package_replicate_dataset(tmp_path, "tok")
    assert ok is False
    assert "Missing caption files" in msg

    # Create one valid caption, one missing trigger word
    txt1_path = tmp_path / "tok_001.txt"
    txt2_path = tmp_path / "tok_002.txt"
    txt1_path.write_text("tok woman, a close-up shot looking forward.", encoding="utf-8")
    txt2_path.write_text("A woman looking forward without trigger.", encoding="utf-8")

    # 3. Test failure when trigger word is missing in caption
    ok, zip_path, msg = package_replicate_dataset(tmp_path, "tok")
    assert ok is False
    assert "Trigger word" in msg

    # Fix caption 2
    txt2_path.write_text("tok woman, a waist-up shot in garden.", encoding="utf-8")

    # 4. Successful packaging
    ok, zip_path, msg = package_replicate_dataset(tmp_path, "tok")
    assert ok is True
    assert zip_path is not None
    assert zip_path.exists()
    assert "Successfully packaged 2" in msg

    # Check zip contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "tok_001.jpg" in names
        assert "tok_001.txt" in names
        assert "tok_002.jpg" in names
        assert "tok_002.txt" in names


def test_hand_integrity_and_limb_penalty(monkeypatch):
    """Verify hand integrity checks and Pass 2 score penalty on truncated hands."""
    from image_reference_generator.pass1_filter import check_hand_integrity
    from image_reference_generator.pass2_ranking import process_pass2

    # Verify helper runs safely without crash even on blank images
    blank_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    count, truncated, status = check_hand_integrity(blank_rgb)
    assert count == 0
    assert truncated is False

    # Test that Pass 2 penalizes truncated hands
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 120
    img[20:60, 20:60, :] = 200

    def mock_load_image(path):
        return img

    monkeypatch.setattr("image_reference_generator.pass2_ranking.load_image", mock_load_image)

    m1 = ImageQualityMetrics(
        Path("intact.jpg"), 100, 100, (20, 20, 40, 40), "close-up", hands_truncated=False
    )
    m2 = ImageQualityMetrics(
        Path("truncated.jpg"), 100, 100, (20, 20, 40, 40), "close-up", hands_truncated=True
    )

    result = process_pass2([m1, m2])
    score_intact = next(m.total_score for m in result if m.path == Path("intact.jpg"))
    score_truncated = next(m.total_score for m in result if m.path == Path("truncated.jpg"))

    assert score_intact > score_truncated


def test_audit_caption_valid_and_violations():
    """Verify audit_caption flags duplicate trigger words, banned cliches, missing hands, etc."""
    from image_reference_generator.ollama_integration import audit_caption

    trigger = "tok"

    # 1. Valid caption
    valid = (
        "tok woman, a close-up shot looking slightly away with a calm smile. "
        "She is wearing a charcoal wool sweater against a soft studio backdrop. "
        "Her hands are resting relaxed out of frame with shallow depth of field."
    )
    ok, violations = audit_caption(valid, trigger)
    assert ok is True
    assert len(violations) == 0

    # 2. Duplicate trigger word
    dup_trigger = (
        "tok woman, a close-up shot, tok is looking slightly away with a calm smile. "
        "She is wearing a wool sweater and her hands are out of frame."
    )
    ok, violations = audit_caption(dup_trigger, trigger)
    assert ok is False
    assert any("Duplicate trigger word" in v for v in violations)

    # 3. Banned clichés
    cliche_caption = (
        "tok woman, a waist-up shot looking to her right with a contemplative expression. "
        "She is wearing a brown dress that accentuates her figure. "
        "Her hands are out of frame, adding a sense of mystery to the image."
    )
    ok, violations = audit_caption(cliche_caption, trigger)
    assert ok is False
    assert any("accentuates figure/physique" in v for v in violations)
    assert any("a sense of mystery" in v for v in violations)

    # 4. Missing hands
    no_hands = (
        "tok woman, a full-body shot standing in a modern art gallery under soft gallery lights. "
        "She is wearing an elegant silk evening gown and matching stiletto heels with sharp focus."
    )
    ok, violations = audit_caption(no_hands, trigger)
    assert ok is False
    assert any("hand placement" in v.lower() for v in violations)


def test_sanitize_caption_deterministic():
    """Verify sanitize_caption_deterministic strips banned clichés and guarantees hand mention."""
    from image_reference_generator.ollama_integration import (
        audit_caption,
        sanitize_caption_deterministic,
    )

    trigger = "tok"
    lead = "tok woman, a waist-up shot"
    cliche_caption = (
        "tok woman, a waist-up shot looking to her right with a contemplative expression. "
        "She is wearing a brown dress that accentuates her figure. "
        "The setting appears to be an outdoor area with a blurred background, "
        "adding a sense of mystery."
    )
    sanitized = sanitize_caption_deterministic(cliche_caption, trigger, lead)
    assert "accentuates her figure" not in sanitized
    assert "adding a sense of mystery" not in sanitized
    assert "Hands are out of frame." in sanitized
    ok, _ = audit_caption(sanitized, trigger)
    assert ok is True


def test_audit_directory_and_recaption_directory(tmp_path, monkeypatch):
    """Verify audit_directory computes pass rates and recaption_directory updates files in place."""
    from PIL import Image

    from image_reference_generator.ollama_integration import (
        audit_directory,
        recaption_directory,
    )

    trigger = "testtok"
    # Create 2 dummy images with captions
    img = Image.new("RGB", (100, 100), color="blue")
    img1 = tmp_path / f"{trigger}_001.jpg"
    img2 = tmp_path / f"{trigger}_002.jpg"
    img.save(img1, "JPEG")
    img.save(img2, "JPEG")

    # One valid caption, one flawed
    txt1 = img1.with_suffix(".txt")
    txt2 = img2.with_suffix(".txt")
    txt1.write_text(
        f"{trigger} woman, a close-up shot looking at camera with a smile. "
        "She is wearing a black shirt with hands out of frame in a bright room.",
        encoding="utf-8",
    )
    txt2.write_text(
        f"{trigger} woman, a waist-up shot, {trigger} woman looking away in dress "
        "that accentuates her figure.",
        encoding="utf-8",
    )

    report = audit_directory(tmp_path, trigger)
    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["pass_rate"] == 50.0

    # Mock Ollama generation for recaptioning
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    def mock_post(url, json=None, timeout=None):
        return MockResponse({
            "response": (
                f"{trigger} woman, a waist-up shot looking to the side with a relaxed expression. "
                "She is wearing a tailored linen blazer in a sunlit studio with hands out of frame."
            )
        })

    monkeypatch.setattr("requests.post", mock_post)

    success, total = recaption_directory(tmp_path, trigger, model_name="qwen2.5vl:32b")
    assert total == 2
    assert success == 2

    # Verify both pass audit now
    new_report = audit_directory(tmp_path, trigger)
    assert new_report["passed"] == 2
    assert new_report["pass_rate"] == 100.0


def test_complex_real_world_caption_cleaning():
    """Verify clean_and_format_caption strips complex layered prefixes like
    'is captured in a waist-up shot'.
    """
    from image_reference_generator.ollama_integration import (
        audit_caption,
        clean_and_format_caption,
    )

    lead = "tok woman, a waist-up shot"
    trigger = "tok"

    # Layered intro: "tok is a woman captured in a waist-up shot, looking..."
    raw1 = (
        "tok woman, a waist-up shot, tok is a woman captured in a waist-up shot, "
        "looking directly at the camera with a playful expression. "
        "She is wearing a black crop top. "
        "Her hands are resting on her thighs in a sunlit bedroom with soft bokeh."
    )
    ok, cleaned, _ = clean_and_format_caption(raw1, trigger, lead)
    assert ok is True
    assert cleaned.count("waist-up shot") == 1
    assert cleaned.count(trigger) == 1
    pass_audit, viols = audit_caption(cleaned, trigger)
    assert pass_audit is True, f"Failed audit: {viols}"

    # Layered intro: "tok, a woman, is the subject of this waist-up shot. She is..."
    raw2 = (
        "tok woman, a waist-up shot, tok, a woman, is the subject of this waist-up shot. "
        "She is looking directly at the camera with a confident expression wearing a red swimsuit. "
        "Hands are out of frame in a bright room with shallow depth of field."
    )
    ok2, cleaned2, _ = clean_and_format_caption(raw2, trigger, lead)
    assert ok2 is True
    assert cleaned2.count("waist-up shot") == 1
    assert cleaned2.count(trigger) == 1
    pass_audit2, viols2 = audit_caption(cleaned2, trigger)
    assert pass_audit2 is True, f"Failed audit: {viols2}"


def test_temporal_extraction_and_exif(tmp_path):
    """Verify that extract_image_timestamp extracts calendar year from EXIF,
    and falls back cleanly to filesystem mtime when EXIF is not present.
    """
    from PIL import Image

    from image_reference_generator.pass1_filter import extract_image_timestamp

    # 1. Image without EXIF - falls back to mtime
    plain_img = Image.new("RGB", (100, 100), color="blue")
    plain_path = tmp_path / "plain.jpg"
    plain_img.save(plain_path, "JPEG")

    dt, year = extract_image_timestamp(plain_path)
    assert dt is not None
    assert year is not None
    assert year >= 2020

    # 2. Image with EXIF DateTime (Tag 306 = '2018:06:15 12:30:00')
    exif_img = Image.new("RGB", (100, 100), color="white")
    exif_data = exif_img.getexif()
    exif_data[306] = "2018:06:15 12:30:00"
    exif_path = tmp_path / "exif.jpg"
    exif_img.save(exif_path, "JPEG", exif=exif_data)

    dt_exif, year_exif = extract_image_timestamp(exif_path)
    assert dt_exif is not None
    assert year_exif == 2018


def test_scale_invariant_pass2_scoring(tmp_path):
    """Verify that process_pass2_single normalizes images > 1024 to canonical resolution
    so high-megapixel images are evaluated on the exact same spatial scale.
    """
    from PIL import Image

    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass2_ranking import process_pass2_single

    # Create high-res 4000x3000 image
    high_res = Image.new("RGB", (4000, 3000), color="gray")
    high_res_path = tmp_path / "high_res.jpg"
    high_res.save(high_res_path, "JPEG")

    metrics = ImageQualityMetrics(
        path=high_res_path,
        width=4000,
        height=3000,
        face_box=(1000, 800, 500, 500),
        composition="close-up",
    )
    result = process_pass2_single(metrics)
    assert result.sharpness >= 0.0
    assert result.exposure_score > 0.0


def test_pass2_temporal_stratification():
    """Verify that process_pass2 with temporal_balance=True stratifies candidates across
    distinct calendar years instead of letting a dominant recent year take 100% of top_n.
    """
    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass2_ranking import process_pass2

    # Construct mock metrics:
    # 30 images from 2024 (scores between 0.70 and 0.99)
    # 10 images from 2018 (scores between 0.50 and 0.65)
    metrics_list = []
    for i in range(30):
        m = ImageQualityMetrics(
            path=Path(f"2024_{i}.jpg"),
            width=1024,
            height=1024,
            face_box=(100, 100, 200, 200),
            composition="medium",
            year=2024,
            total_score=0.70 + (i * 0.01),
        )
        metrics_list.append(m)

    for i in range(10):
        m = ImageQualityMetrics(
            path=Path(f"2018_{i}.jpg"),
            width=1024,
            height=1024,
            face_box=(100, 100, 200, 200),
            composition="medium",
            year=2018,
            total_score=0.50 + (i * 0.01),
        )
        metrics_list.append(m)

    # Without temporal balance: top 10 would be entirely 2024 images (scores > 0.89)
    unbalanced = process_pass2(metrics_list, top_n=10, temporal_balance=False)
    assert all(m.year == 2024 for m in unbalanced)

    # With temporal balance: top 10 MUST include candidates from 2018!
    balanced = process_pass2(metrics_list, top_n=10, temporal_balance=True)
    years_present = {m.year for m in balanced}
    assert 2018 in years_present
    assert 2024 in years_present
    # Top 2018 performers are included
    count_2018 = sum(1 for m in balanced if m.year == 2018)
    assert count_2018 >= 3


def test_pass3_temporal_and_pose_diversity(monkeypatch):
    """Verify that process_pass3 respects temporal balance soft caps while
    adhering to compositional quotas.
    """
    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass3_diversity import process_pass3

    metrics = []
    # 60 images across 3 years (2020, 2022, 2024)
    for i in range(60):
        comp = "close-up" if i % 3 == 0 else "medium" if i % 3 == 1 else "full-body"
        year = 2020 if i < 20 else 2022 if i < 40 else 2024
        pose = "frontal" if i % 2 == 0 else "three-quarter-left"
        m = ImageQualityMetrics(
            Path(f"img_{i}.jpg"),
            1024,
            1024,
            (0, 0, 10, 10),
            comp,
            year=year,
            head_pose=pose,
        )
        metrics.append(m)

    def mock_extract(metrics_list, batch_size=32):
        # Vary features slightly by index so cosine distance is non-zero
        features = np.zeros((len(metrics_list), 50))
        for idx in range(len(metrics_list)):
            features[idx, idx % 50] = 1.0
        return features, metrics_list

    monkeypatch.setattr("image_reference_generator.pass3_diversity.extract_features", mock_extract)

    result = process_pass3(metrics, target_count=30, temporal_balance=True)
    assert len(result) == 30

    years_in_result = {r.year for r in result}
    assert years_in_result == {2020, 2022, 2024}
    # No single year monopolizes the selection
    for y in [2020, 2022, 2024]:
        count = sum(1 for r in result if r.year == y)
        assert 5 <= count <= 15


def test_era_window_filtering():
    """Verify era window filtering logic accurately restricts candidates to target year range."""
    from image_reference_generator.pass1_filter import ImageQualityMetrics

    candidates = [
        ImageQualityMetrics(Path("2018.jpg"), 1024, 1024, None, "medium", year=2018),
        ImageQualityMetrics(Path("2020.jpg"), 1024, 1024, None, "medium", year=2020),
        ImageQualityMetrics(Path("2022.jpg"), 1024, 1024, None, "medium", year=2022),
        ImageQualityMetrics(Path("2024.jpg"), 1024, 1024, None, "medium", year=2024),
    ]

    # Test range "2020-2023"
    start_y, end_y = 2020, 2023
    filtered = [m for m in candidates if m.year is not None and start_y <= m.year <= end_y]
    assert len(filtered) == 2
    assert {m.year for m in filtered} == {2020, 2022}

    # Test single year "2024"
    single = [m for m in candidates if m.year == 2024]
    assert len(single) == 1
    assert single[0].year == 2024


def test_pass4_single_person_filtering(monkeypatch):
    """Verify Pass 4 detects and replaces multi-person images with single-person candidates."""
    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass4_refinement import process_pass4

    metrics = []
    for i in range(8):
        m = ImageQualityMetrics(
            Path(f"person_{i}.jpg"),
            1024,
            1024,
            (10, 10, 100, 100),
            "medium",
            total_score=float(10 - i),
            lighting_condition="Indoor",
        )
        vec = np.zeros((1, 8))
        vec[0, i] = 1.0
        m.feature_vector = vec
        metrics.append(m)

    selected = metrics[:4]  # person_0, person_1, person_2, person_3
    pool = metrics

    # person_1 has two people, person_5 has one person
    def mock_verify_single(path, model_name="llava:13b"):
        return path.name != "person_1.jpg"

    def mock_analyze_lighting(path, model_name="llava:13b"):
        return "Indoor"

    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.verify_single_person", mock_verify_single
    )
    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.analyze_lighting", mock_analyze_lighting
    )

    refined = process_pass4(selected, pool, model_name="llava:13b")
    refined_names = [m.path.name for m in refined]

    # person_1 must be removed
    assert "person_1.jpg" not in refined_names
    assert len(refined) == len(selected)


def test_pass2_eyewear_penalty(monkeypatch):
    """Verify Pass 2 applies a -0.15 penalty to sunglasses to favor clear facial likeness."""
    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass2_ranking import process_pass2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 120
    img[20:60, 20:60, :] = 200

    monkeypatch.setattr("image_reference_generator.pass2_ranking.load_image", lambda path: img)

    m1 = ImageQualityMetrics(
        Path("bare_eyes.jpg"), 100, 100, (20, 20, 40, 40), "close-up", has_sunglasses=False
    )
    m2 = ImageQualityMetrics(
        Path("sunglasses.jpg"), 100, 100, (20, 20, 40, 40), "close-up", has_sunglasses=True
    )

    result = process_pass2([m1, m2], favor_no_glasses=True)
    score_bare = next(m.total_score for m in result if m.path == Path("bare_eyes.jpg"))
    score_shades = next(m.total_score for m in result if m.path == Path("sunglasses.jpg"))

    assert score_bare > score_shades
    assert abs((score_bare - score_shades) - 0.15) < 1e-4

    # When favor_no_glasses is False, scores are identical
    result_neutral = process_pass2([m1, m2], favor_no_glasses=False)
    score_bare_n = next(m.total_score for m in result_neutral if m.path == Path("bare_eyes.jpg"))
    score_shades_n = next(m.total_score for m in result_neutral if m.path == Path("sunglasses.jpg"))
    assert abs(score_bare_n - score_shades_n) < 1e-5


def test_pass4_sunglasses_replacement(monkeypatch):
    """Verify Pass 4 detects sunglasses and replaces them with clear-eyed photos from pool."""
    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass4_refinement import process_pass4

    metrics = []
    for i in range(6):
        m = ImageQualityMetrics(
            Path(f"photo_{i}.jpg"),
            1024,
            1024,
            (10, 10, 100, 100),
            "medium",
            total_score=float(10 - i),
            lighting_condition="Indoor",
            has_sunglasses=(i == 1),  # photo_1 has sunglasses
        )
        vec = np.zeros((1, 6))
        vec[0, i] = 1.0
        m.feature_vector = vec
        metrics.append(m)

    selected = metrics[:3]  # photo_0, photo_1 (sunglasses), photo_2
    pool = metrics  # photo_3, photo_4, photo_5 are clear-eyed in pool

    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.verify_single_person",
        lambda p, model_name="": True,
    )
    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.analyze_lighting",
        lambda p, model_name="": "Indoor",
    )
    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.check_sunglasses",
        lambda p, model_name="": p.name == "photo_1.jpg",
    )

    refined = process_pass4(selected, pool, model_name="llava:13b", favor_no_glasses=True)
    refined_names = [m.path.name for m in refined]

    assert "photo_1.jpg" not in refined_names
    assert len(refined) == 3
    # Check that replacement came from pool (photo_3)
    assert any(name in refined_names for name in ["photo_3.jpg", "photo_4.jpg", "photo_5.jpg"])


def test_pass4_drop_sunglasses_when_pool_exhausted(monkeypatch):
    """Verify Pass 4 drops sunglasses when drop_sunglasses=True and no replacement exists."""
    from image_reference_generator.pass1_filter import ImageQualityMetrics
    from image_reference_generator.pass4_refinement import process_pass4

    metrics = []
    for i in range(2):
        m = ImageQualityMetrics(
            Path(f"sunglasses_{i}.jpg"),
            1024,
            1024,
            (10, 10, 100, 100),
            "medium",
            total_score=float(10 - i),
            lighting_condition="Indoor",
            has_sunglasses=True,
        )
        vec = np.zeros((1, 2))
        vec[0, i] = 1.0
        m.feature_vector = vec
        metrics.append(m)

    selected = [metrics[0]]
    pool = metrics  # only sunglasses exist

    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.verify_single_person",
        lambda p, model_name="": True,
    )
    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.analyze_lighting",
        lambda p, model_name="": "Indoor",
    )
    monkeypatch.setattr(
        "image_reference_generator.pass4_refinement.check_sunglasses", lambda p, model_name="": True
    )

    refined = process_pass4(
        selected, pool, model_name="llava:13b", favor_no_glasses=True, drop_sunglasses=True
    )
    assert len(refined) == 0


def test_flux_lighting_and_buzzword_audit():
    """Verify audit_caption and clean_and_format_caption handle numbered rubrics,
    semicolons, quality buzzwords ('photorealistic', 'masterpiece'), and subjective
    lighting buzzwords ('dramatic lighting', 'cinematic lighting').
    """
    from image_reference_generator.ollama_integration import (
        audit_caption,
        clean_and_format_caption,
        sanitize_caption_deterministic,
    )

    trigger = "jujub"
    lead = "jujub woman, a waist-up shot"

    # 1. Numbered rubric output from 32B model gets stripped into clean continuous prose
    rubric_output = (
        "1. **Framing and camera angle**: Close-up shot at eye-level.\n"
        "2. **Subject pose, action, hand placement, and expression**: Standing with hands "
        "resting on a wooden railing and a calm expression.\n"
        "3. **Clothing and accessories**: Wearing a black linen top.\n"
        "4. **Setting, background elements, environment, and optical depth of field**: "
        "Outdoor patio with blurred greenery and shallow depth of field.\n"
        "5. **Lighting and color palette**: Lit by soft morning window light from the right "
        "with cool shadow fill."
    )
    ok, cleaned, _ = clean_and_format_caption(rubric_output, trigger, lead)
    assert ok is True
    assert "**Framing" not in cleaned
    assert "1." not in cleaned
    assert cleaned.startswith(lead)
    audit_ok, viols = audit_caption(cleaned, trigger)
    assert audit_ok is True, f"Unexpected violations: {viols}"

    # 2. Quality & subjective lighting buzzwords are flagged by audit_caption
    buzzword_caption = (
        "jujub woman, a waist-up shot at eye-level with hands resting on her hips and a "
        "neutral expression. She wears a white cotton dress in a photorealistic masterpiece "
        "indoor room with dramatic lighting and cinematic lighting."
    )
    audit_ok2, viols2 = audit_caption(buzzword_caption, trigger)
    assert audit_ok2 is False
    assert any("photorealistic/masterpiece" in v for v in viols2)
    assert any("dramatic/cinematic lighting" in v for v in viols2)

    # 3. Deterministic sanitizer removes buzzwords cleanly
    sanitized = sanitize_caption_deterministic(buzzword_caption, trigger, lead)
    audit_ok3, viols3 = audit_caption(sanitized, trigger)
    assert audit_ok3 is True, f"Unexpected violations after sanitize: {viols3}"


def test_arb_no_upscale_for_smaller_images(tmp_path):
    """Verify that process_and_save_images does not artificially upscale images
    smaller than 1024^2 area, snapping them to multiples of 64 at native scale.
    """
    from PIL import Image

    from image_reference_generator.post_processor import process_and_save_images

    # 720x880 image (< 1024^2 area)
    img_small = Image.new("RGB", (720, 880), color="purple")
    small_path = tmp_path / "small.jpg"
    img_small.save(small_path, "JPEG")

    metrics_list = [
        ImageQualityMetrics(small_path, 720, 880, (100, 100, 100, 100), "medium"),
    ]
    out_dir = tmp_path / "output_small_arb"

    class MockReader:
        def readtext(self, roi):
            return []

    process_and_save_images.reader = MockReader()

    process_and_save_images(metrics_list, out_dir, "tok")
    exported = out_dir / "tok_001.jpg"
    assert exported.exists()

    with Image.open(exported) as res_img:
        ew, eh = res_img.size
        assert ew % 64 == 0
        assert eh % 64 == 0
        # 720 rounds to 704, 880 rounds to 896 (not upscaled to 1024^2)
        assert ew == 704
        assert eh == 896

    if hasattr(process_and_save_images, "reader"):
        delattr(process_and_save_images, "reader")


def test_replicate_trainer_api_workflow(tmp_path, monkeypatch):
    """Verify ensure_destination_model and start_replicate_training for both ostris and fast
    trainers using a mocked Replicate client.
    """
    from image_reference_generator.replicate_trainer import (
        ensure_destination_model,
        start_replicate_training,
    )

    zip_path = tmp_path / "tok_dataset.zip"
    zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    created_models = []
    submitted_trainings = []

    class DummyVersion:
        id = "mock_ver_12345"

    class DummyModel:
        latest_version = DummyVersion()

    class DummyAccount:
        username = "mockuser"

    class DummyTraining:
        id = "trn_999"
        status = "starting"

    class MockModelsAPI:
        def get(self, slug):
            if slug == "mockuser/new-lora":
                raise RuntimeError("Model not found")
            return DummyModel()

        def create(self, **kwargs):
            created_models.append(kwargs)
            return DummyModel()

    class MockAccountsAPI:
        def current(self):
            return DummyAccount()

    class MockTrainingsAPI:
        def create(self, **kwargs):
            submitted_trainings.append(kwargs)
            return DummyTraining()

    class MockReplicateClient:
        def __init__(self, api_token=None):
            self.api_token = api_token
            self.models = MockModelsAPI()
            self.accounts = MockAccountsAPI()
            self.trainings = MockTrainingsAPI()

    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test_token_mock")
    monkeypatch.setattr("replicate.Client", MockReplicateClient)

    # 1. Test automatic username resolution and model creation
    client = MockReplicateClient()
    ok_dest, slug, msg = ensure_destination_model(client, "new-lora", "tok")
    assert ok_dest is True
    assert slug == "mockuser/new-lora"
    assert len(created_models) == 1
    assert created_models[0]["name"] == "new-lora"

    # 2. Test ostris trainer launch (default: steps=3000, lora_rank=32, autocaption=False)
    ok_ostris, tid_ostris, msg_ostris = start_replicate_training(
        zip_path=zip_path,
        destination="mockuser/existing-lora",
        trigger_word="tok",
        trainer="ostris",
        steps=3000,
        lora_rank=32,
    )
    assert ok_ostris is True
    assert tid_ostris == "trn_999"
    assert len(submitted_trainings) == 1
    ostris_call = submitted_trainings[0]
    assert ostris_call["destination"] == "mockuser/existing-lora"
    assert ostris_call["version"].startswith("ostris/flux-dev-lora-trainer:")
    assert ostris_call["input"]["autocaption"] is False
    assert ostris_call["input"]["steps"] == 3000
    assert ostris_call["input"]["lora_rank"] == 32
    assert ostris_call["input"]["resolution"] == "512,768,1024"

    # 3. Test fast trainer launch (replicate/fast-flux-trainer)
    ok_fast, tid_fast, _ = start_replicate_training(
        zip_path=zip_path,
        destination="mockuser/existing-lora",
        trigger_word="tok",
        trainer="fast",
        steps=3000,
    )
    assert ok_fast is True
    assert tid_fast == "trn_999"
    fast_call = submitted_trainings[1]
    assert fast_call["version"].startswith("replicate/fast-flux-trainer:")
    assert fast_call["input"]["training_steps"] == 3000
    assert fast_call["input"]["lora_type"] == "subject"


def test_screenshot_letterbox_filter():
    """Verify is_letterboxed_screenshot detects solid black letterbox and pillarbox bars."""
    from image_reference_generator.pass1_filter import is_letterboxed_screenshot

    # 1. Normal clean image
    clean = np.full((100, 100, 3), 150, dtype=np.uint8)
    assert is_letterboxed_screenshot(clean) is False

    # 2. Letterboxed image (black top and bottom bars)
    letterboxed = np.full((100, 100, 3), 150, dtype=np.uint8)
    letterboxed[:8, :, :] = 0
    letterboxed[-8:, :, :] = 0
    assert is_letterboxed_screenshot(letterboxed) is True

    # 3. Pillarboxed image (black left and right bars)
    pillarboxed = np.full((100, 100, 3), 150, dtype=np.uint8)
    pillarboxed[:, :8, :] = 0
    pillarboxed[:, -8:, :] = 0
    assert is_letterboxed_screenshot(pillarboxed) is True
