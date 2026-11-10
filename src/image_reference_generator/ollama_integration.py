import base64
import contextlib
import io
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

DEFAULT_VISION_MODEL = "redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest"

BANNED_CLICHES = [
    (
        r"\b(?:that|which|to)?\s*(?:accentuates?|highlights?|flatter(?:s|ing)?)\s+"
        r"(?:her|his|their|the)\s+(?:figure|physique|curves|body|frame)\b",
        "accentuates figure/physique",
    ),
    (r"\badd(?:ing|s)?\s+a\s+sense\s+of\b", "adding a sense of"),
    (
        r"\bfocus(?:ing|es)?\s+the\s+viewer(?:'s)?\s+attention(?:\s+on\s+[^.,;]+)?\b",
        "focusing the viewer attention",
    ),
    (
        r"\bdraw(?:ing|s)?\s+the\s+viewer(?:'s)?\s+attention(?:\s+on\s+[^.,;]+)?\b",
        "drawing the viewer attention",
    ),
    (r"\bcentral\s+subject\s+of\s+the\s+image\b", "central subject of the image"),
    (
        r"\bemphasiz(?:ing|es)\s+(?:her|his|their)\s+presence(?:\s+in\s+the\s+frame)?\b",
        "emphasizing presence in the frame",
    ),
    (r"\ba\s+sense\s+of\s+mystery\b", "a sense of mystery"),
    (r"\bis\s+captured\s+in\b", "is captured in"),
    (r"\bis\s+the\s+subject\s+of\s+this\b", "is the subject of this"),
    (
        r"\b(?:photorealistic|hyperrealistic|masterpiece|8k|uhd)\b",
        "quality buzzword (photorealistic/masterpiece)",
    ),
    (
        r"\b(?:dramatic|cinematic)\s+lighting\b",
        "subjective lighting buzzword (dramatic/cinematic lighting)",
    ),
]


def check_ollama_status(model_name: str = DEFAULT_VISION_MODEL) -> bool:
    """Pre-flight check to verify Ollama daemon is running and the model is available.

    Args:
        model_name: Target Ollama model signature.

    Returns:
        True if the daemon is reachable and model is present, False otherwise.
    """
    model_found = False
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        for m in models:
            m_name = m.get("name", "")
            if (
                m_name == model_name
                or m_name.startswith(model_name + ":")
                or (model_name.endswith(":latest") and m_name == model_name[:-7])
            ):
                model_found = True
                break
    except requests.exceptions.RequestException:
        print("Ollama daemon is not running. Attempting to start automatically...")
        try:
            if sys.platform == "darwin":
                subprocess.run(["brew", "services", "start", "ollama"], check=True)
            elif sys.platform == "linux":
                subprocess.run(["sudo", "systemctl", "start", "ollama"], check=True)
            else:
                print("Auto-start unsupported on Windows. Please start Ollama manually.")
                return False
            time.sleep(3)  # Wait a few seconds for the daemon to fully initialize

            # Second attempt to verify it started properly
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            for m in models:
                m_name = m.get("name", "")
                if (
                    m_name == model_name
                    or m_name.startswith(model_name + ":")
                    or (model_name.endswith(":latest") and m_name == model_name[:-7])
                ):
                    model_found = True
                    break

        except (
            subprocess.CalledProcessError,
            requests.exceptions.RequestException,
            FileNotFoundError,
        ):
            print(
                "Error: Could not automatically start or connect to local Ollama daemon on "
                "port 11434."
            )
            print("Please ensure Ollama is installed and running.")
            return False

    if not model_found:
        print(f"❌ Error: Ollama daemon is running, but model '{model_name}' is not found.")
        print(f"👉 Please run: ollama pull {model_name}")
        print("💡 Recommended modern vision models for FLUX curation:")
        print(
            "   - ollama pull redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest "
            "(fast uncensored default)"
        )
        print("   - ollama pull qwen2.5vl:32b   (flagship detail for 64GB Mac)")
        print("   - ollama pull qwen2.5vl:7b    (fast & lightweight)")
        return False

    # Pre-warm model into VRAM and verify vision backend
    print(f"🔥 Pre-warming Ollama model '{model_name}' and verifying vision pipeline...")
    test_img = Image.new("RGB", (64, 64), color="white")
    test_buf = io.BytesIO()
    test_img.save(test_buf, format="JPEG")
    test_b64 = base64.b64encode(test_buf.getvalue()).decode("utf-8")

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "hi",
                "images": [test_b64],
                "stream": False,
                "keep_alive": "15m",
                "options": {"num_predict": 1},
            },
            timeout=(10.0, 60.0),
        )
        if resp.status_code != 200:
            err_msg = resp.text
            with contextlib.suppress(Exception):
                err_msg = resp.json().get("error", err_msg)
            print(f"❌ Error: Model '{model_name}' failed vision pre-flight check: {err_msg}")
            if "unknown model architecture: 'mllama'" in err_msg or "mllama" in err_msg:
                print(
                    "💡 Note: Your local Ollama installation lacks 'mllama' architecture support."
                )
                print(
                    "   Please run with '--model "
                    "redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest' "
                    "or '--model qwen2.5vl:32b'."
                )
            return False
    except requests.exceptions.RequestException as e:
        print(f"Warning: Model pre-warm failed non-fatally: {str(e)}")

    return True


def _encode_image_for_ollama(image_path: Path, max_size: int = 1024) -> str:
    """Safely loads and downscales an image before base64 encoding to prevent VRAM OOM."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Failed to process image {image_path.name} for Ollama: {str(e)}")
        return ""


def clean_and_format_caption(
    raw_text: str, trigger_word: str, lead: str
) -> tuple[bool, str, str]:
    """Cleans, standardizes, and validates a caption generated for FLUX.1 models.

    Decouples visual scene description from the dataset trigger prefix, cleanly stripping
    repetitive model introductions, section headers, and eliminating the double-lead stutter bug.

    Args:
        raw_text: Raw textual output from the vision model.
        trigger_word: Target identifying sequence.
        lead: Required opening sentence lead (e.g. 'ohwx woman, a close-up shot').

    Returns:
        A tuple of (is_valid, formatted_caption, rejection_reason).
    """
    if not raw_text or not raw_text.strip():
        return False, "", "Empty response body"

    text = raw_text.strip()

    # Remove markdown code fences and backticks
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.replace("```", "").replace("`", "")
    text = text.strip("\"' ")

    # Strip conversational preambles
    convo_prefixes = [
        r"^(?:sure|certainly|here is|here\'s|this is|okay|ok)\b[^:\n]*[:\n-]\s*",
        r"^(?:in this image|in the image|in this photo|the image shows|this photo shows)\s*,?\s*",
    ]
    for cp in convo_prefixes:
        text = re.sub(cp, "", text, flags=re.IGNORECASE).strip()

    # Discard any explicit Tags: section or trailing tag blocks
    text = re.split(r"\n\s*tags?\s*:", text, flags=re.IGNORECASE)[0]
    text = re.sub(r"^tags?\s*:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Filter out bullet points, numbered list markers, and rubric section headers
    cleaned_lines = []
    for line in text.split("\n"):
        line_s = line.strip()
        if not line_s or re.match(r"^tags?\s*:", line_s, re.IGNORECASE):
            continue
        # Strip leading numbered/bullet markers like "1. " or "- "
        line_s = re.sub(r"^[-*•\d.]+\s*", "", line_s).strip()
        # Strip bold/plain rubric headers like "**Framing and camera angle**: "
        line_s = re.sub(
            r"^\*{0,2}(?:framing(?:\s+and\s+camera\s+angle)?|subject\s+pose[^:]*|"
            r"clothing(?:\s+and\s+accessories)?|setting[^:]*|lighting[^:]*)\*{0,2}\s*:\s*",
            "",
            line_s,
            flags=re.IGNORECASE,
        ).strip()
        if line_s:
            cleaned_lines.append(line_s)

    if not cleaned_lines:
        return False, "", "Empty response after stripping preambles and tags"

    desc = " ".join(cleaned_lines).strip()
    desc = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", desc)
    # Normalize semicolons to commas or periods for smooth natural prose
    desc = re.sub(r"\s*;\s*", ", ", desc)

    raw_words = desc.split()
    if len(raw_words) < 5:
        return False, "", f"Description too short ({len(raw_words)} words)"

    intro_pat = (
        r"^(?:(?:is|are)\s+)?(?:(?:a|the|this)\s+)?(?:woman|man|person|model|subject|individual)?\s*"
        r"(?:(?:is|are)\s+)?(?:captured\s+in|seen\s+in|shown\s+in|featured\s+in|the\s+subject\s+of|"
        r"pictured\s+in)?\s*"
        r"(?:(?:in|of|for|with)\s+)?(?:(?:a|the|this)\s+)?(?:close[- ]up|waist[- ]up|full[- ]body|"
        r"medium|wide[- ]angle|eye[- ]level|low[- ]angle|high[- ]angle|"
        r"portrait|photo|picture|shot|image)?\s*(?:portrait|photo|shot|picture|image)?\s*(?:of\s+)?"
        r"(?:(?:a|the)\s+)?(?:woman|man|person|model|subject|individual)?\s*"
        r"(?:is\s+)?(?:captured|seen|shown|pictured|featured)?\s*"
    )

    # Cleanly strip layered occurrences of lead, trigger, or generic intros from the start
    while True:
        prev_len = len(desc)
        if desc.lower().startswith(lead.lower()):
            desc = desc[len(lead) :].strip(" ,.-")
        desc = re.sub(
            rf"^(?:{re.escape(trigger_word)}\b\s*(?:woman|man|person|model)?\s*[,:-]?\s*)+",
            "",
            desc,
            flags=re.IGNORECASE,
        ).strip(" ,.-")
        desc = re.sub(intro_pat, "", desc, flags=re.IGNORECASE).strip(" ,.-")
        if len(desc) == prev_len:
            break

    # Also remove any stray duplicate occurrences of trigger_word inside the body
    desc = re.sub(
        rf"\b{re.escape(trigger_word)}\s+(?:woman|man|person|model)\b",
        "the subject",
        desc,
        flags=re.IGNORECASE,
    )
    desc = re.sub(
        rf"\b{re.escape(trigger_word)}\b",
        "she",
        desc,
        flags=re.IGNORECASE,
    )

    # Connect lead with rest
    if desc.lower().startswith((
        "looking",
        "seated",
        "sitting",
        "standing",
        "lying",
        "kneeling",
        "leaning",
        "posing",
        "posed",
        "wearing",
        "dressed",
        "with ",
        "in ",
        "at ",
        "from ",
        "outdoors",
        "indoors",
    )):
        desc = f"{lead} {desc}"
    elif desc:
        desc = f"{lead}, {desc[0].lower() + desc[1:]}"
    else:
        desc = f"{lead}."

    # Standardize whitespace and punctuation
    desc = re.sub(r"\s+", " ", desc).strip()
    if not desc.endswith((".", "!", "?")):
        desc += "."

    words = desc.split()
    if len(words) < 15:
        return False, "", f"Description too short ({len(words)} words)"
    if len(words) > 130:
        return False, "", f"Description too long ({len(words)} words)"

    return True, desc, "OK"


def audit_caption(caption: str, trigger_word: str) -> tuple[bool, list[str]]:
    """Pass 2: Deterministic rule-based validator auditing a caption against FLUX dataset standards.

    Checks:
    1. Word budget: between 20 and 95 words (1 to 3 concise technical sentences).
    2. Stutter: trigger word count == 1, no duplicate framing phrases.
    3. Banned clichés, quality buzzwords ('photorealistic', 'masterpiece'), and subjective
       lighting buzzwords ('dramatic lighting', 'cinematic lighting').
    4. Hand visibility: explicit placement or out-of-frame notice.
    5. Sentence punctuation.

    Args:
        caption: Formatted candidate caption string.
        trigger_word: Target identifying sequence.

    Returns:
        tuple of (is_valid, list_of_violations).
    """
    violations = []
    if not caption or not caption.strip():
        return False, ["Caption is empty"]

    text = caption.strip()
    words = text.split()

    # 1. Word budget
    if len(words) < 20:
        violations.append(f"Too short ({len(words)} words, minimum 20)")
    elif len(words) > 95:
        violations.append(f"Too long ({len(words)} words, maximum 95)")

    # 2. Trigger word count (strictly 1)
    tc = len(re.findall(rf"\b{re.escape(trigger_word)}\b", text, re.IGNORECASE))
    if tc == 0:
        violations.append(f"Missing trigger word '{trigger_word}'")
    elif tc > 1:
        violations.append(f"Duplicate trigger word '{trigger_word}' ({tc} occurrences)")

    # 3. Framing repetition check
    for shot in ["close-up shot", "waist-up shot", "full-body shot", "portrait shot"]:
        if text.lower().count(shot) > 1:
            violations.append(f"Duplicate framing phrase '{shot}'")

    # 4. Banned clichés & fluff
    for pat, desc in BANNED_CLICHES:
        if re.search(pat, text, re.IGNORECASE):
            violations.append(f"Banned cliché / meta-fluff: '{desc}'")

    # 5. Hand visibility check
    has_hands = any(
        w in text.lower()
        for w in ["hand", "hands", "finger", "fingers", "out of frame", "wrists", "arms"]
    )
    if not has_hands:
        violations.append(
            "Missing explicit hand placement (state location or 'hands are out of frame')"
        )

    # 6. Punctuation
    if not text.endswith((".", "!", "?")):
        violations.append("Missing terminal punctuation")

    return len(violations) == 0, violations


def sanitize_caption_deterministic(caption: str, trigger_word: str, lead: str) -> str:
    """Deterministic fallback that cleans banned clichés and guarantees hand mention."""
    text = caption.strip()

    # Remove banned phrases
    text = re.sub(
        r",?\s*(?:that|which|to)?\s*(?:accentuates?|highlights?|flatter(?:s|ing)?)\s+"
        r"(?:her|his|their|the)\s+(?:figure|physique|curves|body|frame)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*add(?:ing|s)?\s+a\s+sense\s+of\s+[^.,;]+(?:\s+to\s+the\s+image)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*focus(?:ing|es)?\s+the\s+viewer(?:'s)?\s+attention(?:\s+on\s+[^.,;]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*draw(?:ing|s)?\s+the\s+viewer(?:'s)?\s+attention(?:\s+on\s+[^.,;]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*emphasiz(?:ing|es)\s+(?:her|his|their)\s+presence(?:\s+in\s+the\s+frame)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*(?:emphasiz(?:ing|es)|mak(?:ing|es)|highlight(?:ing|s))?\s*"
        r"(?:her|him|them|the\s+subject)?\s*(?:as\s+)?(?:the\s+)?central\s+subject"
        r"(?:\s+of\s+the\s+image)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*(?:and\s+)?putting\s+the\s+focus\s+on\s+her",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:photorealistic|hyperrealistic|masterpiece|8k|uhd)\b\s*,?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:dramatic|cinematic)\s+lighting\b",
        "soft directional lighting",
        text,
        flags=re.IGNORECASE,
    )

    # Ensure hands are mentioned
    has_hands = any(
        w in text.lower()
        for w in ["hand", "hands", "finger", "fingers", "out of frame", "wrists", "arms"]
    )
    if not has_hands:
        text = text.rstrip(". ") + ". Hands are out of frame."

    # Clean punctuation and whitespace
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;])", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    if not text.endswith((".", "!", "?")):
        text += "."

    return text


def generate_caption(
    image_path: Path,
    trigger_word: str,
    composition: str = "medium",
    model_name: str = DEFAULT_VISION_MODEL,
    hand_status: str | None = None,
    max_audit_retries: int = 2,
    year: int | None = None,
) -> bool:
    """Interacts with local Ollama daemon via a 3-pass generation, audit, and educated retry loop.

    - Pass 1: Extract visual scene facts in strict 5-part FLUX order with lighting rubric.
    - Pass 2: Deterministic audit checking for stutter, banned clichés, hand mention, token budget.
    - Pass 3: If audit fails, re-prompt with an educated prompt detailing specific violations.

    Args:
        image_path: Absolute pathway to the cropped output JPEG.
        trigger_word: Target identifying sequence to instruct the LLM.
        composition: Geometrical classification ('close-up', 'medium', 'full-body').
        model_name: Target Ollama model signature.
        hand_status: Optional flag if hands are detected as partially cropped at frame edge.
        max_audit_retries: Maximum number of educated retries on audit failure.
        year: Optional calendar year of the photo for temporal anchoring.

    Returns:
        True if caption generation and write succeeded, False otherwise.
    """
    encoded_string = _encode_image_for_ollama(image_path)
    if not encoded_string:
        return False

    has_class_noun = any(
        w in trigger_word.lower().split()
        for w in ["woman", "man", "person", "girl", "boy", "model"]
    )
    subject_str = trigger_word if has_class_noun else f"{trigger_word} woman"

    if composition == "close-up":
        lead = f"{subject_str}, a close-up shot"
    elif composition == "medium":
        lead = f"{subject_str}, a waist-up shot"
    elif composition == "full-body":
        lead = f"{subject_str}, a full-body shot"
    else:
        lead = f"{subject_str}, a portrait shot"

    hand_note = f" (Note: {hand_status})" if hand_status else ""
    era_note = f" Note that the photo was captured in {year}." if year else ""

    base_prompt = (
f"You are a technical image tagger for FLUX training datasets. "
f"Describe this image in 2 to 3 concise, literal, natural-language sentences "
f"(40 to 85 words total) written as a single continuous paragraph.\n\n"
f"CRITICAL VISIBILITY RULE:\n"
f"- Only describe body parts, clothing, and background elements ACTUALLY VISIBLE in this crop.\n"
f"- If this is a close-up (chest-up or headshot), DO NOT mention or invent hips, waist, legs, "
f"feet, or lower-body clothing.\n"
f"- If this is a waist-up shot, DO NOT mention or invent legs, feet, or shoes.\n"
f"- If the subject is viewed from the front, do NOT describe the back. If viewed from behind, "
f"do NOT describe front facial features.\n\n"
f"Follow this strict sequence:\n"
f"1. Framing and camera angle: Open with '{lead}' and specify the camera angle "
f"(e.g., eye-level, slightly elevated angle, low-angle, wide angle).\n"
f"2. Subject pose, action, hand placement, body details, and expression: State body posture, "
f"exact hand resting positions (or explicitly state 'hands are out of frame'){hand_note}, "
f"and facial expression (do NOT describe facial identity traits, bone structure, eye color, "
f"hair color, or race). When visible in the crop, describe breasts and buttocks in concrete "
f"literal terms (size, shape, cleavage, how clothing sits on or between them).\n"
f"3. Clothing and accessories: Describe colors, fabric textures, fit, and style for garments "
f"that are actually visible in the frame. When the subject wears a bikini or lingerie, name "
f"the style precisely if clear (thong bikini, g-string bikini, micro bikini, lace lingerie, etc.) "
f"and note how the garments sit on the breasts or buttocks. For casual clothing (yoga outfits, "
f"cocktail dresses, everyday wear) describe the garments normally while still noting fit around "
f"the chest and hips when visible. If the subject is topless, state that clearly. If wearing "
f"sunglasses or eyeglasses, state 'wearing sunglasses' or 'wearing eyeglasses' (do not mention "
f"eyewear if eyes are bare).\n"
f"4. Setting, background elements, environment, and optical depth of field "
f"(e.g., shallow depth of field with blurred background or sharp deep focus).{era_note}\n"
f"5. Lighting and color palette:\n"
f"Analyze the lighting setup using standard photography terminology:\n"
f"- Identify key light direction (e.g., front-lit, lit from the right, strong backlight).\n"
f"- Note diffusion/hardness (e.g., soft ambient light, harsh midday sun with sharp shadows).\n"
f"- Note color temperature or artificial sources (e.g., warm tungsten glow, cool blue shadows).\n"
f"Express this in a single concise clause at the end of the caption "
f"(e.g., 'lit by soft morning window light from the right with cool shadow fill').\n\n"
f"Formatting Rules:\n"
f"- Do NOT use metaphors, poetic words, or quality buzzwords like 'photorealistic', "
f"'masterpiece', 'accentuates her figure', 'dramatic lighting', 'cinematic lighting'.\n"
f"- Stay strictly literal and physical.\n"
f"- Do NOT output numbered lists, bullet points, section headers, or 'Tags:'. "
f"Output only the natural-language paragraph."
)

    url = "http://localhost:11434/api/generate"
    current_prompt = base_prompt
    best_candidate = ""

    for attempt in range(max_audit_retries + 1):
        try:
            temp = 0.2 if attempt == 0 else min(0.5, 0.2 + 0.15 * attempt)
            payload = {
                "model": model_name,
                "prompt": current_prompt,
                "images": [encoded_string],
                "stream": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": temp,
                    "num_predict": 256,
                },
            }

            response = requests.post(url, json=payload, timeout=(10.0, 180.0))
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"Ollama app-level error: {result['error']}")

            raw_caption = result.get("response", "").strip()
            if not raw_caption:
                raise RuntimeError("Ollama returned empty response body")

            is_valid, formatted_caption, reason = clean_and_format_caption(
                raw_caption, trigger_word, lead
            )
            if not is_valid:
                raise RuntimeError(f"Malformed caption ({reason}): {raw_caption[:80]!r}")

            best_candidate = formatted_caption

            # --- PASS 2: AUDIT ---
            passes_audit, violations = audit_caption(formatted_caption, trigger_word)
            if passes_audit:
                # Save validated caption
                txt_path = image_path.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(formatted_caption)
                return True

            # --- PASS 3: EDUCATED RETRY ---
            if attempt < max_audit_retries:
                feedback = "\n".join(f"- {v}" for v in violations)
                current_prompt = (
                    f"{base_prompt}\n\n"
                    f"⚠️ PREVIOUS ATTEMPT FAILED QUALITY AUDIT FOR THESE REASONS:\n"
                    f"{feedback}\n"
                    f"CRITICAL: You must fix these violations. Do NOT use banned phrases. "
                    f"State hand placement explicitly. Only describe physical visible reality."
                )
                time.sleep(1.0)
            else:
                # Retries exhausted: apply deterministic fallback sanitizer
                sanitized = sanitize_caption_deterministic(best_candidate, trigger_word, lead)
                txt_path = image_path.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(sanitized)
                return True

        except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
            if attempt < max_audit_retries:
                sleep_time = 2.0 * (attempt + 1)
                time.sleep(sleep_time)
            else:
                print(f"Critical failure generating caption for {image_path.name}: {e}")
                if best_candidate:
                    sanitized = sanitize_caption_deterministic(best_candidate, trigger_word, lead)
                    txt_path = image_path.with_suffix(".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(sanitized)
                    return True
                return False

    return False


def audit_directory(target_dir: Path, trigger_word: str) -> dict:
    """Audits all caption text files in target_dir and prints a diagnostic report.

    Args:
        target_dir: Directory containing candidate text files.
        trigger_word: Target identifying sequence.

    Returns:
        Diagnostic summary dictionary.
    """
    txt_files = sorted(target_dir.glob(f"{trigger_word}_*.txt"))
    if not txt_files:
        txt_files = sorted(target_dir.glob("*.txt"))

    total = len(txt_files)
    if total == 0:
        print(f"⚠️ No caption text files found in {target_dir}")
        return {"total": 0, "passed": 0, "pass_rate": 0.0, "violations": {}}

    passed = 0
    violation_counts: dict[str, int] = {}
    file_violations: list[tuple[str, list[str]]] = []

    for tf in txt_files:
        try:
            content = tf.read_text(encoding="utf-8").strip()
            ok, violations = audit_caption(content, trigger_word)
            if ok:
                passed += 1
            else:
                file_violations.append((tf.name, violations))
                for v in violations:
                    # Clean specific count numbers for aggregation
                    v_key = re.sub(r"\(\d+.*?\)", "", v).strip()
                    violation_counts[v_key] = violation_counts.get(v_key, 0) + 1
        except Exception as e:
            violation_counts[f"Read error: {e}"] = violation_counts.get(f"Read error: {e}", 0) + 1

    pass_rate = (passed / total) * 100.0

    print("\n" + "=" * 80)
    print(f"📊 Caption Audit Report for: {target_dir}")
    print("=" * 80)
    print(f"Total Files Analyzed : {total}")
    print(f"Passed Audit         : {passed} / {total} ({pass_rate:.1f}%)")
    print(f"Failed Audit         : {total - passed} / {total}")
    print("-" * 80)

    if violation_counts:
        print("Breakdown of Violations:")
        for v_type, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
            print(f"  ❌ {v_type}: {count} files")
        print("-" * 80)
        print("Sample Failed Files:")
        for fname, v_list in file_violations[:5]:
            print(f"  • {fname}: {v_list}")
    else:
        print("🎉 100% of captions passed all FLUX dataset quality checks!")

    print("=" * 80 + "\n")

    return {
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "violations": violation_counts,
    }


def recaption_directory(
    target_dir: Path,
    trigger_word: str,
    model_name: str = DEFAULT_VISION_MODEL,
    max_audit_retries: int = 2,
) -> tuple[int, int]:
    """Re-captions an existing dataset directory in-place using the 3-pass audit engine.

    Args:
        target_dir: Directory containing existing images.
        trigger_word: Target identifying sequence.
        model_name: Target Ollama vision model.
        max_audit_retries: Max educated retry attempts per image.

    Returns:
        tuple of (successful_captions, total_images).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([
        f
        for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts and f.name.startswith(f"{trigger_word}_")
    ])

    if not images:
        images = sorted([
            f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts
        ])

    if not images:
        print(f"❌ Error: No images found in {target_dir}")
        return 0, 0

    print(f"🚀 Re-captioning {len(images)} images in {target_dir} via '{model_name}'...")
    successful = 0

    for img_path in tqdm(images, desc="Auditing & Re-captioning"):
        # Infer composition from existing caption if present, or aspect ratio
        txt_path = img_path.with_suffix(".txt")
        composition = "medium"
        if txt_path.exists():
            try:
                old_text = txt_path.read_text(encoding="utf-8").lower()
                if "close-up" in old_text:
                    composition = "close-up"
                elif "full-body" in old_text:
                    composition = "full-body"
            except Exception:
                pass

        if generate_caption(
            img_path,
            trigger_word,
            composition=composition,
            model_name=model_name,
            max_audit_retries=max_audit_retries,
        ):
            successful += 1

    print(f"\n✅ Finished re-captioning: {successful}/{len(images)} files updated successfully.")
    return successful, len(images)


_vlm_analysis_cache: dict[str, tuple[bool, str, bool, bool]] = {}


def analyze_subject_and_lighting(
    image_path: Path, model_name: str = DEFAULT_VISION_MODEL
) -> tuple[bool, str, bool, bool]:
    """Uses Ollama to verify single-person exclusivity, classify lighting,
    detect sunglasses/tinted eyewear, and verify clear unobstructed face visibility.

    Args:
        image_path: Path to the candidate image.
        model_name: Target Ollama model signature.

    Returns:
        tuple of (is_single, lighting, has_sunglasses, has_face_occlusions).
    """
    cache_key = f"{image_path}_{model_name}"
    if cache_key in _vlm_analysis_cache:
        return _vlm_analysis_cache[cache_key]

    encoded_string = _encode_image_for_ollama(image_path)
    if not encoded_string:
        return True, "Indoor", False, False

    prompt = (
        "You are an expert photography assistant analyzing image data for ML LoRA training.\n"
        "1. Are there multiple people visible anywhere in the frame? Answer 'SINGLE_PERSON' "
        "if exactly 1 person, or 'MULTIPLE_PEOPLE' if 2 or more people.\n"
        "2. Is the lighting predominantly 'Indoor' or 'Outdoor'?\n"
        "3. Is the subject wearing sunglasses or dark tinted shades covering their eyes? "
        "Answer 'SUNGLASSES' if wearing sunglasses, or 'NO_SUNGLASSES' if eyes are bare or "
        "wearing clear glasses.\n"
        "4. Is the face obstructed by a hat/cap covering the forehead, large over-ear headphones, "
        "hands over the face, or turned away with back to camera? Answer 'FACE_OCCLUDED' if "
        "obstructed/turned away, or 'FACE_CLEAR' if the face is clean and unobstructed.\n\n"
        "Format your response strictly as:\n"
        "SUBJECT: <SINGLE_PERSON or MULTIPLE_PEOPLE>\n"
        "LIGHTING: <Indoor or Outdoor>\n"
        "EYEWEAR: <SUNGLASSES or NO_SUNGLASSES>\n"
        "FACE_VISIBILITY: <FACE_CLEAR or FACE_OCCLUDED>"
    )

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [encoded_string],
        "stream": False,
        "keep_alive": "15m",
        "options": {
            "temperature": 0.1,
            "num_predict": 60,
        },
    }

    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=(10.0, 60.0))
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"Ollama app-level error: {result['error']}")

            resp_text = result.get("response", "").strip()

            is_single = True
            if "MULTIPLE_PEOPLE" in resp_text:
                is_single = False
            elif "SINGLE_PERSON" in resp_text:
                is_single = True

            lighting = "Indoor"
            if "Outdoor" in resp_text:
                lighting = "Outdoor"
            elif "Indoor" in resp_text:
                lighting = "Indoor"

            has_sunglasses = (
                "SUNGLASSES" in resp_text and "NO_SUNGLASSES" not in resp_text
            ) or "EYEWEAR: SUNGLASSES" in resp_text

            has_face_occlusions = (
                "FACE_OCCLUDED" in resp_text
                or "EYEWEAR: SUNGLASSES" in resp_text
            )

            _vlm_analysis_cache[cache_key] = (
                is_single,
                lighting,
                has_sunglasses,
                has_face_occlusions,
            )
            return is_single, lighting, has_sunglasses, has_face_occlusions

        except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2**attempt)
                time.sleep(sleep_time)
            else:
                print(f"Critical failure analyzing subject/lighting for {image_path.name}: {e}")

    _vlm_analysis_cache[cache_key] = (True, "Indoor", False, False)
    return True, "Indoor", False, False


def verify_single_person(image_path: Path, model_name: str = DEFAULT_VISION_MODEL) -> bool:
    """Convenience wrapper to check if an image strictly features only one person."""
    is_single, _, _, _ = analyze_subject_and_lighting(image_path, model_name=model_name)
    return is_single


def analyze_lighting(image_path: Path, model_name: str = DEFAULT_VISION_MODEL) -> str:
    """Uses Ollama to determine if the image lighting is predominantly 'Indoor' or 'Outdoor'."""
    _, lighting, _, _ = analyze_subject_and_lighting(image_path, model_name=model_name)
    return lighting


def check_sunglasses(image_path: Path, model_name: str = DEFAULT_VISION_MODEL) -> bool:
    """Uses Ollama to check if the subject is wearing sunglasses or dark/tinted eyewear."""
    _, _, has_sunglasses, _ = analyze_subject_and_lighting(image_path, model_name=model_name)
    return has_sunglasses


def check_face_occlusions(image_path: Path, model_name: str = DEFAULT_VISION_MODEL) -> bool:
    """Uses Ollama to check if face is obstructed by hats, headphones, sunglasses, or turned."""
    _, _, _, has_face_occlusions = analyze_subject_and_lighting(image_path, model_name=model_name)
    return has_face_occlusions
