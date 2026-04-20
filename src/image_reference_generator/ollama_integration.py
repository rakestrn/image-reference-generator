import base64
import time
from pathlib import Path

import requests


def generate_caption(
    image_path: Path, trigger_word: str, model_name: str = "llama3.2-vision"
) -> bool:
    """Interacts with the local Ollama daemon to extract a caption describing the image subject.

    Includes an exponentially decaying retry mechanism. When a Vision model receives its
    first inference call, Ollama must flush VRAM and load up thick multi-GB safetensors,
    often provoking typical client timeouts strictly on the first attempt sequence.

    Args:
        image_path: Absolute pathway to the cropped output JPEG.
        trigger_word: Target identifying sequence to instruct the LLM.
        model_name: Target Ollama model signature.

    Returns:
        True if the caption generation and write succeeded, False otherwise.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    except OSError as e:
        print(f"Failed to access {image_path.name} for captioning: {str(e)}")
        return False

    prompt = (
        f"You are a professional captioning assistant for an image generation model. "
        f"Look at this image of {trigger_word} and write a single, natural, "
        f"descriptive English sentence. "
        f"The sentence must begin with: 'A photo of {trigger_word}...'. "
        f"Describe {trigger_word}'s outfit, pose, expression, and the "
        f"background clearly and concisely. "
        f"Do not add fluff. Focus on accurate identification."
    )

    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "images": [encoded_string], "stream": False}

    max_retries = 3
    base_delay = 5.0  # seconds

    for attempt in range(max_retries):
        try:
            # 120s timeout per attempt to accommodate dense local inference layers
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            caption = result.get("response", "").strip()

            # Save caption structurally adjacent to the JPEG
            txt_path = image_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            return True

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2**attempt)
                print(
                    f"Ollama connection timeout on {image_path.name} "
                    f"(VRAM cold boot). Retrying in {sleep_time}s..."
                )
                time.sleep(sleep_time)
            else:
                print(f"Critial failure generating caption for {image_path.name}: {str(e)}")
                return False

    return False
