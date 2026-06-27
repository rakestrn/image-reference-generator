import base64
import io
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image


def check_ollama_status(model_name: str = "gemma4:26b") -> bool:
    """Pre-flight check to verify Ollama daemon is running and the model is available.

    Args:
        model_name: Target Ollama model signature.

    Returns:
        True if the daemon is reachable and model is present, False otherwise.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        for m in models:
            if m.get("name") == model_name or m.get("name", "").startswith(model_name + ":"):
                return True
        print(f"Error: Ollama daemon is running, but model '{model_name}' is not found.")
        print(f"Please run: ollama run {model_name}")
        return False
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
                if m.get("name") == model_name or m.get("name", "").startswith(model_name + ":"):
                    return True
            print(f"Error: Ollama daemon started, but model '{model_name}' is not found.")
            print(f"Please run: ollama run {model_name}")
            return False

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


def _encode_image_for_ollama(image_path: Path, max_size: int = 768) -> str:
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


def generate_caption(
    image_path: Path,
    trigger_word: str,
    composition: str = "medium",
    model_name: str = "gemma4:26b",
) -> bool:
    """Interacts with the local Ollama daemon to extract a caption describing the image subject.

    Includes an exponentially decaying retry mechanism. When a Vision model receives its
    first inference call, Ollama must flush VRAM and load up thick multi-GB safetensors,
    often provoking typical client timeouts strictly on the first attempt sequence.

    Args:
        image_path: Absolute pathway to the cropped output JPEG.
        trigger_word: Target identifying sequence to instruct the LLM.
        composition: Geometrical classification ('close-up', 'medium', 'full-body').
        model_name: Target Ollama model signature.

    Returns:
        True if the caption generation and write succeeded, False otherwise.
    """
    encoded_string = _encode_image_for_ollama(image_path)
    if not encoded_string:
        return False

    if composition == "close-up":
        lead = f"A close up portrait of {trigger_word}"
    elif composition == "medium":
        lead = f"A waist-up photo of {trigger_word}"
    elif composition == "full-body":
        lead = f"A full body photo of {trigger_word}"
    else:
        lead = f"A photo of {trigger_word}"

    prompt = (
        f"You are a professional captioning assistant for an image generation model. "
        f"Look at this image of {trigger_word} and write a single, natural, "
        f"descriptive English sentence. "
        f"The sentence must begin with: '{lead}...'. "
        f"Describe {trigger_word}'s pose, expression, background, and provide a highly "
        f"detailed description of their clothing (including style, color, fabric, "
        f"e.g., 'white lace lingerie', 'black thong panties', 'thong bikini') "
        f"and hairstyle (such as cut, color, length, and texture) "
        f"clearly and concisely. Do not add fluff. Focus on accurate identification.\n\n"
        f"End your response with 5-10 comma-separated tags reflecting the most prominent "
        f"visual elements on a new line starting with 'Tags: ' "
        f"(e.g., 'Tags: outdoor, sunny, white lace')."
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

        except (requests.exceptions.RequestException, ValueError) as e:
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


def analyze_lighting(image_path: Path, model_name: str = "gemma4:26b") -> str:
    """Uses Ollama to determine if the image lighting is predominantly 'Indoor' or 'Outdoor'.

    Args:
        image_path: Path to the candidate image.
        model_name: Target Ollama model signature.

    Returns:
        The string 'Indoor' or 'Outdoor'. Defaults to 'Indoor' if unclear or fails.
    """
    encoded_string = _encode_image_for_ollama(image_path)
    if not encoded_string:
        return "Indoor"

    prompt = (
        "You are an expert photography assistant analyzing image data. "
        "Categorize this image's lighting condition as either 'Indoor' or 'Outdoor'. "
        "Respond with only the exact single word 'Indoor' or 'Outdoor'."
    )

    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "images": [encoded_string], "stream": False}

    max_retries = 3
    base_delay = 5.0

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            analysis = result.get("response", "").strip().capitalize()
            if "Outdoor" in analysis:
                return "Outdoor"
            return "Indoor"
        except requests.exceptions.RequestException, ValueError:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))

    return "Indoor"
