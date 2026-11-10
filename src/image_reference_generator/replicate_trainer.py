import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

OSTRIS_TRAINER_MODEL = "ostris/flux-dev-lora-trainer"
OSTRIS_DEFAULT_VERSION = "26dce37af90b9d997eeb970d92e47de3064d46c300504ae376c75bef6a9022d2"

FAST_TRAINER_MODEL = "replicate/fast-flux-trainer"
FAST_DEFAULT_VERSION = "e5a5bc821112c107e6ddb8491c5b898f94d06eaca861d1dbf37b29cd69ba8988"


def load_replicate_token() -> str | None:
    """Loads Replicate API token from environment or standard .env locations.

    Supports REPLICATE_API_TOKEN, REPLICATE_API_KEY, or a bare r8_... token line in .env.
    """
    env_keys = ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY", "REPLICATE_TOKEN")
    for k in env_keys:
        token = os.environ.get(k)
        if token and token.strip():
            t = token.strip()
            os.environ["REPLICATE_API_TOKEN"] = t
            return t

    # Load from project .env and ~/.env
    env_paths = [Path.cwd() / ".env", Path.home() / ".env"]
    for env_path in env_paths:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)

    for k in env_keys:
        token = os.environ.get(k)
        if token and token.strip():
            t = token.strip()
            os.environ["REPLICATE_API_TOKEN"] = t
            return t

    # Fallback: check if ~/.env or ./.env has a bare r8_... token line without KEY= prefix
    for env_path in env_paths:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    cleaned = line.strip().strip("'").strip('"')
                    if cleaned.startswith("r8_") and "=" not in cleaned and " " not in cleaned:
                        os.environ["REPLICATE_API_TOKEN"] = cleaned
                        return cleaned
            except Exception:
                pass

    return None


def resolve_trainer_version(client, trainer: Literal["ostris", "fast"] = "ostris") -> str:
    """Resolves the latest trainer version string ('owner/name:version_id')."""
    model_slug = OSTRIS_TRAINER_MODEL if trainer == "ostris" else FAST_TRAINER_MODEL
    fallback_hash = OSTRIS_DEFAULT_VERSION if trainer == "ostris" else FAST_DEFAULT_VERSION

    try:
        trainer_model = client.models.get(model_slug)
        if trainer_model and getattr(trainer_model, "latest_version", None):
            ver_id = trainer_model.latest_version.id
            if ver_id:
                return f"{model_slug}:{ver_id}"
    except Exception:
        pass

    return f"{model_slug}:{fallback_hash}"


def ensure_destination_model(
    client,
    destination: str,
    trigger_word: str,
    visibility: Literal["private", "public"] = "private",
) -> tuple[bool, str, str]:
    """Verifies or creates the target Replicate destination model ('owner/model-name').

    If `destination` does not include an owner prefix ('model-name' only), queries
    the authenticated Replicate account username automatically.

    Args:
        client: Initialized replicate.Client instance.
        destination: Target model string ('owner/name' or 'name').
        trigger_word: Trigger word for model description.
        visibility: 'private' (default) or 'public' when creating a new model.

    Returns:
        tuple of (success, full_destination_slug, message).
    """
    dest = destination.strip().strip("/")
    if not dest:
        return False, "", "Empty destination model name"

    if "/" in dest:
        owner, name = dest.split("/", 1)
    else:
        try:
            account = client.accounts.current()
            owner = account.username
            name = dest
        except Exception as e:
            return (
                False,
                dest,
                f"Could not resolve Replicate username for '{dest}' (pass 'owner/model-name'): {e}",
            )

    full_slug = f"{owner}/{name}"

    try:
        client.models.get(full_slug)
        return True, full_slug, f"Verified existing Replicate destination model: {full_slug}"
    except Exception:
        # Model does not exist yet -> create it
        try:
            client.models.create(
                owner=owner,
                name=name,
                visibility=visibility,
                hardware="gpu-t4",  # Replicate automatically overrides to H100 for FLUX fine-tunes
                description=f"FLUX.1 LoRA model fine-tuned for trigger word '{trigger_word}'",
            )
            return (
                True,
                full_slug,
                f"Created new {visibility} Replicate destination model: {full_slug}",
            )
        except Exception as create_err:
            return (
                False,
                full_slug,
                f"Failed to verify or create destination model '{full_slug}': {create_err}",
            )


def start_replicate_training(
    zip_path: Path,
    destination: str,
    trigger_word: str,
    trainer: Literal["ostris", "fast"] = "ostris",
    steps: int = 3000,
    lora_rank: int = 32,
    learning_rate: float = 0.0004,
    resolution: str = "512,768,1024",
    caption_dropout_rate: float = 0.05,
    visibility: Literal["private", "public"] = "private",
    wait: bool = False,
) -> tuple[bool, str | None, str]:
    """Uploads the packaged dataset zip and starts a FLUX.1 LoRA training job on Replicate.

    Defaults to `ostris/flux-dev-lora-trainer` with maximum-quality hyperparameters
    (`steps=3000`, `lora_rank=32`, `autocaption=False`, `resolution="512,768,1024"`).

    Args:
        zip_path: Path to the validated `{trigger_word}_dataset.zip` archive.
        destination: Target Replicate model ('owner/model-name' or 'model-name').
        trigger_word: Dataset trigger token.
        trainer: 'ostris' (`ostris/flux-dev-lora-trainer`) or 'fast'
            (`replicate/fast-flux-trainer`).
        steps: Training steps (default: 3000).
        lora_rank: LoRA rank dimension for ostris trainer (default: 32).
        learning_rate: Learning rate for ostris trainer (default: 0.0004).
        resolution: Multi-scale resolution buckets for ostris trainer (default: '512,768,1024').
        caption_dropout_rate: Caption dropout rate for ostris trainer (default: 0.05).
        visibility: Visibility if creating a new destination model ('private' or 'public').
        wait: If True, polls training status until completion.

    Returns:
        tuple of (success, training_id, status_message).
    """
    if not zip_path or not zip_path.is_file():
        return False, None, f"Dataset zip archive not found at: {zip_path}"

    token = load_replicate_token()
    if not token:
        return (
            False,
            None,
            "REPLICATE_API_TOKEN not found in environment or ~/.env. "
            "Please set REPLICATE_API_TOKEN before running API training.",
        )

    import replicate

    client = replicate.Client(api_token=token)

    ok_dest, full_destination, dest_msg = ensure_destination_model(
        client, destination, trigger_word, visibility=visibility
    )
    if not ok_dest:
        return False, None, dest_msg
    print(f" -> ✅ {dest_msg}")

    version_str = resolve_trainer_version(client, trainer=trainer)
    print(f" -> 🎯 Using trainer version: {version_str}")
    print(f" -> 📦 Uploading {zip_path.name} and launching training job...")

    try:
        with open(zip_path, "rb") as zip_file:
            if trainer == "fast":
                train_input = {
                    "input_images": zip_file,
                    "trigger_word": trigger_word,
                    "lora_type": "subject",
                    "training_steps": steps,
                }
            else:
                train_input = {
                    "input_images": zip_file,
                    "trigger_word": trigger_word,
                    "autocaption": False,
                    "steps": steps,
                    "lora_rank": lora_rank,
                    "optimizer": "adamw8bit",
                    "learning_rate": learning_rate,
                    "resolution": resolution,
                    "caption_dropout_rate": caption_dropout_rate,
                }

            training = client.trainings.create(
                version=version_str,
                destination=full_destination,
                input=train_input,
            )

        training_id = getattr(training, "id", None)
        status = getattr(training, "status", "starting")
        dashboard_url = (
            f"https://replicate.com/p/{training_id}"
            if training_id
            else "https://replicate.com/trainings"
        )

        print("\n" + "=" * 80)
        print("🚀 Replicate FLUX.1 LoRA Training Started!")
        print("=" * 80)
        print(f"Trainer           : {version_str.split(':')[0]}")
        print(f"Destination Model : https://replicate.com/{full_destination}")
        print(f"Training ID       : {training_id}")
        print(f"Initial Status    : {status}")
        print(f"Live Dashboard    : {dashboard_url}")
        if trainer == "ostris":
            print(
                f"Hyperparameters   : steps={steps}, lora_rank={lora_rank}, "
                f"lr={learning_rate}, autocaption=False, resolution='{resolution}'"
            )
        else:
            print(f"Hyperparameters   : training_steps={steps}, lora_type='subject'")
        print("=" * 80 + "\n")

        if wait and training_id:
            print(
                "⏳ Waiting for Replicate training to complete "
                "(Ctrl+C to detach; job continues on Replicate)..."
            )
            last_status = status
            while status not in ("succeeded", "failed", "canceled"):
                time.sleep(15)
                t_obj = client.trainings.get(training_id)
                status = getattr(t_obj, "status", status)
                if status != last_status:
                    print(f"   [Replicate Status]: {status}")
                    last_status = status

            if status == "succeeded":
                return (
                    True,
                    training_id,
                    f"Training {training_id} succeeded! "
                    f"Model ready at https://replicate.com/{full_destination}",
                )
            return False, training_id, f"Training {training_id} ended with status: {status}"

        return (
            True,
            training_id,
            f"Training {training_id} launched ({status}). Track progress at {dashboard_url}",
        )

    except Exception as e:
        return False, None, f"Failed to launch Replicate training: {e}"
