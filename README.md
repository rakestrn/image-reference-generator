# Image Reference Generator (Anti-Gravity Curation Tool)

An elite, automated Machine Learning dataset preparation pipeline optimized for **FLUX.1 LoRA training** (Replicate `ostris/flux-dev-lora-trainer`, `replicate/fast-flux-trainer`) and modern diffusion architectures. Designed under an "anti-gravity" philosophy: assume all images in a raw dataset are inherently weighed down by technical flaws, amputations, and visual redundancy. This tool strategically evaluates and filters raw datasets, providing "lift" to only the most technically optimal, visually diverse images—outputting a curated portfolio of a customized target size (default: 30).

Built natively for Apple Silicon (Metal/MPS) and CUDA-accelerated modern Python development architectures.

---

## ✨ Core Pipeline Features

### 🔄 The 4-Pass Lift Algorithm

1. **Pass 1: Initial Liquidity Filter & Limb/Pose Integrity Barrier**
   - **Single-Face Constraint:** Strictly filters raw images (JPEG, PNG, WebP) using MediaPipe Tasks face detection. Drops images with zero faces or multiple subjects to prevent subject ambiguity during LoRA training.
   - **Limb & Hand Integrity:** Evaluates visible hands using MediaPipe Hand Landmarker. Flags hands truncated at frame boundaries (e.g. wrist inside with severed fingers) to prevent the LoRA from baking amputated limb hallucinations into its latent weights.
   - **Temporal & EXIF Extraction:** Reads embedded EXIF metadata (`DateTimeOriginal`, `CreateDate`) or filesystem mtime to extract exact capture timestamps and calendar years for chronological balancing.
   - **Head Pose Estimation:** Analyzes 3D facial landmark geometry to determine head orientation (`frontal`, `three-quarter-left`, `three-quarter-right`, `profile`) for rotational diversity.
   - **Integrity & Resolution Check:** Verifies image headers and enforces configurable minimum resolution bounds (`--min_resolution`, default: 1024px shortest edge).
   - **Calibrated BlazeFace Composition Ratios:** Uses calibrated MediaPipe eye-to-chin facial area ratios (`> 6%` close-up, `> 1.5%` medium, `<= 1.5%` full-body) for accurate shot classification.
   - **Parallelized:** Optimized using `ProcessPoolExecutor` across all available CPU cores with case-insensitive path deduplication.

2. **Pass 2: Scale-Invariant Technical Ranking & Temporal Stratification**
   - **Canonical Resolution Normalization (1024px):** Downscales images > 1024px to a standardized canonical reference scale before computing frequency metrics. Eliminates the artificial megapixel bias where modern 48MP smartphone computational photography crushes older 8–12MP DSLR/camera photos.
   - **Laplacian Sharpness:** Computes variance of Laplacian across the canonical image and isolated facial ROI.
   - **Robust 95th-Percentile Scaling:** Prevents extreme contrast/texture outliers from compressing normalized score spreads.
   - **Exposure Safety:** Calculates histogram balance to penalize overexposed highlights and underexposed shadows.
   - **Contrast, Colorfulness & Noise:** Computes RMS contrast, Hasler & Suesstrunk colorfulness, and high-frequency noise penalties.
   - **Stratified Temporal Allocation:** When `--temporal_balance` is active, distributes candidate quotas across distinct calendar years so earlier eras are guaranteed representation in the top N pool.

3. **Pass 3: Visual Diversity Optimization (CLIP & Multi-Strata Balance)**
   - **Neural Feature Extraction:** Feeds candidate images in batched tensors through `openai/clip-vit-base-patch32` with hardware acceleration (Apple Silicon `mps` or NVIDIA `cuda`).
   - **Greedy Cosine Distance Maximization:** Selects images that maximize mutual visual diversity while satisfying balanced multi-objective quotas:
     - **Compositional Quotas:** 35% close-up / 45% medium / 20% full-body framing.
     - **Chronological Balance:** Enforces era quotas so no single year monopolizes the final selection.
     - **Head Pose Variation:** Actively rewards unrepresented facial orientations (frontal, 3/4 angles, profile).
     - Graceful escape-hatch automatically reallocates quotas if specific focal lengths or eras are deficient.
   - **Zero-Overhead Caching:** Feature vectors are cached directly on candidate metrics for instant downstream reuse.

4. **Pass 4: Holistic Set Refinement & Consensus**
   - **Subject Consensus Check:** Identifies and removes semantic/geometric outliers furthest from the median dataset feature vector, substituting with top unused candidates while preserving composition.
   - **Single-Subject Verification & Lighting Balancing (70/30 Split):** Uses local Ollama Vision to strictly enforce single-person exclusivity (rejecting multi-person photos or companions that bypassed low-confidence face detection) while categorizing lighting (`Indoor` vs `Outdoor`) in a single unified VLM forward pass.
   - **Aesthetic Lift:** Replaces the lowest-scoring 10% with higher-scoring non-duplicate candidates.
   - **Vectorized Deduplication:** Mathematically prevents near-duplicate insertions using vectorized batch cosine distance matrices.

---

### 🛡️ 3-Pass Closed-Loop Captioning & Audit Engine

FLUX.1 LoRA training is exceptionally sensitive to caption quality. Booru comma lists, token stutter, and conversational filler (*"adding mystery"*, *"accentuates her figure"*) confuse Google T5-XXL text conditioning. To guarantee spotless captions, the pipeline employs a **3-pass closed-loop audit & self-correction engine**:

```
[ Image ] ──► [ Pass 1: Vision Extraction ]
                         │
                         ▼ (Raw Description)
              [ Clean & Deduplicate Lead ]
                         │
                         ▼ (Formatted Caption)
              [ Pass 2: Rule-Based Audit ] ──► ✅ Passes ──► [ Save .txt ]
                         │
                         ▼ ❌ Violations Detected
              [ Pass 3: Educated Retry Loop ]
                - Injects exact violation feedback
                - Escalates temperature (0.2 -> 0.5)
                - Up to 2 retries
                         │
                         ▼ (Fallback if exhausted)
              [ Deterministic Sanitizer ] ─────────────► [ Save .txt ]
```

1. **Pass 1: Strict 5-Part FLUX Sequence & Photography Lighting Rubric**
   - Instructs the VLM to describe physical visual facts in strict order:
     1. **Framing & Camera Angle** (e.g., close-up shot, eye-level, wide angle)
     2. **Pose, Action, Explicit Hand Placement & Expression** (strictly omitting facial identity traits, bone structure, eye color, hair color, or race)
     3. **Clothing & Accessories** (colors, textures, fit, eyewear)
     4. **Setting, Background Elements & Optical Depth of Field**
     5. **Technical Photography Lighting Clause** (key light direction + diffusion/hardness + color temperature/fill expressed in a single concise terminal clause)
   - `clean_and_format_caption` completely decouples model text from the trigger prefix and strips rubric headers or semicolons, guaranteeing **0% double-lead stutter**.

2. **Pass 2: Deterministic Rule-Based Quality Audit (`audit_caption`)**
   - Executes in **< 5ms** without LLM non-determinism, strictly auditing against FLUX standards:
     - **Stutter & Repetition:** Trigger word count == 1; zero duplicate framing phrases (`close-up shot`, `waist-up shot`, `full-body shot`).
     - **Banned Clichés, Quality & Lighting Buzzwords:** Flags and rejects phrases like *"photorealistic"*, *"masterpiece"*, *"dramatic lighting"*, *"cinematic lighting"*, *"accentuates her figure"*, *"adding a sense of mystery"*, *"focusing the viewer's attention"*, *"is captured in"*.
     - **Hand Placement Check:** Verifies hands are explicitly described (exact resting location or explicitly *"hands are out of frame"*).
     - **Token Budget:** Between 20 and 95 words; 1 to 3 complete sentences with terminal punctuation.

3. **Pass 3: Educated Re-Prompt Loop**
   - When Pass 2 flags violations, it feeds an **educated correction prompt** directly to the vision model citing the exact failure reasons (e.g. `⚠️ PREVIOUS ATTEMPT FAILED: Banned phrase 'accentuates her figure' detected. State hand placement explicitly.`).
   - Escalates temperature (`0.2` $\rightarrow$ `0.35` $\rightarrow$ `0.50`) to break deterministic loops.
   - If retries are exhausted, applies `sanitize_caption_deterministic` as a graceful fallback so no image is dropped.

---

### 🖼️ Multi-Aspect Ratio Bucketing (ARB) & Post-Processing
- **Aspect Ratio Bucketing by Default:** Downscales images larger than $1024^2$ maintaining their native aspect ratio so total area $\approx 1024^2$ pixels, snapping both dimensions to the nearest multiple of 64 (e.g. `832x1216`, `896x1152`, `1024x1024`). Never artificially upscales smaller images (if `--min_resolution` is lowered below 1024, smaller images snap to multiples of 64 at native scale for Replicate's `512`/`768` buckets). Prevents destructive cropping of limbs and torsos. Override with `--crop_square` if 1:1 cropping is required.
- **Selective Watermark Blurring:** Uses `easyocr` to detect text in margin zones (top/bottom 15%) and applies tight Gaussian blur strictly to text bounding boxes. Alternatively, `--drop_watermarked` drops watermarked images entirely.
- **Format Standardization:** Standardizes output to `.jpg` (default) or `.png`. Alerts on `.webp` inputs/outputs due to known decoder container bugs in certain LoRA trainers.

---

### 📦 Replicate Dataset Packaging & Direct API Training
- **Automated Archive Creation:** Passing `--package_zip` or `--train_replicate` automatically validates image-caption pairs (ensuring matching `.txt` files with the trigger word present) and bundles them into `{trigger_word}_dataset.zip`.
- **Direct Replicate API Fine-Tuning (`--train_replicate` / `--train_only`):** Reads `REPLICATE_API_TOKEN` from your environment or `~/.env`, verifies or automatically creates your private destination model on Replicate, uploads the packaged dataset zip, and launches a remote H100 training run.
- **Trainer Selection:**
  - **`ostris` (`ostris/flux-dev-lora-trainer`, Default):** Maximum identity & lighting fidelity. Configured with `autocaption=False` (strictly honoring our 5-part `.txt` captions), `steps=3000`, `lora_rank=32`, `resolution="512,768,1024"`, and `caption_dropout_rate=0.05`.
  - **`fast` (`replicate/fast-flux-trainer`):** High-speed subject trainer (`lora_type="subject"`, `training_steps=3000`).

---

## ⚙️ Prerequisites & Setup

1. **Python Environment:** Python 3.14+ managed with `uv`.
2. **Local Ollama Daemon:** [Ollama](https://ollama.com/) running locally on port `11434`.
3. **Replicate API Key (Optional, for `--train_replicate` / `--train_only`):** Add `REPLICATE_API_TOKEN=r8_...` to `~/.env` or project `.env`.
4. **Vision Model Selection:**
   - **`redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest` (Recommended Default):** Uncensored/abliterated Qwen2.5-VL 7B model (~1.3s/image on Apple Silicon, 6GB VRAM, zero refusals or conversational hedging, tight adherence to the 5-part FLUX rubric).
   - **`qwen2.5vl:32b` (High-detail alternative for 64GB Mac):** Flagship visual detail, 4K native dynamic resolution.
   - **`qwen2.5vl:7b` (Standard 7B option):** Fast inference, low VRAM footprint.

Pull your model of choice:
```bash
ollama pull redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest
# or
ollama pull qwen2.5vl:32b
```

### Installation

```bash
git clone https://github.com/rakestrn/image-reference-generator.git
cd image_reference_generator
uv sync
```

---

## 💻 Usage & Workflows

### 1. Full Dataset Curation + Direct Replicate API Training
Run the end-to-end 4-pass curation pipeline, captioning, packaging, and immediately launch a 3,000-step `ostris/flux-dev-lora-trainer` job on Replicate:
```bash
uv run curate \
  --input /path/to/raw/images \
  --output /path/to/curated_portfolio \
  --trigger my_trigger_word \
  --target_count 30 \
  --train_replicate \
  --replicate_destination your-username/my-flux-lora
```

### 2. Launch Replicate Training on an Existing Curated Directory
Already curated and captioned your folder? Package and push it directly to Replicate without re-running curation:
```bash
uv run curate \
  --train_only /path/to/curated_portfolio \
  --trigger my_trigger_word \
  --replicate_destination your-username/my-flux-lora \
  --trainer ostris \
  --steps 3000 \
  --lora_rank 32
```

### 3. In-Place Dataset Re-Captioning & Repair
Re-caption an already-curated directory of images using the 3-pass engine and repackage for Replicate without re-running passes 1–3:
```bash
uv run curate \
  --recaption_dir /path/to/curated_portfolio \
  --trigger my_trigger_word \
  --model redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest \
  --package_zip
```

### 4. Standalone Dataset Quality Audit
Instantly audit all caption `.txt` files in a directory and print a diagnostic report:
```bash
uv run curate \
  --audit_only /path/to/curated_portfolio \
  --trigger my_trigger_word
```

---

### CLI Options Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | Path | `None` | Input directory containing raw images (required for full curation). |
| `--output` | Path | `None` | Target directory for curated images and `.txt` files (required for curation). |
| `--trigger` | str | *(Required)* | Trigger keyword prefixed in filenames and injected into captions. |
| `--recaption_dir` | Path | `None` | Re-caption an existing portfolio directory in place with the 3-pass engine. |
| `--audit_only` | Path | `None` | Audit an existing directory of `.txt` files and print a diagnostic report. |
| `--train_replicate` | flag | `False` | Automatically package dataset and launch FLUX.1 LoRA training on Replicate via API. |
| `--train_only` | Path | `None` | Package and launch Replicate API training on an existing curated directory. |
| `--replicate_destination` | str | `flux-{trigger}-lora` | Destination Replicate model (`owner/model-name` or `model-name`). |
| `--trainer` | str | `ostris` | Replicate trainer: `ostris` (`ostris/flux-dev-lora-trainer`) or `fast` (`replicate/fast-flux-trainer`). |
| `--steps` | int | `3000` | Number of FLUX.1 LoRA training steps on Replicate. |
| `--lora_rank` | int | `32` | LoRA rank dimension for `ostris` trainer. |
| `--wait_for_training` | flag | `False` | Poll Replicate training status in terminal until completion. |
| `--model` | str | `redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest` | Ollama vision model (`redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest`, `qwen2.5vl:32b`, `qwen2.5vl:7b`). |
| `--target_count` | int | `30` | Target number of images in final curated portfolio. |
| `--top_n` | int | `100` | Pool size limit for technical ranking before diversity pass. |
| `--min_resolution` | int | `1024` | Minimum allowed shortest edge in pixels for Pass 1 filter. |
| `--format` | str | `jpg` | Output format: `jpg`, `png`, `webp`, or `preserve` (default: `jpg`). |
| `--preserve_aspect_ratio` | bool | `True` | Multi-aspect ratio bucketing snapping to 64px increments around ~1024² area. |
| `--crop_square` | flag | `False` | Force 1024x1024 square crop instead of aspect ratio bucketing. |
| `--temporal_balance` | bool | `True` | Stratify candidates across chronological years/eras to eliminate recency bias. |
| `--era_window` | str | `None` | Filter candidates to specific era window (e.g., `'2020-2024'`, `'2023'`, `'all'`). |
| `--package_zip` | flag | `False` | Package validated dataset into a Replicate-ready training zip archive. |
| `--drop_watermarked` | flag | `False` | Drop watermarked images entirely instead of selective blurring. |
| `--favor_no_glasses` | bool | `True` | Strongly favor pictures without sunglasses to maximize facial/eye identity in LoRAs. |
| `--drop_sunglasses` | flag | `False` | Strictly drop all photos with sunglasses or tinted eyewear from the final dataset. |
| `--batch_size` | int | `32` | Tensor batch size for CLIP neural feature extraction. |

---

## 🧪 Testing & Code Quality

Run the test suite (21 unit and integration tests):
```bash
uv run pytest
```

Run linter checks:
```bash
uv run ruff check .
```
