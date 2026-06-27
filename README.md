# Image Reference Generator (Anti-Gravity Curation Tool)

An elite, automated Machine Learning dataset preparation pipeline. Designed under an "anti-gravity" philosophy: assume all images in a raw dataset are inherently weighed down by technical flaws and visual redundancy. This tool strategically evaluates and filters raw datasets, providing "lift" to only the most technically optimal, visually diverse images—outputting a perfectly curated portfolio of a customized target size (default 30).

Built entirely for Apple Silicon and modern Python development architectures.

## ✨ Features

- **The 4-Pass Lift Algorithm:**
  - **Pass 1 (Liquidity Filter):** Strictly filters images (JPEG, PNG, WebP) using OpenCV and MediaPipe, parallelized using `ProcessPoolExecutor` for high-speed multi-core execution. Drops any images lacking exactly one detectable human face, checks structural file integrity via PIL, and strictly enforces minimum resolution bounds (shortest edge >= 1024px).
  - **Pass 2 (Technical Ranking):** Ranks the structurally valid images, parallelized across processes. Utilizes Laplacian metrics for overall variance (sharpness), isolates face-specific Laplacian clarity, and applies histogram checks to punish high/low-key clipping.
  - **Pass 3 (Diversity Optimization):** Feeds the top technical candidates into `torchvision:MobileNetV3` batched tensors using a cached model instance to prevent weight loading overhead. Executes a greedy maximization constraint against cosine distances to guarantee maximum dataset variance. Gracefully enforces "Face-First" compositional quotas (e.g. Close-up, Medium, Full-body) scaled dynamically to your requested output count.
  - **Pass 4 (Holistic Set Refinement):** Scrutinizes the output portfolio, removing extreme geometric outliers from the subject median. Re-engages the local Ollama backend to dynamically verify image lighting environments, enforcing a strict 70/30 split between distinct lighting zones. Concludes with a final "aesthetic lift" by systematically upgrading the lowest-scoring 10% while preserving identical compositions. All Pass 4 replacements are strictly mathematically evaluated against existing portfolio components to guarantee robust global deduplication.
- **Automated Post-Processing & OCR:** Leverages `easyocr` to detect and blur only the precise bounding boxes of textual watermarks in the image margins (with padding) to preserve background details. Executes adaptive smart scaling to exactly 1024x1024 pixels, meticulously centered on the facial bounding box, preserving the source image format (JPEG, PNG, or WebP) in output exports, using a robust loading/saving pipeline designed specifically for mounted vaults (e.g. Cryptomator).
- **Local AI Captioning:** Integrates directly with your local Ollama daemon to attach high-precision sidecar `.txt` captions utilizing Vision Language Models (VLMs) like `gemma4:26b`. Dynamically shifts the prompt structure corresponding to the exact geometry of the bounding box (e.g. leading with "A close up portrait..." vs "A full body photo...").

## ⚙️ Prerequisites

1. **Python & Package Manager:** `uv` (Python 3.14+)
2. **Local Vision API:** [Ollama](https://ollama.com/) must be installed and running on port `11434`.
3. **Model Weights:** You must pull your designated vision model before execution:
   ```bash
   ollama run gemma4:26b
   ```

## 🚀 Installation & Setup

Because this project utilizes `uv`, dependencies and virtual environments are handled seamlessly simply by executing the code. No explicit `pip install` loop is required.

Clone the repository and jump into the directory:
```bash
git clone https://github.com/rakestrn/image-reference-generator.git
cd image_reference_generator
```

*(Optional)* Lock the latest dependencies immediately:
```bash
uv lock
```

## 💻 Usage

Point the tool at any raw folder containing potential image candidates. The pipeline will compute the 4-step matrix and export your specified number of optimally distinct files to your output directory.

```bash
uv run curate --input /path/to/raw/directory --output /path/to/save/generated_dataset --trigger my_trigger_word
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | **(Required)** Absolute or relative path to the raw input directory. | None |
| `--output` | **(Required)** Path where the final curated images (preserving original format) and text sidecar files will be systematically saved. | None |
| `--trigger` | **(Required)** Identifying keyword used as the file prefix and injected into the LLM caption. | None |
| `--target_count` | Exact number of total images required in the final generated portfolio. | `30` |
| `--top_n` | Strictness limit bridging Pass 2 and Pass 3. Dynamically ensures enough candidates for the target_count. | `100` |
| `--model` | Target Ollama model signature for caption rendering. | `gemma4:26b` |

## 🧠 LoRA Dataset Engineering Guidelines

This curation pipeline is designed under rigorous machine learning principles to produce production-grade datasets for training **Stable Diffusion (SD1.5, SDXL)** and **Flux** LoRA models.

### 🎯 Subject/Character Focus
*   **Single-Face Constraint (Pass 1):** The pipeline strictly drops images containing multiple faces or zero faces. This is highly optimized for training **Subject, Actor, or Character LoRAs** to ensure the model focuses purely on the target identity without learning secondary subjects.
*   **Subject Consensus (Pass 4.1):** Outliers that deviate significantly from the median feature vector are discarded. This guards against unrelated graphics, non-subject images, or badly cropped portraits corrupting the dataset's identity signature.

### 📐 Resolution, Cropping, and Aspect Ratios
*   **Square Crops:** The post-processor yields exact `1024x1024` JPEGs. This is standard for SDXL and Flux training.
*   **Aspect Ratio Note:** While square crops simplify training on basic scripts, modern training pipelines (e.g. Kohya_ss, EveryDream2, Civitai) utilize *aspect ratio bucketing*. If your raw images have high artistic value in non-square compositions, keep in mind that this tool centers crops specifically on the detected facial bounding box.

### 🎭 Diversity vs. Overfitting
To prevent your LoRA from "overfitting" (generating the same pose or lighting in every generation), the tool utilizes mathematically controlled diversity quotas:
*   **Focal/Composition Quota (Pass 3):** Standardizes the portfolio to a ~67% close-up, ~20% medium-shot, and ~13% full-body split. This solves the common "portrait-locking" failure mode where a LoRA is incapable of generating full-body poses.
*   **Lighting Diversity (Pass 4.2):** Enforces a strict 70/30 split between Indoor and Outdoor lighting conditions via local VLM classification. This prevents the LoRA from binding the subject to a specific lighting environment.

### 🏷️ Captioning Strategy & Concept Bleeding
The local Ollama captioning daemon produces natural language captions starting with the composition-based trigger phrase:
*   **Concept Preservation:** By describing the environment, background, clothing, and pose in detail, the VLM prevents the LoRA from "bleeding" these attributes into the trigger word. For example, if the caption says `A close up portrait of [trigger] wearing a black jacket...`, the model associates the black jacket with the phrase "black jacket" rather than attributing it directly to the `[trigger]` word.
*   **Rare Word Selection:** Ensure that your `--trigger` word is a unique word not commonly found in the base model's dictionary (e.g., use a custom name or short alphanumeric combination) to prevent overwriting existing concepts in the model's vocabulary.

## 🧪 Development & Testing

This project adheres to strict type-hinting, Google-style docstrings, and `ruff` standards.

**Run Linter:**
```bash
uv run ruff check .
```

**Run Test Suite:**
```bash
uv run pytest
```

---
*Developed securely with anti-gravity principles in mind.*
