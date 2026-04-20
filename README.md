# Image Reference Generator (Anti-Gravity Curation Tool)

An elite, automated Machine Learning dataset preparation pipeline. Designed under an "anti-gravity" philosophy: assume all images in a raw dataset are inherently weighed down by technical flaws and visual redundancy. This tool strategically evaluates and filters raw datasets, providing "lift" to only the most technically optimal, visually diverse images—outputting a perfectly curated portfolio of exactly 25 high-quality samples.

Built entirely for Apple Silicon and modern Python development architectures.

## ✨ Features

- **The 3-Pass Lift Algorithm:**
  - **Pass 1 (Liquidity Filter):** Strictly filters images using OpenCV and MediaPipe. Drops any images lacking exactly one detectable human face, checks structural file integrity via PIL, and strictly enforces minimum resolution bounds (shortest edge >= 1024px).
  - **Pass 2 (Technical Ranking):** Ranks the structurally valid images utilizing Laplacian metrics for overall variance (sharpness), isolates face-specific Laplacian clarity, and applies histogram checks to punish high/low-key clipping.
  - **Pass 3 (Diversity Optimization):** Feeds the top technical candidates into `torchvision:MobileNetV3` batched tensors. Executes a greedy maximization constraint against cosine distances to guarantee maximum dataset variance. Gracefully enforces compositional quotas (Close-up, Medium, Full-body).
- **Automated Post-Processing:** Executes adaptive smart scaling to exactly 1024x1024 pixels, meticulously centered on the facial bounding box, exporting as a Safetensors-compliant `quality=95` JPEG.
- **Local AI Captioning:** Integrates directly with your local Ollama daemon to attach high-precision sidecar `.txt` captions utilizing Vision Language Models (VLMs) like `llama3.2-vision`.

## ⚙️ Prerequisites

1. **Python & Package Manager:** `uv` (Python 3.14+)
2. **Local Vision API:** [Ollama](https://ollama.com/) must be installed and running on port `11434`.
3. **Model Weights:** You must pull your designated vision model before execution:
   ```bash
   ollama run llama3.2-vision
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

Point the tool at any raw folder containing potential image candidates. The pipeline will compute the 3-step matrix and export exactly 25 optimally distinct files to your output directory.

```bash
uv run curate --input /path/to/raw/directory --output /path/to/save/generated_dataset --trigger my_trigger_word
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | **(Required)** Absolute or relative path to the raw input directory. | None |
| `--output` | **(Required)** Path where the final 25 JPEGs and text files will be systematically saved. | None |
| `--trigger` | **(Required)** Identifying keyword used as the file prefix and injected into the LLM caption. | None |
| `--top_n` | Strictness limit bridging Pass 2 and Pass 3. Controls how many images calculate feature tensors. | `100` |
| `--model` | Target Ollama model signature for caption rendering. | `llama3.2-vision` |

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
