# FLUX.1 LoRA Model Generation & Dataset Engineering Optimization Guide

> **Context**: This technical guide bridges the hallucination-reduction work established in [`replicate-interface`](file:///Users/ryanrakestraw/dev/replicate-interface) with the dataset curation pipeline in [`image_reference_generator`](file:///Users/ryanrakestraw/dev/image_reference_generator). It incorporates extensive research on FLUX.1 architecture, Replicate's LoRA trainers (`ostris/flux-dev-lora-trainer` and `replicate/fast-flux-trainer`), Black Forest Labs guidelines, and state-of-the-art diffusion dataset engineering.

---

## 1. Executive Summary & Cross-Project Parity

In `replicate-interface`, significant improvements in generation fidelity and hallucination reduction were achieved through:
1. **Trigger & Facial Anchoring**: Positioning the trigger word within the first 10 tokens and anchoring facial features explicitly (`exactly {trigger_word} face, anchored {trigger_word} facial features`).
2. **Active Hand Anchoring**: Assigning explicit resting poses or physical tasks to prevent mangled, floating, or extra hands.
3. **Background De-crowding & Bokeh**: Specifying uncluttered environments (max 2 ambient elements) and camera parameters (e.g., 85mm f/1.8, shallow depth of field) to isolate the subject cleanly.
4. **Token Budget & Grammar**: Enforcing tight (70–95 word) natural language prompts and stripping redundant buzzwords.
5. **Trait Isolation**: Never describing immutable facial or anatomical traits at inference time because they must be baked into the LoRA.

However, **inference prompt engineering cannot compensate for a fundamentally misaligned training dataset**. If a LoRA model is trained on square-cropped images with severed hands, comma-separated tag lists, or un-captioned background clutter, the model bakes those hallucinations directly into its latent weights.

To achieve maximum likeness, prompt adherence, and zero hallucinations, `image_reference_generator` must align with the **exact mechanics of the FLUX.1 architecture** and **Replicate's training pipelines**.

---

## 2. Core Mechanics of FLUX.1 LoRA Generation

### A. The Dual Text Encoder: Why "Tags" Fail in FLUX
FLUX.1 (Schnell, Dev) departs radically from Stable Diffusion 1.5 and SDXL by utilizing **two parallel text encoders**:
- **OpenAI CLIP ViT-L/14**: Captures high-level visual semantic tokens (77-token limit).
- **Google T5-XXL**: A dense, 4.7B parameter sequence-to-sequence language model with up to 512 tokens.

**The Crucial Distinction**:
- Older models (SD 1.5, Danbooru-based fine-tunes) thrived on comma-separated tag lists (`1girl, close-up, wavy hair, blue eyes, sweater`).
- **FLUX's T5-XXL is a natural language parser.** It understands grammatical syntax, spatial prepositions ("on her lap", "behind the chair"), adjective-noun binding, and clause relationships.
- When captions contain comma lists like `Tags: portrait, woman, close-up, expression, wavy hair`, T5 parses "Tags:" as literal text or encounters fragmented syntax that breaks relational binding.
- **Distribution Shift**: In `replicate-interface`, prompts are formatted as coherent prose (e.g. `[trigger], a medium shot of a woman...`). When training data uses Booru-style tags while inference uses natural prose, the model experiences severe prompt degradation and hallucinations.

### B. The Subtraction Principle (Concept Bleeding vs. Identity Baking)
LoRA training optimizes weights so that:
$$\text{Image Content} - \text{Captioned Content} = \text{Trigger Word Concept}$$

1. **What is explicitly captioned is "explained away"**:
   - If you describe the subject's outfit (`wearing a black silk blouse`), the model learns that the blouse belongs to the text description, not to the trigger word. You can later generate the subject in any outfit.
   - If you describe the background (`in a sunlit kitchen with marble countertops`), the model knows the kitchen is distinct from the person.
   - If you describe camera depth of field (`shallow depth of field with creamy bokeh`), the model does not assume the character only exists in blurry backgrounds.
2. **What is NOT captioned gets baked into the trigger word**:
   - **Identity Traits**: Bone structure, distinctive facial features, natural nose shape, and overall likeness must **never** be described in detail. If you write `with almond brown eyes and high cheekbones`, you dilute the trigger word's association with those features.
   - **Accidental Invariants**: If the subject wears glasses in 100% of photos and you do **not** caption `wearing glasses`, the model bakes glasses permanently into the person's face.

---

## 3. Replicate & FLUX Trainer Specifications

Replicate's standard trainer is [`ostris/flux-dev-lora-trainer`](https://replicate.com/ostris/flux-dev-lora-trainer) (built on `ai-toolkit`), alongside `replicate/fast-flux-trainer`.

| Parameter / Requirement | Specification | Best Practice for Character LoRA |
| :--- | :--- | :--- |
| **Input Format** | Single `.zip` file | Paired image files (`.jpg` or `.png`) and caption files (`.txt`) sharing the exact same basename. |
| **File Types** | `.jpg`, `.jpeg`, `.png` | **Avoid `.webp`** (PIL/torchvision loaders in certain training containers have known decoders bugs with WebP). Standardize to `.jpg` or `.png`. |
| **Dataset Size** | 12 to 30 images | **15–25 curated images** is the empirical sweet spot. Quality and diversity strictly beat raw volume. |
| **Autocaption Flag** | `autocaption: false` | Set to `false` when uploading custom `.txt` files to prevent Replicate's basic BLIP/LLaVA pass from overwriting carefully crafted captions. |
| **Trigger Word Injection** | `trigger_word: "TOK"` | Include the trigger word at the **beginning** of every `.txt` caption. In `ostris`, `[trigger]` can also be used as a replacement token. |
| **LoRA Rank (Dim)** | 16 to 32 | **Rank 16** for style/composition; **Rank 32** for tight facial likeness. Ranks $\ge 64$ drastically increase risk of overfitting and identity rigidity. |
| **Training Steps** | 1,000 to 2,500 | Typically **80–120 steps per image** (e.g. 20 images $\times$ 100 = 2,000 steps). |
| **Resolution / Aspect Ratio** | Multi-Resolution Bucketing | **Multi-aspect ratio bucketing (ARB)** snapping to multiples of 64 or 16 around $\sim 1024^2$ area. **Never center-crop to 1:1 squares.** |

---

## 4. Current Audit & Identified Flaws in `image_reference_generator`

Evaluating the current implementation in `image_reference_generator` against FLUX requirements reveals 6 major technical issues:

### ❌ Flaw 1: The "Tags:" Suffix Anti-Pattern
- **Current Code (`ollama_integration.py`)**:
  ```python
  formatted_tags = "Tags: " + ", ".join(tags_list[:10])
  formatted_caption = f"{desc}\n{formatted_tags}"
  ```
- **Impact**: Injects fragmented keywords that degrade T5-XXL's syntactic attention, causing concept bleeding and prompt drift during generation in `replicate-interface`.

### ❌ Flaw 2: The Severed-Hand / Amputation Hallucination Loop
- **Current Pipeline**: Pass 1 checks exclusively for face presence (`len(results.detections) == 1`). It performs zero checks on whether hands are visible, deformed, blurred, or cut off.
- **Current Cropping (`post_processor.py`)**: Defaults to a 1024x1024 square crop centered on the face:
  ```python
  crop_start_y = int(face_top_y - (face_height * 0.5))
  img_cropped = img_resized[crop_start_y : crop_start_y + crop_size, :]
  ```
- **Impact**: On vertical portraits or half-body shots, a square crop centered on the face routinely cuts directly through forearms, wrists, and fingers. The trainer is fed training images with amputated limbs, teaching the LoRA that the subject naturally has deformed or missing hands.

### ❌ Flaw 3: Default Square Cropping vs. Native Aspect Ratio Bucketing (ARB)
- **Current Pipeline**: `--preserve_aspect_ratio` defaults to `False`.
- **Impact**: Forcing images to 1:1 squares is an obsolete SD 1.5 limitation. FLUX natively uses patchified latents and supports arbitrary aspect ratio buckets (e.g., 832x1216, 896x1152, 1024x1024, 1152x896). Square cropping destroys vertical posture, hand gestures, and environmental context.

### ❌ Flaw 4: Composition Quota Skew ("Portrait Locking")
- **Current Pipeline (`pass3_diversity.py`)**:
  Enforces ~67% close-ups, ~20% medium, and ~13% full-body.
- **Impact**: With two-thirds of the training dataset being tight face close-ups, the LoRA develops **severe portrait locking**. When prompted in `replicate-interface` for a "full body shot" or "wide angle street scene", the model ignores the prompt and generates an extreme close-up.

### ❌ Flaw 5: Missing Photographic & Depth-of-Field Subtraction
- **Current Captioning (`ollama_integration.py`)**: Prompts for clothing and background, but fails to describe:
  - Camera angle (eye-level, low-angle, high-angle)
  - Focal length / Depth of field ("shallow depth of field, creamy background blur")
  - Hand status ("hands resting relaxed on lap", "hands out of frame")
- **Impact**: If background blur is not captioned, FLUX learns that the subject's face is intrinsically linked to blurred backgrounds, preventing clean deep-focus generations.

### ❌ Flaw 6: Legacy Vision Model (`llava:13b`)
- **Current Model**: Uses `llava:13b`, which is known for conversational preambles, weak OCR/watermark detection, and hallucinated hand descriptions.
- **Impact**: Requires heavy regex cleaning and retry logic; modern models (`qwen2.5-vl`, `minicpm-v`, `llama3.2-vision`) provide dramatically higher descriptive precision.

---

## 5. Architectural Improvements for `image_reference_generator`

### Phase 1: Overhaul Captioning for FLUX T5-XXL
1. **Eliminate the `Tags:` section entirely**.
2. **Standardize on a 3-Sentence Natural Language Structure (50–80 words)**:
   - **Sentence 1 (Trigger & Framing)**: `[trigger] [class noun], a [shot type] looking [eye direction] with a [expression] expression.`
   - **Sentence 2 (Wardrobe & Hand Placement)**: Detailed clothing textures/colors, plus **explicit hand position** (e.g. `Hands are resting gently on the wooden table` or `Hands are out of frame`).
   - **Sentence 3 (Setting, Lighting & Camera Optics)**: Environment details, lighting direction, and optical depth (`soft studio lighting, 85mm portrait framing with shallow depth of field and soft background separation`).

#### Concrete Prompt Template for Local Vision Model:
```python
caption_prompt = (
    f"You are an expert diffusion dataset captioner for FLUX.1 models. "
    f"Describe this image of {trigger_word} in 2 to 3 concise, grammatically complete sentences (50-80 words total).\n"
    f"Follow these strict rules:\n"
    f"1. Open with: '{trigger_word} woman, a [close-up / waist-up / full-body] shot...'\n"
    f"2. Explicitly describe clothing (style, color, fabric) and setting.\n"
    f"3. Explicitly describe HANDS: state their exact resting position, or explicitly say 'hands are out of frame' if not visible.\n"
    f"4. Explicitly describe camera depth of field (e.g., 'shallow depth of field with soft blurred background' or 'sharp deep focus').\n"
    f"5. DO NOT describe permanent facial features, eye shape, or bone structure.\n"
    f"6. DO NOT include any 'Tags:' or comma-separated keyword lists. Output pure natural prose only."
)
```

---

### Phase 2: Aspect Ratio Bucketing by Default
1. **Invert the default**: Set `--preserve_aspect_ratio` to `True` by default.
2. **Dimension Snapping**:
   - Maintain original aspect ratio.
   - Scale shortest dimension such that total area $\approx 1024 \times 1024 = 1,048,576\text{ pixels}$.
   - Snap both width and height to the nearest multiple of 64:
     $$\text{new\_w} = \text{round}(w / 64) \times 64, \quad \text{new\_h} = \text{round}(h / 64) \times 64$$
3. **No Destructive Cropping**: Pass the fully bucketed image directly to the dataset folder. Never slice through torsos or wrists.

---

### Phase 3: Rebalance Composition Quotas
Update `pass3_diversity.py` from the legacy 67/20/13 split to a balanced, generalizable distribution:
- **35% Close-up / Headshots**: Anchors facial micro-textures, skin pores, and eye reflections.
- **45% Medium / Waist-up / Cowboy Shots**: Optimal for hand gestures, torso proportions, and outfit interaction.
- **20% Full-body / Environmental Shots**: Teaches the model overall anatomy, posture, and environmental scale.

---

### Phase 4: Hand Quality & Limb Integrity Barrier
In `pass1_filter.py` or `pass2_ranking.py`:
1. Use MediaPipe Hands or YOLO-pose to evaluate visible hands.
2. If hands are detected near the edge of the image, ensure they are not cut in half (e.g., wrist visible with fingers clipped off by the frame). If truncated, either:
   - Prefer uncropped framing, or
   - In the caption pass, explicitly flag: `"hands partially cropped at image edge"`.

---

### Phase 5: Replicate Dataset Exporter & Packaging
Add an automated Replicate training exporter directly into `image_reference_generator`:
1. Validates that every image has an exact matching `.txt` caption.
2. Validates that the trigger word is present in every caption.
3. Automatically creates `{trigger_word}_dataset.zip`.
4. Generates a ready-to-use Replicate API training script or CLI command:
   ```bash
   replicate train ostris/flux-dev-lora-trainer \
     --destination "your-username/model-name" \
     input_images=@my_dataset.zip \
     trigger_word="my_trigger" \
     autocaption=false \
     steps=2000 \
     lora_rank=16
   ```

---

## 6. Implementation Checklist for Next Development Sprint

- [ ] **`ollama_integration.py`**:
  - Remove `Tags: ...` parsing and generation logic.
  - Upgrade prompt to generate 3-sentence FLUX prose with explicit hand and camera depth descriptions.
  - Support modern Ollama vision models (`qwen2.5-vl`, `llama3.2-vision`).
- [ ] **`post_processor.py`**:
  - Set `preserve_aspect_ratio = True` by default.
  - Restrict output formats to `jpg` and `png` (warn or drop `webp` due to trainer container quirks).
  - Prevent aggressive square crop mutilations of hands/bodies.
- [ ] **`pass3_diversity.py`**:
  - Rebalance composition quotas to 35% close-up / 45% medium / 20% full-body.
- [ ] **`main.py`**:
  - Add `--package_zip` flag to bundle the curated output into a Replicate-ready archive.
  - Print training command template with `autocaption=false`.
