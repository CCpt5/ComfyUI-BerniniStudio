"""
BerniniStudio v2.3

All-in-one ComfyUI conditioning node for Bernini (ByteDance).
Wraps text encoding, VAE encoding of source video / reference images,
and context_latent construction into a single node with built-in
Ollama prompt enhancement using Bernini's official task templates.

Outputs standard CONDITIONING + LATENT ready for SamplerCustom with
Bernini HIGH/LOW models (loaded via UNETLoader).

Architecture note: Bernini conditions through 'context_latents' --
VAE-encoded reference material appended as extra tokens in the
transformer's attention sequence, each with a unique source_id
in the RoPE positional encoding. This is fundamentally different
from WanAnimate's ref_latent conditioning or SVI Pro's image_embeds.

Requires Kijai's PR #14216 (or equivalent) for ComfyUI core model
support -- specifically comfy.conds.CONDList and the context_latents
pathway in comfy/ldm/wan/model.py and comfy/model_base.py.
"""

import json
import logging
import os
import re
import torch

import comfy.model_management as mm
import comfy.utils
from comfy.utils import common_upscale

try:
    import node_helpers
except ImportError:
    node_helpers = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

log = logging.getLogger("BerniniStudio")


def _llm_headers(api_format="Ollama", include_json=True):
    """Build HTTP headers for Ollama or OpenAI-compatible prompt enhancement.

    For OpenAI / vLLM mode:
    - Reads OPENAI_API_KEY from environment (never stored in workflows).
    - Adds Authorization: Bearer <key> when available.
    - Optionally reads OPENAI_ORG_ID / OPENAI_PROJECT_ID if set.
    Local vLLM/LiteLLM without auth works unchanged (no key -> no header).
    """
    headers = {}
    if include_json:
        headers["Content-Type"] = "application/json"

    if api_format != "Ollama":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        org_id = os.environ.get("OPENAI_ORG_ID", "").strip()
        if org_id:
            headers["OpenAI-Organization"] = org_id

        project_id = os.environ.get("OPENAI_PROJECT_ID", "").strip()
        if project_id:
            headers["OpenAI-Project"] = project_id

    return headers


def _is_openai_new_completion_model(model):
    """True for OpenAI chat models that reject the legacy max_tokens field.

    GPT-5 and reasoning-family models on /v1/chat/completions expect
    max_completion_tokens instead of max_tokens, and may reject custom
    temperature values, so we omit temperature for these models.
    """
    model = (model or "").strip().lower()
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _apply_openai_generation_options(payload, model, max_tokens=2048, temperature=0.7):
    """Add generation-limit options to an OpenAI-compatible payload.

    - OpenAI GPT-5 / o-series: max_completion_tokens, omit temperature.
    - Older OpenAI-compatible models / vLLM: max_tokens + temperature.
    """
    if _is_openai_new_completion_model(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature


# =========================================================================
# Task types and system prompts (verbatim from bytedance/Bernini prompt_enhancer.py)
# =========================================================================

TASK_TYPES = [
    "v2v", "rv2v", "r2v", "t2v", "t2i", "r2i", "i2i",
    "i2v", "mv2v", "vi2v", "ads2v", "vrc2v",
]

SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": (
        "You are a helpful assistant for editing. "
        "You may need to adjust the subject's action or position."
    ),
    "mv2v": (
        "You are a helpful assistant for editing. "
        "You might need to adjust the video's style, lighting, colors, "
        "textures, and the subject's pose or action."
    ),
}

# Bernini's default negative prompt (from cli.py DEFAULT_NEG_PROMPT)
DEFAULT_NEG_PROMPT = (
    "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c"
    "\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce"
    "\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c"
    "\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28"
    "\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0cJPEG\u538b\u7f29\u6b8b\u7559"
    "\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59"
    "\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8"
    "\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62"
    "\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c\u5f62\u6001\u7578\u5f62\u7684"
    "\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d"
    "\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c"
    "\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012"
    "\u7740\u8d70"
)

# Task -> recommended guidance_mode (from gradio_demo.py GUIDANCE_MODE_BY_TASK)
GUIDANCE_MODE_BY_TASK = {
    "t2i": "t2v_apg",
    "t2v": "t2v_apg",
    "i2i": "v2v",
    "i2v": "r2v_apg",
    "v2v": "v2v_apg",
    "mv2v": "v2v_apg",
    "r2v": "r2v_apg",
    "rv2v": "rv2v",
    "ads2v": "v2v_apg",
    "vi2v": "v2v_apg",
    "vrc2v": "rv2v",
}

# Task -> required inputs (from gradio_demo.py TASK_INPUTS)
TASK_INPUTS = {
    "t2i": {"video": False, "image_role": "none", "images": False},
    "t2v": {"video": False, "image_role": "none", "images": False},
    "i2i": {"video": False, "image_role": "source", "images": False},
    "i2v": {"video": False, "image_role": "reference", "images": True},
    "v2v": {"video": True, "image_role": "none", "images": False},
    "mv2v": {"video": True, "image_role": "none", "images": False},
    "r2v": {"video": False, "image_role": "reference", "images": True},
    "rv2v": {"video": True, "image_role": "reference", "images": True},
    "ads2v": {"video": True, "image_role": "reference", "images": True},
    "vi2v": {"video": True, "image_role": "reference", "images": True},
    "vrc2v": {"video": True, "image_role": "reference", "images": True},
}


def _get_system_prompt(task_type):
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])


# =========================================================================
# Ollama enhancement templates (from bytedance/Bernini prompt_enhancer.py)
# These are the FULL templates used by Bernini's PromptEnhancer.
# The t2v/t2i templates are in Chinese (designed for GPT-5.4); they will
# produce lower quality output with English-only local models. The v2v,
# i2i, rv2v, r2v, r2i templates are in English and work well with any LLM.
# =========================================================================

ENHANCE_TEMPLATES = {
    "v2v": """Task: Video Editing
# ROLE
You are an expert Video-to-Video (V2V) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source video frames to generate a detailed V2V editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: Frames of the source video are provided.

# CORE GENERATION RULE
Unless specified otherwise by the task type, your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed. Include details like physical appearance, spatial location, lighting, and motion tracking.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references to characters, objects, outfits, or styles, you MUST replace them with specific, well-known, named instances that match the existing visual style.
Note: Describe it naturally, for example, "Add an apple. The table and curtains remain unchanged."

# TASK CATEGORIES & TEMPLATES
Analyze the instruction and determine the specific editing task type, then generate using the corresponding template:
1. Replacement: "Replace [original element] with [new element]."
2. Addition: "Add [element] + [location/action]."
3. Object/Background Removal: "Delete [object description] + [location]."
4. Subtitle Removal: "Remove subtitles from the video."
5. Depth-to-Video: "Generate video with depth map. [Detailed description]"
6. Sketch-to-Video: Provide a detailed T2V-style description.
7. Colorization: "Colorize the video. [Scene and color description]"
8. Inpainting: "Inpaint this video. [Scene description to fill]"
9. Detection: "Detect the mask region of the [specific object]."
10. Stylization: "Convert the video to [style name]: [brief style details]."
11. Mixed Tasks: Seamlessly integrate all requirements into a single, cohesive instruction.
12. Camera Movement: "Apply camera motion: [Camera Movement Description]"
13. Change Camera Perspective: "Switch the camera to a [first/third]-person perspective" or "Move the camera [description]"
14. Change Focus: "Shift the focus to [subject], making it sharp. Blur [objects to be blurred]."
15. Other Tasks: Generate logically based on the specific situation.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the video.""",

    "i2i": """Task: Image Editing
# ROLE
You are an expert Image-to-Image (I2I) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source image to generate a detailed I2I editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: The source image is provided.

# CORE GENERATION RULE
Your generated prompt MUST follow this structure:
1. Modifications: Specifically describe what needs to be changed, including physical appearance, spatial location, lighting, shadows, and perspective consistency.
2. Preservations: Explicitly describe key visual elements that MUST remain unchanged.
3. Concretization: Replace vague references with specific, well-known, named instances matching the existing visual style.
Describe it naturally, e.g., "Add an apple. The table and curtains remain unchanged."

# TASK CATEGORIES
1. Replacement: "Replace [original] with [new]."
2. Addition: "Add [element] + [location]."
3. Removal: "Delete [object] + [location]."
4. Text/Watermark Removal: "Remove [text/watermark] from the image."
5. Depth-to-Image: "Generate image with depth map. [Target description]"
6. Sketch-to-Image: Detailed T2I description.
7. Colorization: "Colorize the image. [Scene and colors]"
8. Inpainting: "Inpaint this image. [Region description]"
9. Outpainting: "Extend the image [direction]. [Extended content]"
10. Detection: "Detect the mask region of the [specific object]."
11. Stylization: "Convert the image to [style]: [details]."
12. Relighting: "Relight the image: [direction, temperature, intensity, shadows]."
13. Pose/Expression: "Change the [subject]'s [pose/expression] to [target]."
14. Viewpoint Change: "View the scene from [target viewpoint]."
15. Focus Change: "Shift focus to [subject]. Blur [objects] with bokeh."
16. Mixed: Integrate all requirements cohesively.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not imagine things not in the image.""",

    "rv2v": """You are an expert at writing prompts for reference-image-guided video editing. I'm providing you with:
1. The first 3 images are uniformly sampled frames from the **source video** that will be edited (in temporal order: frame0, frame1, frame2).
2. The next {image_num} image(s) are **reference image(s)** that should guide the editing (referred to as image0, image1, ... in order).
3. An original editing instruction.

Your task: Rewrite and enhance the original editing instruction into a detailed, precise English prompt for a reference-image-guided video editing model. The output is a single paragraph: **editing instruction + detailed description of the target edited video**.

Follow these rules strictly:
1. Output format: editing instruction followed by detailed target video description, as one continuous paragraph.
2. Match the edit type: use the verb matching the intent -- "Replace...", "Remove...", "Add...", "Restyle...", "Transfer the motion/pose of... to...", etc.
3. Add != Replace: for addition tasks, write as additions, not replacements.
4. Allow natural shape/size differences.
5. Describe the target video directly: don't use "after editing..." or "in the edited video...".
6. Faithful reference appearance: match what's visible in the reference image.
7. Screen-perspective left/right: all directions from camera perspective.
8. Preserve unchanged elements explicitly: camera framing, lighting, background, motion, etc.
9. For style/motion references: describe resulting style/motion in concrete language.
10. No parentheses in output.
11. English only.
12. Keep detail level similar to this example:

"Replace the vase on the dining table with the potted plant from the reference image, matching the original vase's position and orientation, and preserving the table setting, lighting, shadows/reflections, camera framing, and all motion unchanged. A bright, modern dining/living room in soft daylight with a light-wood rectangular dining table..."

Return ONLY a JSON object with key "rewritten_text". No extra text.

Original instruction:
{user_prompt}""",

    "r2v": """You are an expert at writing subject-driven video generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) that will appear in the video (referred to as image0, image1, image2, ... in order).
2. An original video description text.

Your task is to rewrite the original description into TWO parts concatenated together:

**Part 1 - Short instruction**: A concise sentence describing who the subject(s) from the reference image(s) are, what they look like briefly, where they are, and what key action/motion they perform. Reference the subject(s) using "image0", "image1", etc.

**Part 2 - Long instruction**: A detailed "Generate a video where..." paragraph that describes:
- The subject(s) with detailed appearance (hair, clothing, accessories, expression), referencing as "the person/man/woman from image0" etc.
- The scene/environment in detail (background, lighting, objects, atmosphere).
- The motion and actions in a step-by-step temporal sequence.

Requirements:
- Reference each subject using "image0", "image1", etc.
- Appearance description based on what you actually see in the reference image(s).
- English only.
- Return ONLY a JSON object with key "rewritten_text".

Original description:
{user_prompt}""",

    "r2i": """You are an expert at writing subject-driven image generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) (referred to as image0, image1, ... in order).
2. An original image description text.

Rewrite into TWO parts concatenated together:
**Part 1 - Short instruction**: Concise sentence about the subjects, their appearance, location, and composition.
**Part 2 - Long instruction**: Detailed "Generate an image where..." paragraph with appearance, scene, and composition.

Reference subjects using "image0", "image1", etc. English only.
Return ONLY a JSON object with key "rewritten_text".

Original description:
{user_prompt}""",

    "t2v": """You are a film director enhancing a text-to-video prompt. Add cinematic elements: lighting (source, intensity, angle), camera (shot size, angle, composition), color tone, and detailed motion sequences. Keep the original intent. Output 60-200 words, English only.

If the prompt describes a specific style (anime, 2D illustration, etc.), do NOT add film/cinematography aesthetics that contradict it. For non-realistic styles, focus on composition, color palette, and motion only.

Original prompt: {user_prompt}""",

    "t2i": """You are a photographer enhancing a text-to-image prompt. Add photographic elements: lighting (source, intensity, angle), camera (shot size, angle, composition), color tone, and spatial composition. Do NOT describe any motion, camera movement, or temporal sequences -- this is a static image. Output 60-200 words, English only.

If the prompt describes a non-photographic style, focus on composition, color, and spatial arrangement only.

Original prompt: {user_prompt}""",

    "vi2v": """Task: Video Content Propagation / Reference Insertion
User's editing instruction: "{user_prompt}"

Determine which sub-task applies based on the instruction:
- Propagation: Return exactly "edit the video following the first frame."
- Reference insertion: Format as "Integrate the [object] from the image into the video in a reasonable way."
- Reference replacement: Describe replacing the source object with the reference object.

Output ONLY the final English prompt.""",

    "ads2v": """Task: Ads Insertion in Video
User's instruction: "{user_prompt}"

Generate a concise English ad insertion instruction in one sentence.
Example: "Add Starbucks Latte wallpaper on the second floor across the street"

Output ONLY the final English prompt.""",

    "i2v": """Task: Image-to-Video Generation
User's prompt: "{user_prompt}"

Generate an English prompt describing the video content (actions, camera movement, scene).
Describe motion and temporal flow. Output ONLY the final English prompt.""",

    "default": """You are a helpful assistant that enhances prompts for video generation and editing. Rewrite the following instruction to be more detailed and specific, adding visual details, motion descriptions, and preservation notes where appropriate. English only.

Instruction: {user_prompt}""",
}


def _get_enhance_template(task_type):
    return ENHANCE_TEMPLATES.get(task_type, ENHANCE_TEMPLATES["default"])


# =========================================================================
# Image/video helpers (from Kijai's PR)
# =========================================================================

def _resize_long_edge(image, max_size, stride=16):
    """Resize preserving aspect so long edge <= max_size, snap to stride."""
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return common_upscale(
        image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled"
    ).movedim(1, -1)


def _load_slot_image(filename):
    """Load an image from ComfyUI's input directory as a [1,H,W,3] float tensor.
    Used for the built-in reference slots (drag-and-drop in the editor UI).
    Mirrors LoadImage's loading path (EXIF transpose, RGB, /255)."""
    import os
    try:
        from PIL import Image, ImageOps
        import numpy as np
    except ImportError:
        log.warning("[BerniniStudio] PIL/numpy unavailable; cannot load slot image %s", filename)
        return None

    input_dir = None
    if folder_paths is not None:
        try:
            input_dir = folder_paths.get_input_directory()
        except Exception:
            input_dir = None
    if not input_dir:
        return None

    # Guard against path traversal -- filenames come from the editor widget
    safe = os.path.normpath(filename).replace("\\", "/")
    if safe.startswith("..") or os.path.isabs(safe):
        log.warning("[BerniniStudio] Rejected suspicious slot filename: %s", filename)
        return None

    path = os.path.join(input_dir, safe)
    if not os.path.isfile(path):
        log.warning("[BerniniStudio] Slot image not found: %s", path)
        return None
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None,]
    except Exception as e:
        log.warning("[BerniniStudio] Failed to load slot image %s: %s", filename, e)
        return None


# =========================================================================
# BerniniStudio node
# =========================================================================

class BerniniStudio:
    """All-in-one Bernini conditioning node.

    Handles text encoding (with task system prompt auto-prepend), VAE
    encoding of source video and reference images, context_latent
    construction, and outputs standard CONDITIONING + LATENT for
    SamplerCustom.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "T5 text encoder from CLIPLoader (wan type)."}),
                "vae": ("VAE", {"tooltip": "Wan 2.1/2.2 VAE from VAELoader."}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4,
                    "tooltip": "Number of output frames. Wan grid: 4n+1 (81, 121, 145...)."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "task_type": (TASK_TYPES, {"default": "v2v",
                    "tooltip": "Bernini task mode. Determines the system prompt and conditioning behavior. "
                               "v2v: general video edit. rv2v: edit guided by reference images. "
                               "r2v: generate video from reference subject(s). mv2v: change motion/pose. "
                               "ads2v: insert logo/ad into scene. vi2v: propagate first-frame edit to video. "
                               "i2i/i2v: image edit/animate. t2v/t2i: pure text generation. "
                               "See the hint box in the editor for full descriptions per mode."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Editing instruction or generation prompt. The task system prompt is auto-prepended. "
                               "For reference image tasks (rv2v, r2v, r2i, etc.), use image0, image1, image2 "
                               "to refer to connected reference images (not reference_image_0). "
                               "e.g. 'Replace the man with the person from image0'."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Negative prompt. Leave empty to use Bernini's default Chinese negative."}),
                "use_default_neg": ("BOOLEAN", {"default": True,
                    "tooltip": "When enabled and negative_prompt is empty, uses Bernini's standard negative prompt."}),
            },
            "optional": {
                "source_video": ("IMAGE", {
                    "tooltip": "Source video for editing (v2v, rv2v, mv2v). "
                               "Resized to width x height, trimmed to length."}),
                "image0": ("IMAGE", {
                    "tooltip": "Reference image slot 0 = 'image0' in your prompt. "
                               "IMPORTANT: fill slots in order (0, 1, 2...) with no gaps. "
                               "Skipped slots are compacted -- image6 with nothing before it becomes image0 to the model."}),
                "image1": ("IMAGE", {
                    "tooltip": "Reference image slot 1. Use 'image1' in your prompt."}),
                "image2": ("IMAGE", {
                    "tooltip": "Reference image slot 2. Use 'image2' in your prompt."}),
                "image3": ("IMAGE", {
                    "tooltip": "Reference image slot 3. Use 'image3' in your prompt."}),
                "image4": ("IMAGE", {"tooltip": "Reference image slot 4. Use 'image4' in prompt."}),
                "image5": ("IMAGE", {"tooltip": "Reference image slot 5. Use 'image5' in prompt."}),
                "image6": ("IMAGE", {"tooltip": "Reference image slot 6. Use 'image6' in prompt."}),
                "image7": ("IMAGE", {"tooltip": "Reference image slot 7. Use 'image7' in prompt."}),
                "reference_video": ("IMAGE", {
                    "tooltip": "Moving content to composite (ads2v / video insertion). "
                               "Kept at native aspect, trimmed to length."}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16,
                    "tooltip": "Max long-edge size for reference images/video. "
                               "Resized with preserved aspect, snapped to 16px."}),
                "auto_enhance": ("BOOLEAN", {"default": False,
                    "tooltip": "When enabled, automatically enhances the prompt via the LLM "
                               "server-side every time the node executes (on queue). "
                               "Uses the selected task template. Disable for manual-only enhancement."}),
                "unload_ollama": ("BOOLEAN", {"default": False,
                    "tooltip": "When enabled, tells Ollama to unload the model from VRAM immediately "
                               "after enhancement (keep_alive=0). Useful on low-VRAM systems. "
                               "Only affects Ollama, not OpenAI/vLLM endpoints."}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": ("STRING", {"default": ""}),
                "slot_images": ("STRING", {"default": "[]",
                    "tooltip": "Internal: JSON list of image filenames for the built-in reference slots. "
                               "Managed by the editor UI drag-and-drop. Wired imageN jacks override slots."}),
                "augment_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Experimental: std-dev of Gaussian noise added to the empty init latent "
                               "(0 = off, standard Bernini behavior). Low values (0.05-0.2) may add motion/detail "
                               "variation. Bernini was trained on a clean empty latent, so treat as an experiment."}),
                "augment_decay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-frame linear falloff of augment noise across the temporal axis. "
                               "0 = uniform noise on all frames; 1 = full strength on frame 0 fading to zero on the last frame."}),
                "augment_seed": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "Seed for the augment noise (independent of the sampler seed)."}),
                "api_format": (["Ollama", "OpenAI / vLLM"], {"default": "Ollama",
                    "tooltip": "API format for the prompt enhancer. "
                               "Ollama uses /api/chat + /api/tags. "
                               "OpenAI / vLLM uses /v1/chat/completions + /v1/models. "
                               "Vision models will receive connected reference images automatically."}),
                "send_ref_images": ("BOOLEAN", {"default": True,
                    "tooltip": "Send connected reference images / source frames to the LLM during Enhance "
                               "(vision models only). Disable for text-only models. If a text-only model "
                               "rejects the images anyway, the request is automatically retried text-only."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(
        self,
        clip, vae, width, height, length, batch_size,
        task_type, prompt, negative_prompt, use_default_neg,
        source_video=None,
        image0=None, image1=None, image2=None, image3=None,
        image4=None, image5=None, image6=None, image7=None,
        reference_video=None,
        ref_max_size=848,
        auto_enhance=False,
        unload_ollama=False,
        ollama_url="http://127.0.0.1:11434",
        ollama_model="",
        slot_images="[]",
        augment_strength=0.0,
        augment_decay=0.0,
        augment_seed=0,
        api_format="Ollama",
        send_ref_images=True,
        unique_id=None,
    ):
        # --- 0. Server-side auto-enhance (runs on every queue if enabled) ---
        working_prompt = prompt.strip()
        if auto_enhance and ollama_model and working_prompt:
            log.info("[BerniniStudio] Auto-enhance triggered (model=%s, task=%s)", ollama_model, task_type)
            enhanced = self._server_enhance(
                working_prompt, task_type, ollama_url, ollama_model, api_format, unload_ollama
            )
            if enhanced:
                working_prompt = enhanced
                log.info("[BerniniStudio] Auto-enhanced prompt (%d chars):\n%s",
                         len(enhanced), enhanced[:500] + ("..." if len(enhanced) > 500 else ""))
                # Push enhanced text back to the frontend so the user can see it
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("bernini_enhanced", {
                        "node": unique_id, "text": enhanced,
                    })
                except Exception:
                    pass
            else:
                log.warning("[BerniniStudio] Auto-enhance returned nothing; using original prompt")
        elif auto_enhance and not ollama_model:
            log.warning("[BerniniStudio] Auto-enhance enabled but no model selected; skipping")

        # --- 1. Text encoding with system prompt prepend ---
        sys_prompt = _get_system_prompt(task_type)
        full_prompt = sys_prompt + " " + working_prompt if working_prompt else sys_prompt

        # Determine negative prompt
        neg_text = negative_prompt.strip()
        if not neg_text and use_default_neg:
            neg_text = DEFAULT_NEG_PROMPT

        # Use the actual CLIPTextEncode node to encode text. This is the same
        # code path native ComfyUI uses, so it handles T5/Wan correctly regardless
        # of ComfyUI version or internal CLIP API changes.
        from nodes import CLIPTextEncode
        _encoder = CLIPTextEncode()
        positive = _encoder.encode(clip, full_prompt)[0]
        negative = _encoder.encode(clip, neg_text)[0]

        # --- 2. Empty latent for the output ---
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=mm.intermediate_device(),
        )

        # Experimental: augment the empty init latent with seeded Gaussian noise.
        # Bernini was trained on a clean empty latent, so default is 0 (off).
        # Sanitize against stale values from workflows saved before these widgets existed.
        try:
            augment_strength = max(0.0, min(1.0, float(augment_strength)))
        except (TypeError, ValueError):
            augment_strength = 0.0
        try:
            augment_decay = max(0.0, min(1.0, float(augment_decay)))
        except (TypeError, ValueError):
            augment_decay = 0.0
        if augment_strength > 0.0:
            gen = torch.Generator(device="cpu").manual_seed(int(augment_seed))
            noise = torch.randn(latent.shape, generator=gen, device="cpu").to(latent.device)
            t_frames = latent.shape[2]
            if augment_decay > 0.0 and t_frames > 1:
                ramp = torch.linspace(1.0, 1.0 - augment_decay, t_frames, device=latent.device)
                ramp = ramp.clamp(min=0.0).view(1, 1, t_frames, 1, 1)
                latent = noise * augment_strength * ramp
            else:
                latent = noise * augment_strength
            log.info("[BerniniStudio] Augmented empty latent: strength=%.3f decay=%.3f seed=%d",
                     augment_strength, augment_decay, augment_seed)

        # --- 3. Build context_latents (ordered by source_id) ---
        # Order: source_video (1), reference_video (2), reference_images (3, 4, ...)
        context = []

        if source_video is not None:
            vid = common_upscale(
                source_video[:length, :, :, :3].movedim(-1, 1),
                width, height, "area", "center",
            ).movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))
            log.info("[BerniniStudio] Encoded source video: %d frames at %dx%d",
                     vid.shape[0], width, height)

        if reference_video is not None:
            ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)
            context.append(vae.encode(ref_vid[:, :, :, :3]))
            log.info("[BerniniStudio] Encoded reference video: %d frames", ref_vid.shape[0])

        # Merge wired jacks with built-in editor slots (per slot: jack wins).
        # slot_images is a JSON list of filenames managed by the editor UI.
        slot_files = []
        try:
            parsed = json.loads(slot_images) if slot_images else []
            if isinstance(parsed, list):
                slot_files = parsed
        except Exception:
            pass

        wired = [image0, image1, image2, image3,
                 image4, image5, image6, image7]
        ref_images = []
        for i in range(8):
            if wired[i] is not None:
                ref_images.append(wired[i])
            elif i < len(slot_files) and slot_files[i]:
                tensor = _load_slot_image(slot_files[i])
                if tensor is not None:
                    ref_images.append(tensor)
                    log.info("[BerniniStudio] Loaded slot image%d from editor: %s", i, slot_files[i])

        for idx, ref_img in enumerate(ref_images):
            for frame_idx in range(ref_img.shape[0]):
                img = _resize_long_edge(ref_img[frame_idx:frame_idx + 1], ref_max_size)
                context.append(vae.encode(img[:, :, :, :3]))
            log.info("[BerniniStudio] Encoded image%d: %d frame(s)", idx, ref_img.shape[0])

        # --- 4. Attach context_latents to conditioning ---
        if context:
            if node_helpers is not None:
                positive = node_helpers.conditioning_set_values(
                    positive, {"context_latents": context}
                )
                negative = node_helpers.conditioning_set_values(
                    negative, {"context_latents": context}
                )
            else:
                for cond_list in [positive, negative]:
                    for item in cond_list:
                        item[1]["context_latents"] = context

            log.info("[BerniniStudio] Task '%s' (guidance: %s): %d context stream(s) attached",
                     task_type, GUIDANCE_MODE_BY_TASK.get(task_type, "?"), len(context))
        else:
            log.info("[BerniniStudio] Task '%s' (guidance: %s): no context (pure text generation)",
                     task_type, GUIDANCE_MODE_BY_TASK.get(task_type, "?"))

        return (positive, negative, {"samples": latent})

    @staticmethod
    def _server_enhance(user_prompt, task_type, url, model, api_format, unload_ollama=False):
        """Server-side prompt enhancement via Ollama/vLLM. Called during execute()
        when auto_enhance is True. Uses urllib (stdlib) to avoid async complications."""
        import urllib.request
        import urllib.error

        url = url.rstrip("/")
        template = _get_enhance_template(task_type)
        formatted = template.format(user_prompt=user_prompt, image_num=1)
        sys_prompt = _get_system_prompt(task_type)

        if api_format == "Ollama":
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": formatted},
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 8192},
            }
            if unload_ollama:
                payload["keep_alive"] = 0
            endpoint = f"{url}/api/chat"
        else:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": formatted},
                ],
                "stream": False,
            }
            _apply_openai_generation_options(payload, model, max_tokens=2048, temperature=0.7)
            endpoint = f"{url}/v1/chat/completions"

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint, data=data,
                headers=_llm_headers(api_format, include_json=True),
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            if api_format == "Ollama":
                text = (result.get("message", {}).get("content") or "").strip()
            else:
                text = (result.get("choices", [{}])[0]
                        .get("message", {}).get("content") or "").strip()

            # Extract from JSON wrapper if present
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            json_text = json_match.group(1).strip() if json_match else text.strip()
            if json_text.startswith("{"):
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, dict) and "rewritten_text" in parsed:
                        text = parsed["rewritten_text"]
                except Exception:
                    pass

            return text if text else None
        except Exception as e:
            log.warning("[BerniniStudio] Auto-enhance failed: %s: %s", type(e).__name__, e)
            return None


# =========================================================================
# Node registration
# =========================================================================

NODE_CLASS_MAPPINGS = {"BerniniStudio": BerniniStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"BerniniStudio": "Bernini Studio"}
WEB_DIRECTORY = "./js"


# =========================================================================
# Server-side Ollama prompt enhancement route
# =========================================================================

try:
    from server import PromptServer
    from aiohttp import web
    import aiohttp

    _routes = PromptServer.instance.routes

    @_routes.post("/bernini_studio/models")
    async def _bs_models(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        url = (data.get("ollama_url") or "http://127.0.0.1:11434").rstrip("/")
        api_format = data.get("api_format", "Ollama")

        try:
            async with aiohttp.ClientSession() as session:
                if api_format == "Ollama":
                    # Ollama: GET /api/tags
                    async with session.get(
                        f"{url}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as r:
                        if r.status != 200:
                            text = await r.text()
                            return web.json_response(
                                {"error": f"Ollama HTTP {r.status}: {text[:200]}"},
                                status=502,
                            )
                        tags = await r.json()
                    models = sorted(
                        (m.get("name") or m.get("model") or "")
                        for m in tags.get("models", []) if m
                    )
                else:
                    # OpenAI-compatible: GET /v1/models
                    async with session.get(
                        f"{url}/v1/models",
                        headers=_llm_headers(api_format, include_json=False),
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as r:
                        if r.status != 200:
                            text = await r.text()
                            return web.json_response(
                                {"error": f"OpenAI HTTP {r.status}: {text[:200]}"},
                                status=502,
                            )
                        resp = await r.json()
                    models = sorted(
                        m.get("id", "") for m in resp.get("data", []) if m
                    )
            return web.json_response({"models": [m for m in models if m]})
        except Exception as e:
            return web.json_response(
                {"error": f"Failed: {type(e).__name__}: {e}"}, status=502
            )

    @_routes.post("/bernini_studio/generate")
    async def _bs_generate(request):
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        url = (data.get("ollama_url") or "http://127.0.0.1:11434").rstrip("/")
        model = data.get("model")
        if not model:
            return web.json_response({"error": "No model selected"}, status=400)

        user_prompt = (data.get("prompt") or "").strip()
        if not user_prompt:
            return web.json_response({"error": "Empty prompt"}, status=400)

        task_type = data.get("task_type", "default")
        api_format = data.get("api_format", "Ollama")
        images_b64 = data.get("images", [])  # list of base64 strings (no data: prefix)
        image_num = data.get("image_num", len(images_b64) if images_b64 else 1)

        # Allow custom template override from the editor
        custom_template = data.get("custom_template", "")
        template = custom_template if custom_template.strip() else _get_enhance_template(task_type)
        formatted_prompt = template.format(user_prompt=user_prompt, image_num=image_num)
        sys_prompt = _get_system_prompt(task_type)

        # Build request based on API format
        if api_format == "Ollama":
            # Ollama native format
            user_msg = {"role": "user", "content": formatted_prompt}
            if images_b64:
                # Ollama expects raw base64 (no data: prefix) in the images field
                clean_images = []
                for img in images_b64:
                    if img.startswith("data:"):
                        img = img.split(",", 1)[-1]
                    clean_images.append(img)
                user_msg["images"] = clean_images

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    user_msg,
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 8192},
            }
            if data.get("unload_ollama"):
                payload["keep_alive"] = 0
            endpoint = f"{url}/api/chat"
        else:
            # OpenAI-compatible format (vLLM, LiteLLM, etc.)
            user_content = []
            if images_b64:
                for img_b64 in images_b64:
                    prefix = img_b64 if img_b64.startswith("data:") else f"data:image/jpeg;base64,{img_b64}"
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": prefix},
                    })
            user_content.append({"type": "text", "text": formatted_prompt})

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            }
            _apply_openai_generation_options(payload, model, max_tokens=2048, temperature=0.7)
            endpoint = f"{url}/v1/chat/completions"

        try:
            vision_fallback = False
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=_llm_headers(api_format, include_json=True),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as r:
                    first_status = r.status
                    if r.status != 200:
                        err_text = await r.text()
                    else:
                        resp = await r.json()

                if first_status != 200:
                    # Non-vision model rejecting images? Strip them and retry once.
                    lowered = err_text.lower()
                    retryable = images_b64 and any(
                        k in lowered for k in ("multimodal", "vision", "image")
                    )
                    if not retryable:
                        return web.json_response(
                            {"error": f"LLM HTTP {first_status}: {err_text[:300]}"},
                            status=502,
                        )
                    log.info("[BerniniStudio] Model rejected images; retrying text-only")
                    if api_format == "Ollama":
                        payload["messages"][1].pop("images", None)
                    else:
                        payload["messages"][1]["content"] = [
                            c for c in payload["messages"][1]["content"]
                            if c.get("type") == "text"
                        ]
                    images_b64 = []
                    vision_fallback = True
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=_llm_headers(api_format, include_json=True),
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as r2:
                        if r2.status != 200:
                            text = await r2.text()
                            return web.json_response(
                                {"error": f"LLM HTTP {r2.status} (text-only retry): {text[:300]}"},
                                status=502,
                            )
                        resp = await r2.json()

            # Extract text from response (handle both Ollama and OpenAI formats)
            if api_format == "Ollama":
                text = (resp.get("message", {}).get("content") or "").strip()
            else:
                text = (resp.get("choices", [{}])[0]
                        .get("message", {}).get("content") or "").strip()

            # Some templates ask for JSON with "rewritten_text" key.
            # LLMs often wrap JSON in ```json ... ``` markdown blocks.
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            json_text = json_match.group(1).strip() if json_match else text.strip()
            if json_text.startswith("{"):
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, dict) and "rewritten_text" in parsed:
                        text = parsed["rewritten_text"]
                except Exception:
                    pass

            has_vision = len(images_b64) > 0
            return web.json_response({
                "response": text,
                "vision_used": has_vision,
                "vision_fallback": vision_fallback,
                "image_count": len(images_b64),
            })
        except Exception as e:
            return web.json_response(
                {"error": f"Failed: {type(e).__name__}: {e}"}, status=502
            )

    @_routes.post("/bernini_studio/task_info")
    async def _bs_task_info(request):
        """Return task metadata for the JS editor."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        task = data.get("task_type", "v2v")
        return web.json_response({
            "system_prompt": _get_system_prompt(task),
            "guidance_mode": GUIDANCE_MODE_BY_TASK.get(task, "unknown"),
            "inputs": TASK_INPUTS.get(task, {}),
            "default_neg_prompt": DEFAULT_NEG_PROMPT,
        })

    @_routes.post("/bernini_studio/unload")
    async def _bs_unload(request):
        """Immediately unload the Ollama model from VRAM via keep_alive=0."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        url = (data.get("ollama_url") or "http://127.0.0.1:11434").rstrip("/")
        model = data.get("model")
        if not model:
            return web.json_response({"error": "No model selected"}, status=400)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}/api/generate",
                    json={"model": model, "keep_alive": 0},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status != 200:
                        text = await r.text()
                        return web.json_response(
                            {"error": f"Ollama HTTP {r.status}: {text[:200]}"},
                            status=502,
                        )
                    await r.read()
            log.info("[BerniniStudio] Unloaded model '%s' from Ollama VRAM", model)
            return web.json_response({"status": "unloaded", "model": model})
        except Exception as e:
            return web.json_response(
                {"error": f"Failed: {type(e).__name__}: {e}"}, status=502
            )

    @_routes.post("/bernini_studio/extract_frames")
    async def _bs_extract_frames(request):
        """Extract frames from a video in ComfyUI's input folder and return as base64 JPEG.
        Used to send source video frames to the LLM for vision-aware enhancement."""
        import base64
        import subprocess
        import tempfile

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        filename = data.get("filename", "")
        subfolder = data.get("subfolder", "")
        num_frames = min(data.get("num_frames", 3), 5)
        if not filename:
            return web.json_response({"error": "No filename"}, status=400)

        # Find the file in ComfyUI's input directory
        input_dir = None
        if folder_paths is not None:
            input_dir = folder_paths.get_input_directory()
        if not input_dir:
            import os
            input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")

        import os
        video_path = os.path.join(input_dir, subfolder, filename) if subfolder else os.path.join(input_dir, filename)
        if not os.path.isfile(video_path):
            return web.json_response({"error": f"File not found: {filename}"}, status=404)

        frames_b64 = []
        try:
            # Get total frame count via ffprobe
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames,nb_frames",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=15,
            )
            total_str = probe.stdout.strip().split(",")[0].strip()
            total_frames = int(total_str) if total_str.isdigit() else 100

            # Sample frames uniformly (skip frame 0, start from a few frames in)
            indices = []
            if total_frames <= num_frames:
                indices = list(range(total_frames))
            else:
                step = total_frames / (num_frames + 1)
                indices = [int(step * (i + 1)) for i in range(num_frames)]

            for idx in indices:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path,
                     "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                     "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "pipe:1"],
                    capture_output=True, timeout=15,
                )
                if result.returncode == 0 and result.stdout:
                    frames_b64.append(base64.b64encode(result.stdout).decode("ascii"))

            log.info("[BerniniStudio] Extracted %d frame(s) from %s", len(frames_b64), filename)
            return web.json_response({"frames": frames_b64, "total_frames": total_frames})
        except FileNotFoundError:
            return web.json_response({"error": "ffmpeg/ffprobe not found"}, status=500)
        except Exception as e:
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)

    @_routes.post("/bernini_studio/get_template")
    async def _bs_get_template(request):
        """Return the full enhancement template for a given task type."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        task = data.get("task_type", "default")
        return web.json_response({
            "template": _get_enhance_template(task),
            "task_type": task,
        })

except ImportError:
    log.warning("[BerniniStudio] PromptServer not available; server routes not registered.")
