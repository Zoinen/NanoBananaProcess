import argparse
import asyncio
import sys
import base64
import io
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from google import genai
from google.genai import types
from openai import AsyncOpenAI

console = Console()
google_client = None
openai_client = None
xai_client = None

def _init_clients():
    global google_client, openai_client, xai_client
    google_client = genai.Client()
    openai_client = AsyncOpenAI()
    xai_client = AsyncOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=os.environ.get("XAI_API_KEY", "not-set"),
    )
LOG_FILE = Path(__file__).parent / "requests.log"

def _load_estimates() -> dict:
    """Return mean successful duration per (model_key, image_size) from the log."""
    if not LOG_FILE.exists():
        return {}
    buckets: dict[tuple, list] = {}
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") != "success":
                continue
            key = (entry["model_key"], entry["image_size"])
            buckets.setdefault(key, []).append(entry["duration_seconds"])
    return {k: sum(v) / len(v) for k, v in buckets.items()}

# Configuration variables
BASE_PROMPT = "redraw this as if it was a professional photo taken on a modern high quality DSLR camera, but keep all the details as close to original as possible"
ALPHA_PROMPT = "draw alpha channel for this image. Background should be transparent and the foreground should be not"
DEFAULT_VARIANTS_PER_IMAGE = 3
MAX_CONCURRENT_REQUESTS = 15 # Adjust based on your API tier's rate limits
MAX_OPENAI_IMAGE_BYTES  = 50 * 1024 * 1024  # OpenAI hard limit for input images

MODEL_MAPPING = {
    "pro":       "gemini-3-pro-image-preview",      # Nano Banana Pro
    "flash":     "gemini-3.1-flash-image-preview",  # Nano Banana 2
    "lite":      "gemini-3.1-flash-lite-image",     # Nano Banana 2 Lite (fast, cheap)
    "gptimage1": "gpt-image-1.5",                   # GPT Image 1.5 (supports transparency; replaces gpt-image-1)
    "gptimage2": "gpt-image-2",                     # GPT Image 2
    "grok":      "grok-imagine-image-quality",      # xAI Grok Imagine (supports transparency)
}

SUFFIX_MAPPING = {
    "pro":       "_nb_pro_",     # Nano Banana Pro
    "flash":     "_nb_two_",     # Nano Banana 2
    "lite":      "_nb_lite_",    # Nano Banana 2 Lite
    "gptimage1": "_gptimg_one_", # GPT Image 1
    "gptimage2": "_gptimg_two_", # GPT Image 2
    "grok":      "_grok_",       # Grok Imagine
}

PROVIDER_MAPPING = {
    "pro":       "google",
    "flash":     "google",
    "lite":      "google",
    "gptimage1": "openai",
    "gptimage2": "openai",
    "grok":      "xai",
}

def _mime_to_ext(mime_type: str) -> str:
    return {"image/jpeg": ".jpg", "image/jpg": ".jpg",
            "image/png": ".png", "image/webp": ".webp"}.get(mime_type.lower(), ".png")

def _pil_format_to_ext(fmt: str | None) -> str:
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt or "", ".png")

def _bytes_to_ext(data: bytes) -> str:
    if data[:3] == b'\xff\xd8\xff':          return ".jpg"
    if data[:8] == b'\x89PNG\r\n\x1a\n':     return ".png"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP': return ".webp"
    return ".png"

async def _compose_alpha_async(source_path: Path, mask_path: Path) -> Path:
    """Compose source RGB with generated mask as alpha → RGBA PNG.

    The mask is converted to a grayscale alpha channel (white=opaque,
    black=transparent). If the mask already has an alpha channel, that
    channel is used directly. Resizes the alpha to match source if needed.
    Saves the result as <mask_stem>_composed.png beside the mask file.
    """
    def _compose():
        src = Image.open(source_path).convert("RGBA")
        mask_img = Image.open(mask_path)

        # Prefer the mask's own alpha channel; fall back to luminance
        if mask_img.mode == "RGBA":
            alpha = mask_img.split()[3]
        else:
            alpha = mask_img.convert("L")

        # Align to source dimensions if the model returned a different size
        if alpha.size != src.size:
            alpha = alpha.resize(src.size, Image.LANCZOS)

        r, g, b, _ = src.split()
        composed = Image.merge("RGBA", (r, g, b, alpha))
        out_path = mask_path.with_stem(mask_path.stem + "_composed").with_suffix(".png")
        composed.save(out_path, format="PNG")
        return out_path

    return await asyncio.to_thread(_compose)

async def save_image_async(image_obj_or_bytes, output_path: Path):
    """Offloads the file saving to a separate thread so it doesn't block the async event loop."""
    def _save():
        if hasattr(image_obj_or_bytes, 'save'):
            image_obj_or_bytes.save(output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(image_obj_or_bytes)

    await asyncio.to_thread(_save)

async def _call_google(imgs: list[Image.Image], base_output: Path, model_id: str, prompt: str, image_size: str, aspect_ratio: str = None) -> Path:
    """Calls the Gemini API and saves the result. Returns the actual saved path."""
    img_cfg = types.ImageConfig(image_size=image_size, aspect_ratio=aspect_ratio) if aspect_ratio \
              else types.ImageConfig(image_size=image_size)
    contents = [*imgs, prompt] if imgs else [prompt]
    response = await google_client.aio.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=img_cfg,
        )
    )

    for part in response.parts:
        if part.inline_data:
            ext = _mime_to_ext(part.inline_data.mime_type)
            output_path = base_output.parent / (base_output.name + ext)
            await save_image_async(part.inline_data.data, output_path)
            return output_path
        if hasattr(part, 'as_image') and callable(part.as_image):
            pil_img = part.as_image()
            ext = _pil_format_to_ext(pil_img.format)
            output_path = base_output.parent / (base_output.name + ext)
            await save_image_async(pil_img, output_path)
            return output_path

    raise RuntimeError("No image data in response")

ASPECT_RATIO_ALIASES = {
    "iphone": "9:19.5",
}

_IPHONE_PAGES_RE  = re.compile(r"^iphone-pages-(\d+)$", re.IGNORECASE)
_CUSTOM_SIZE_RE   = re.compile(r"^(\d+)x(\d+)$", re.IGNORECASE)

def _iphone_pages_ratio(n: int) -> float:
    """Aspect ratio for N iPhone screens side-by-side with 10%-of-page-width gaps."""
    page_w, page_h = 9.0, 19.5
    gap = page_w * 0.10
    return (n * page_w + (n - 1) * gap) / page_h

def _parse_aspect_ratio(ar: str) -> float:
    """Convert a W:H string, named alias, or 'iphone-pages-N' to a w/h float."""
    m = _IPHONE_PAGES_RE.match(ar)
    if m:
        return _iphone_pages_ratio(int(m.group(1)))
    ar = ASPECT_RATIO_ALIASES.get(ar.lower(), ar)
    w, h = ar.split(":")
    return float(w) / float(h)

def _resolution_arg(value: str) -> str:
    """Argparse type for --resolution: keywords or exact WxH (e.g. 1920x1080)."""
    if value.lower() in ("1k", "4k", "source"):
        return value.lower()
    m = _CUSTOM_SIZE_RE.match(value)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if w < 1 or h < 1:
            raise argparse.ArgumentTypeError("Resolution dimensions must be positive integers")
        return f"{w}x{h}"
    raise argparse.ArgumentTypeError(
        f"Invalid resolution '{value}'. Use '1k', '4k', 'source', or WxH (e.g. 1920x1080)."
    )

def _aspect_ratio_arg(value: str) -> str:
    """Argparse type validator — accepts named aliases, 'iphone-pages-N', or 'W:H' strings."""
    if _IPHONE_PAGES_RE.match(value):
        n = int(_IPHONE_PAGES_RE.match(value).group(1))
        if n < 1:
            raise argparse.ArgumentTypeError("iphone-pages-N requires N >= 1")
        return value
    if value.lower() in ASPECT_RATIO_ALIASES:
        return value
    parts = value.split(":")
    if len(parts) != 2:
        aliases = ", ".join(ASPECT_RATIO_ALIASES) + ", iphone-pages-N"
        raise argparse.ArgumentTypeError(
            f"Aspect ratio must be W:H (e.g. 16:9) or a named alias ({aliases}), got '{value}'"
        )
    try:
        w, h = float(parts[0]), float(parts[1])
        if w <= 0 or h <= 0:
            raise argparse.ArgumentTypeError("Aspect ratio values must be positive")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid aspect ratio '{value}' — use W:H format")
    return value

def _compute_openai_size(src_w: int, src_h: int, mode: str = "max", override_ratio: float = None) -> str:
    """Return a WxH string satisfying gpt-image-2 constraints at the source aspect ratio.

    mode="max"    (default): largest valid resolution (4K mode).
    mode="min":              smallest valid resolution (1K mode).
    mode="source":           stay as close to the source resolution as possible.

    Constraints: max edge ≤ 3840, both dims multiples of 16, ratio ≤ 3:1,
    total pixels in [655 360, 8 294 400].
    """
    MIN_PIXELS = 655_360
    MAX_PIXELS = 8_294_400
    MAX_EDGE   = 3840
    MULTIPLE   = 16
    MAX_RATIO  = 3.0

    # Clamp aspect ratio to [1/3, 3]; override_ratio takes precedence over source dimensions
    raw_ratio = override_ratio if override_ratio is not None else src_w / src_h
    ratio = max(1 / MAX_RATIO, min(MAX_RATIO, raw_ratio))

    if mode == "max":
        # Fill the longest edge to MAX_EDGE, derive the other from ratio
        if ratio >= 1.0:  # landscape or square
            w, h = float(MAX_EDGE), MAX_EDGE / ratio
        else:             # portrait
            h, w = float(MAX_EDGE), MAX_EDGE * ratio

        # Scale down if total pixels would exceed the cap
        total = w * h
        if total > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / total)
            w *= scale
            h *= scale

        # Snap to the nearest lower multiple of 16
        w = int(w) // MULTIPLE * MULTIPLE
        h = int(h) // MULTIPLE * MULTIPLE

    elif mode == "min":
        # Find the shortest edge whose square (times ratio) just meets MIN_PIXELS,
        # then round UP to the nearest multiple of 16.
        if ratio >= 1.0:  # landscape: h is short edge
            h = math.ceil(math.sqrt(MIN_PIXELS / ratio) / MULTIPLE) * MULTIPLE
            w = math.ceil(h * ratio / MULTIPLE) * MULTIPLE
        else:             # portrait: w is short edge
            w = math.ceil(math.sqrt(MIN_PIXELS * ratio) / MULTIPLE) * MULTIPLE
            h = math.ceil(w / ratio / MULTIPLE) * MULTIPLE

        # Guard: ceil arithmetic guarantees this, but verify anyway
        while w * h < MIN_PIXELS:
            if ratio >= 1.0:
                h += MULTIPLE
                w = math.ceil(h * ratio / MULTIPLE) * MULTIPLE
            else:
                w += MULTIPLE
                h = math.ceil(w / ratio / MULTIPLE) * MULTIPLE

    else:  # "source" — preserve source resolution, clamped to API limits
        w, h = float(src_w), float(src_h)

        # Apply override_ratio by rescaling while preserving total pixel count
        if override_ratio is not None:
            total = w * h
            if ratio >= 1.0:
                h = math.sqrt(total / ratio)
                w = h * ratio
            else:
                w = math.sqrt(total * ratio)
                h = w / ratio

        # Clamp max edge, then enforce pixel floor/ceiling
        if max(w, h) > MAX_EDGE:
            scale = MAX_EDGE / max(w, h)
            w *= scale; h *= scale
        if w * h < MIN_PIXELS:
            scale = math.sqrt(MIN_PIXELS / (w * h))
            w *= scale; h *= scale
        if w * h > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (w * h))
            w *= scale; h *= scale

        # Snap to multiples of 16 (round down)
        w = int(w) // MULTIPLE * MULTIPLE
        h = int(h) // MULTIPLE * MULTIPLE

        # Post-snap correction if we fell below minimum
        while w * h < MIN_PIXELS:
            if ratio >= 1.0:
                h += MULTIPLE
                w = math.ceil(h * ratio / MULTIPLE) * MULTIPLE
            else:
                w += MULTIPLE
                h = math.ceil(w / ratio / MULTIPLE) * MULTIPLE

    return f"{w}x{h}"


def _source_to_gemini_size(w: int, h: int) -> str:
    """Pick the nearest Gemini image-size preset (1K/2K/4K) for a given source resolution.

    Thresholds are geometric means between adjacent preset pixel counts:
      1K ≈ 1 MP, 2K ≈ 4 MP, 4K ≈ 16 MP
      1K↔2K midpoint: sqrt(1×4)  = 2 MP
      2K↔4K midpoint: sqrt(4×16) = 8 MP
    """
    px = w * h
    if px <= 2_000_000:
        return "1K"
    if px <= 8_000_000:
        return "2K"
    return "4K"

# gpt-image-1 supports only these fixed presets (ratio → WxH string)
_GPT_IMAGE_1_PRESETS = [
    (1.0,        "1024x1024"),  # square
    (1536/1024,  "1536x1024"),  # landscape
    (1024/1536,  "1024x1536"),  # portrait
]

def _pick_gptimage1_size(ratio: float) -> str:
    """Return the gpt-image-1 preset whose aspect ratio is closest to *ratio*."""
    return min(_GPT_IMAGE_1_PRESETS, key=lambda p: abs(math.log(ratio / p[0])))[1]

def _pick_nearest_aspect_ratio(ratio: float, choices: list[str]) -> str:
    """Return the choice string whose w/h ratio is closest (log-scale) to *ratio*."""
    def _to_float(ar: str) -> float:
        w, h = ar.split(":")
        return float(w) / float(h)
    return min(choices, key=lambda ar: abs(math.log(ratio / _to_float(ar))))

# grok-imagine-image-quality supports these aspect ratios natively
_GROK_ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
    "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20",
]

def _pick_grok_aspect_ratio(ratio: float) -> str:
    """Return the nearest grok-supported aspect ratio string for a given w/h ratio."""
    return _pick_nearest_aspect_ratio(ratio, _GROK_ASPECT_RATIOS)

# Gemini image models support these aspect ratios natively
_GEMINI_ASPECT_RATIOS = [
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
    "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
]

def _pick_gemini_aspect_ratio(ratio: float) -> str:
    """Return the nearest Gemini-supported aspect ratio string for a given w/h ratio."""
    return _pick_nearest_aspect_ratio(ratio, _GEMINI_ASPECT_RATIOS)

async def _call_openai(imgs: list[Image.Image], base_output: Path, model_id: str, prompt: str, size: str, quality: str | None, transparent: bool = False, client: AsyncOpenAI | None = None, extra_body: dict | None = None) -> Path:
    """Calls an OpenAI-compatible image API and saves the result. Returns the actual saved path."""
    _client = client or openai_client
    extra = {}
    if transparent:
        extra["background"] = "transparent"
        extra["output_format"] = "png"   # JPEG has no alpha channel

    if imgs:
        # Edit mode: send one or more source images alongside the prompt
        def _to_bufs():
            bufs = []
            for img in imgs:
                work = img
                buf = io.BytesIO()
                work.save(buf, format="PNG")
                # Scale down iteratively if the PNG exceeds the 50 MB API limit
                while buf.tell() > MAX_OPENAI_IMAGE_BYTES:
                    scale = math.sqrt(MAX_OPENAI_IMAGE_BYTES / buf.tell()) * 0.92
                    work = work.resize(
                        (max(1, int(work.width * scale)), max(1, int(work.height * scale))),
                        Image.LANCZOS,
                    )
                    buf = io.BytesIO()
                    work.save(buf, format="PNG")
                buf.seek(0)
                buf.name = "image.png"
                bufs.append(buf)
            return bufs

        bufs = await asyncio.to_thread(_to_bufs)
        call_kwargs = dict(model=model_id, image=bufs[0] if len(bufs) == 1 else bufs,
                           prompt=prompt, size=size, n=1, **extra)
        if quality is not None:
            call_kwargs["quality"] = quality
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        response = await _client.images.edit(**call_kwargs)
    else:
        # Generate mode: pure prompt, no source image
        call_kwargs = dict(model=model_id, prompt=prompt, size=size, n=1, **extra)
        if quality is not None:
            call_kwargs["quality"] = quality
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        response = await _client.images.generate(**call_kwargs)

    img_bytes = base64.b64decode(response.data[0].b64_json)
    ext = _bytes_to_ext(img_bytes)
    output_path = base_output.parent / (base_output.name + ext)
    await save_image_async(img_bytes, output_path)
    return output_path

async def _call_xai(imgs: list[Image.Image], base_output: Path, model_id: str, prompt: str, size: str, transparent: bool = False, aspect_ratio: str = "auto") -> Path:
    """Calls xAI image API via raw JSON POST (multipart not supported by xAI). Returns the saved path."""
    import httpx

    api_key = os.environ.get("XAI_API_KEY", "not-set")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if imgs:
        def _to_b64s():
            result = []
            for img in imgs:
                work = img
                buf = io.BytesIO()
                work.save(buf, format="PNG")
                while buf.tell() > MAX_OPENAI_IMAGE_BYTES:
                    scale = math.sqrt(MAX_OPENAI_IMAGE_BYTES / buf.tell()) * 0.92
                    work = work.resize(
                        (max(1, int(work.width * scale)), max(1, int(work.height * scale))),
                        Image.LANCZOS,
                    )
                    buf = io.BytesIO()
                    work.save(buf, format="PNG")
                result.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
            return result

        b64s = await asyncio.to_thread(_to_b64s)
        img_objs = [{"url": b, "type": "image_url"} for b in b64s]
        body = {"model": model_id, "prompt": prompt, "resolution": size, "aspect_ratio": aspect_ratio, "n": 1,
                "image": img_objs[0] if len(img_objs) == 1 else img_objs}
        url = "https://api.x.ai/v1/images/edits"
    else:
        body = {"model": model_id, "prompt": prompt, "resolution": size, "aspect_ratio": aspect_ratio, "n": 1}
        url = "https://api.x.ai/v1/images/generations"

    async with httpx.AsyncClient(timeout=300.0) as http:
        resp = await http.post(url, json=body, headers=headers)
        if not resp.is_success:
            raise RuntimeError(f"xAI {resp.status_code}: {resp.text}")
        data = resp.json()

    item = data["data"][0]
    if item.get("b64_json"):
        img_bytes = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        async with httpx.AsyncClient(timeout=60.0) as http:
            img_bytes = (await http.get(item["url"])).content
    else:
        raise RuntimeError(f"No image data in xAI response: {data}")

    ext = _bytes_to_ext(img_bytes)
    output_path = base_output.parent / (base_output.name + ext)
    await save_image_async(img_bytes, output_path)
    return output_path

async def _log_request(lock: asyncio.Lock, entry: dict):
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    async with lock:
        def _write():
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        await asyncio.to_thread(_write)

async def generate_variant(image_paths: list[Path], variant_idx: int, semaphore: asyncio.Semaphore, log_lock: asyncio.Lock, model_key: str, model_id: str, suffix: str, extra_prompt: str, override_prompt: str, image_size: str, aspect_ratio: str, progress: Progress, estimates: dict, transparent: bool = False, alpha_mode: bool = False):
    """Asynchronously calls the API to generate/redraw an image and saves it."""
    # Build base path without extension — the call functions detect the real format
    if not image_paths:
        base_output = Path.cwd() / f"generated{suffix}{variant_idx}"
    elif len(image_paths) == 1:
        base_output = image_paths[0].with_name(f"{image_paths[0].stem}{suffix}{variant_idx}")
    else:
        base_output = image_paths[0].with_name(f"{'+'.join(p.stem for p in image_paths)}{suffix}{variant_idx}")
    output_path = base_output  # will be replaced with the real path after the call

    final_prompt = BASE_PROMPT
    if extra_prompt:
        final_prompt = f"{BASE_PROMPT}. {extra_prompt.strip()}"

    # --prompt fully replaces the base prompt when provided
    if override_prompt:
        final_prompt = override_prompt.strip()

    # --alpha always wins — it replaces every other prompt setting, but still
    # allows --extra-prompt to append additional instructions
    if alpha_mode:
        final_prompt = ALPHA_PROMPT
        if extra_prompt:
            final_prompt = f"{ALPHA_PROMPT}. {extra_prompt.strip()}"

    async with semaphore:
        estimated = estimates.get((model_key, image_size))
        est_label = f"/ ~{estimated:.0f}s" if estimated else "/ ?"
        if not image_paths:
            src_label = "[italic]prompt only[/italic]"
        elif len(image_paths) == 1:
            src_label = image_paths[0].name
        else:
            src_label = f"{image_paths[0].name} +{len(image_paths)-1} more"
        task_id = progress.add_task(
            f"{src_label}  [dim]{model_key}/{image_size}[/dim]",
            total=100,
            est=est_label,
        )

        t_start = datetime.now(timezone.utc)
        status, error_msg, resolved_size, quality = "error", None, image_size, None
        done_event = asyncio.Event()

        async def _tick():
            while not done_event.is_set():
                await asyncio.sleep(0.25)
                elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
                pct = min(elapsed / estimated * 100, 99) if estimated else 0
                progress.update(task_id, completed=pct)

        ticker = asyncio.create_task(_tick())

        try:
            imgs = []
            if image_paths:
                def _load_imgs():
                    loaded = []
                    for p in image_paths:
                        src = Image.open(p)
                        src.load()
                        loaded.append(src)
                    return loaded
                imgs = await asyncio.to_thread(_load_imgs)

            if PROVIDER_MAPPING[model_key] != "google":
                override_ratio = _parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
                src_dims = imgs[0].size if imgs else (1, 1)
                raw_ratio = override_ratio if override_ratio is not None else (src_dims[0] / src_dims[1])

                if model_key == "gptimage1":
                    # gpt-image-1: fixed presets only; transparency supported
                    if image_size == "1K":
                        resolved_size, quality = "1024x1024", "medium"
                    elif _CUSTOM_SIZE_RE.match(image_size):
                        cw, ch = map(int, image_size.split("x"))
                        resolved_size, quality = _pick_gptimage1_size(override_ratio or (cw / ch)), "high"
                    else:
                        resolved_size, quality = _pick_gptimage1_size(raw_ratio), "high"
                    output_path = await _call_openai(imgs, base_output, model_id, final_prompt, resolved_size, quality, transparent=transparent)
                elif model_key == "grok":
                    # grok: "1k"/"2k" size strings; aspect ratio as separate param; transparency supported
                    if image_size == "1K":
                        resolved_size = "1k"
                    elif image_size == "source" and imgs:
                        resolved_size = "1k" if src_dims[0] * src_dims[1] <= 2_000_000 else "2k"
                    else:
                        resolved_size = "2k"
                    quality = None
                    target_ratio = override_ratio if override_ratio is not None else (raw_ratio if imgs else None)
                    grok_ar = _pick_grok_aspect_ratio(target_ratio) if target_ratio is not None else "auto"
                    output_path = await _call_xai(imgs, base_output, model_id, final_prompt, resolved_size, transparent, grok_ar)
                else:
                    # gpt-image-2: arbitrary WxH; transparency NOT supported
                    if image_size == "1K":
                        resolved_size, quality = _compute_openai_size(*src_dims, mode="min", override_ratio=override_ratio), "medium"
                    elif image_size == "source" and imgs:
                        resolved_size, quality = _compute_openai_size(*src_dims, mode="source", override_ratio=override_ratio), "high"
                    elif _CUSTOM_SIZE_RE.match(image_size):
                        cw, ch = map(int, image_size.split("x"))
                        resolved_size, quality = _compute_openai_size(cw, ch, mode="source", override_ratio=override_ratio), "high"
                    elif not imgs and override_ratio is None:
                        resolved_size, quality = "auto", "high"
                    else:
                        resolved_size, quality = _compute_openai_size(*src_dims, override_ratio=override_ratio), "high"
                    output_path = await _call_openai(imgs, base_output, model_id, final_prompt, resolved_size, quality, transparent=False)
            else:
                if image_size == "source":
                    resolved_size = _source_to_gemini_size(*imgs[0].size) if imgs else "4K"
                elif _CUSTOM_SIZE_RE.match(image_size):
                    cw, ch = map(int, image_size.split("x"))
                    resolved_size = _source_to_gemini_size(cw, ch)
                    if aspect_ratio is None:
                        aspect_ratio = f"{cw}:{ch}"
                else:
                    resolved_size = image_size
                gemini_ar = _pick_gemini_aspect_ratio(_parse_aspect_ratio(aspect_ratio)) if aspect_ratio else None
                output_path = await _call_google(imgs, base_output, model_id, final_prompt, resolved_size, gemini_ar)

            status = "success"
            progress.print(f"✅ Saved {output_path.name}")

            if alpha_mode and image_paths:
                composed_path = await _compose_alpha_async(image_paths[0], output_path)
                await asyncio.to_thread(output_path.unlink)
                progress.print(f"🔀 Composed {composed_path.name}")

        except Exception as e:
            error_msg = str(e)
            progress.print(f"❌ Error: {output_path.name}: {e}")

        finally:
            done_event.set()
            await ticker
            progress.update(task_id, completed=100)
            progress.remove_task(task_id)
            t_end = datetime.now(timezone.utc)
            await _log_request(log_lock, {
                "timestamp_start":  t_start.isoformat(),
                "timestamp_end":    t_end.isoformat(),
                "duration_seconds": round((t_end - t_start).total_seconds(), 3),
                "source":           image_paths[0].name if len(image_paths) == 1 else [p.name for p in image_paths] if image_paths else None,
                "output":           output_path.name,
                "model_key":        model_key,
                "model_id":         model_id,
                "provider":         PROVIDER_MAPPING[model_key],
                "image_size":       image_size,
                "aspect_ratio":     aspect_ratio,
                "resolved_size":    resolved_size,
                "quality":          quality,
                "transparent":      transparent,
                "prompt":           final_prompt,
                "status":           status,
                "error":            error_msg,
            })

def _next_variant_index(image_paths: list[Path], suffix: str) -> int:
    """Return the next available variant index for this image+suffix pair."""
    if not image_paths:
        directory, stem = Path.cwd(), "generated"
    elif len(image_paths) == 1:
        directory, stem = image_paths[0].parent, image_paths[0].stem
    else:
        directory, stem = image_paths[0].parent, "+".join(p.stem for p in image_paths)
    pattern = re.compile(
        rf"^{re.escape(stem)}{re.escape(suffix)}(\d+)\.(png|jpg|jpeg|webp)$",
        re.IGNORECASE,
    )
    max_idx = 0
    for f in directory.iterdir():
        m = pattern.match(f.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

async def async_main(image_files: list[Path], model_keys: list[str], extra_prompt: str, override_prompt: str, variants: int, image_size: str, aspect_ratio: str, transparent: bool = False, alpha_mode: bool = False):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    log_lock  = asyncio.Lock()
    estimates = _load_estimates()

    model_desc = " + ".join(MODEL_MAPPING[k] for k in model_keys)

    # All input images go into a single request per variant
    if not image_files:
        sources = [[]]
        console.print(f"Prompt-only mode. Generating {variants} variant(s) per model — {model_desc}")
    else:
        sources = [image_files]
        n = len(image_files)
        label = "1 image" if n == 1 else f"{n} images combined"
        console.print(f"{label}. Generating {variants} variant(s) — {model_desc}")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[est]}[/dim]"),
        console=console,
        transient=False,
    ) as progress:
        tasks = []
        for img_paths in sources:
            for model_key in model_keys:
                model_id = MODEL_MAPPING[model_key]
                suffix = SUFFIX_MAPPING[model_key]
                start_idx = _next_variant_index(img_paths, suffix)
                for i in range(start_idx, start_idx + variants):
                    tasks.append(generate_variant(img_paths, i, semaphore, log_lock, model_key, model_id, suffix, extra_prompt, override_prompt, image_size, aspect_ratio, progress, estimates, transparent=transparent, alpha_mode=alpha_mode))

        console.print(f"Firing off {len(tasks)} requests...")
        await asyncio.gather(*tasks)

    console.print("🎉 All done!")

def main():
    parser = argparse.ArgumentParser(description="Redraw images asynchronously using Gemini Image Models.")
    parser.add_argument("paths", nargs="*", default=[],
                        help="One or more image files or folders. Mix freely. Omit to generate from prompt only.")
    parser.add_argument("--model", choices=["pro", "flash", "lite", "gptimage1", "gptimage2", "grok", "both", "google", "all"], default="pro",
                        help="Choose 'pro' (Nano Banana Pro), 'flash' (Nano Banana 2), 'lite' (Nano Banana 2 Lite, fast+cheap), 'gptimage1' (GPT Image 1, supports transparency), 'gptimage2' (GPT Image 2), 'grok' (Grok Imagine, JPEG output only, no transparency), 'both' (pro + gptimage2), 'google' (pro + flash), or 'all' (pro + flash + gptimage2) in parallel.")
    parser.add_argument("--extra-prompt", type=str, default="",
                        help="Additional text to append to the end of the base prompt.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Fully override the base prompt. If --prompt-file is also given, this text comes first.")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to a text file appended after --prompt (or used alone as full override).")
    parser.add_argument("--resolution", type=_resolution_arg, default="4k",
                        help="Output resolution: '1k', '4k' (default), 'source' to match input image dimensions, or exact WxH (e.g. 1920x1080).")
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS_PER_IMAGE,
                        help=f"Number of variants to generate per image (default: {DEFAULT_VARIANTS_PER_IMAGE}).")
    parser.add_argument("--aspect-ratio", type=_aspect_ratio_arg, default=None,
                        help="Output aspect ratio as W:H (e.g. 16:9, 4:3, 1:1). Defaults to source image ratio.")
    parser.add_argument("--transparent", action="store_true",
                        help="Generate image with a transparent background (gpt-image-1 only; ignored for Gemini models and gpt-image-2).")
    parser.add_argument("--alpha", action="store_true",
                        help="Alpha-mask mode: generate a grayscale alpha mask for the source image, then compose it with the source RGB into a final RGBA PNG.")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    _init_clients()

    model_keys    = {"all": ["pro", "flash", "gptimage2"], "both": ["pro", "gptimage2"], "google": ["pro", "flash"]}.get(args.model, [args.model])
    extra_prompt  = args.extra_prompt

    file_prompt = ""
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_file():
            console.print(f"[red]Prompt file not found:[/red] {args.prompt_file}")
            return
        file_prompt = prompt_path.read_text(encoding="utf-8").strip()

    if args.prompt and file_prompt:
        override_prompt = f"{args.prompt.strip()}\n{file_prompt}"
    elif file_prompt:
        override_prompt = file_prompt
    else:
        override_prompt = args.prompt
    image_size    = {"1k": "1K", "4k": "4K", "source": "source"}.get(args.resolution, args.resolution)
    aspect_ratio  = args.aspect_ratio
    transparent   = args.transparent

    if transparent:
        unsupported = [k for k in model_keys if k not in ("gptimage1",)]
        if unsupported:
            console.print(f"[yellow]Note:[/yellow] --transparent is only supported by gpt-image-1; will be ignored for: {', '.join(unsupported)}")

    alpha_mode = args.alpha
    if alpha_mode and not args.paths:
        console.print("[yellow]Note:[/yellow] --alpha has no effect in prompt-only mode (no source image to composite).")

    all_suffixes  = set(SUFFIX_MAPPING.values())

    image_files = []
    valid_exts  = {".jpg", ".jpeg", ".png", ".webp"}

    if args.paths:
        for raw in args.paths:
            input_path = Path(raw)
            if input_path.is_file():
                image_files.append(input_path)  # explicit file — always included
            elif input_path.is_dir():
                # Directory expansion: skip our own output files
                image_files.extend(
                    p for p in input_path.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in valid_exts
                    and not any(s in p.stem for s in all_suffixes)
                )
            else:
                console.print(f"[red]Invalid path:[/red] {raw}")
                return
        if not image_files:
            console.print("No valid, unprocessed images found.")
            return
    else:
        # Prompt-only mode: a meaningful prompt is required
        if not override_prompt and not extra_prompt:
            console.print("[yellow]Warning:[/yellow] no source image and no --prompt given — the default redraw prompt will be used as-is.")


    asyncio.run(async_main(image_files, model_keys, extra_prompt, override_prompt, args.variants, image_size, aspect_ratio, transparent=transparent, alpha_mode=alpha_mode))

if __name__ == "__main__":
    main()
