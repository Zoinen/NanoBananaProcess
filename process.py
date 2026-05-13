import argparse
import asyncio
import base64
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from google import genai
from google.genai import types
from openai import AsyncOpenAI

# Initialize clients
google_client = genai.Client()
openai_client = AsyncOpenAI()
console  = Console()
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
DEFAULT_VARIANTS_PER_IMAGE = 3
MAX_CONCURRENT_REQUESTS = 15 # Adjust based on your API tier's rate limits

MODEL_MAPPING = {
    "pro":       "gemini-3-pro-image-preview",      # Nano Banana Pro
    "flash":     "gemini-3.1-flash-image-preview",  # Nano Banana 2
    "gptimage2": "gpt-image-2",                     # GPT Image 2
}

SUFFIX_MAPPING = {
    "pro":       "_nb_pro_",  # Nano Banana Pro
    "flash":     "_nb_two_",  # Nano Banana 2
    "gptimage2": "_gptimg_two_",  # GPT Image 2
}

PROVIDER_MAPPING = {
    "pro":       "google",
    "flash":     "google",
    "gptimage2": "openai",
}

async def save_image_async(image_obj_or_bytes, output_path: Path):
    """Offloads the file saving to a separate thread so it doesn't block the async event loop."""
    def _save():
        if hasattr(image_obj_or_bytes, 'save'):
            image_obj_or_bytes.save(output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(image_obj_or_bytes)

    await asyncio.to_thread(_save)

async def _call_google(img: Image.Image | None, output_path: Path, model_id: str, prompt: str, image_size: str, aspect_ratio: str = None):
    img_cfg = types.ImageConfig(image_size=image_size, aspect_ratio=aspect_ratio) if aspect_ratio \
              else types.ImageConfig(image_size=image_size)
    contents = [img, prompt] if img is not None else [prompt]
    response = await google_client.aio.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=img_cfg,
        )
    )

    for part in response.parts:
        if hasattr(part, 'as_image') and callable(part.as_image):
            await save_image_async(part.as_image(), output_path)
            return
        elif part.inline_data:
            await save_image_async(part.inline_data.data, output_path)
            return

    raise RuntimeError("No image data in response")

def _parse_aspect_ratio(ar: str) -> float:
    """Convert '16:9' string to a w/h float."""
    w, h = ar.split(":")
    return float(w) / float(h)

def _aspect_ratio_arg(value: str) -> str:
    """Argparse type validator — accepts 'W:H' strings like '16:9'."""
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Aspect ratio must be W:H (e.g. 16:9), got '{value}'")
    try:
        w, h = float(parts[0]), float(parts[1])
        if w <= 0 or h <= 0:
            raise argparse.ArgumentTypeError("Aspect ratio values must be positive")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid aspect ratio '{value}' — use W:H format")
    return value

def _compute_openai_size(src_w: int, src_h: int, minimize: bool = False, override_ratio: float = None) -> str:
    """Return a WxH string satisfying gpt-image-2 constraints at the source aspect ratio.

    minimize=False (default): largest valid resolution (for 4K mode).
    minimize=True:            smallest valid resolution (for 1K mode).

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

    if not minimize:
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

    else:
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

    return f"{w}x{h}"

async def _call_openai(img: Image.Image | None, output_path: Path, model_id: str, prompt: str, size: str, quality: str):
    if img is not None:
        # Edit mode: send source image alongside the prompt
        def _to_buf():
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            buf.name = "image.png"
            return buf

        buf = await asyncio.to_thread(_to_buf)
        response = await openai_client.images.edit(
            model=model_id,
            image=buf,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
    else:
        # Generate mode: pure prompt, no source image
        response = await openai_client.images.generate(
            model=model_id,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

    img_bytes = base64.b64decode(response.data[0].b64_json)
    await save_image_async(img_bytes, output_path)

async def _log_request(lock: asyncio.Lock, entry: dict):
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    async with lock:
        def _write():
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        await asyncio.to_thread(_write)

async def generate_variant(image_path: Path | None, variant_idx: int, semaphore: asyncio.Semaphore, log_lock: asyncio.Lock, model_key: str, model_id: str, suffix: str, extra_prompt: str, override_prompt: str, image_size: str, aspect_ratio: str, progress: Progress, estimates: dict):
    """Asynchronously calls the API to generate/redraw an image and saves it."""
    if image_path is not None:
        output_path = image_path.with_name(f"{image_path.stem}{suffix}{variant_idx}.png")
    else:
        output_path = Path.cwd() / f"generated{suffix}{variant_idx}.png"

    final_prompt = BASE_PROMPT
    if extra_prompt:
        final_prompt = f"{BASE_PROMPT}. {extra_prompt.strip()}"

    # --prompt fully replaces the base prompt when provided
    if override_prompt:
        final_prompt = override_prompt.strip()

    async with semaphore:
        estimated = estimates.get((model_key, image_size))
        est_label = f"/ ~{estimated:.0f}s" if estimated else "/ ?"
        src_label = image_path.name if image_path is not None else "[italic]prompt only[/italic]"
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
            img = None
            if image_path is not None:
                def _load_img():
                    src = Image.open(image_path)
                    src.load()
                    return src
                img = await asyncio.to_thread(_load_img)

            if PROVIDER_MAPPING[model_key] == "openai":
                override_ratio = _parse_aspect_ratio(aspect_ratio) if aspect_ratio else None
                src_dims = img.size if img is not None else (1, 1)
                if image_size == "1K":
                    resolved_size, quality = _compute_openai_size(*src_dims, minimize=True, override_ratio=override_ratio), "medium"
                elif img is None and override_ratio is None:
                    # No source image and no ratio hint: let the API choose
                    resolved_size, quality = "auto", "high"
                else:
                    resolved_size, quality = _compute_openai_size(*src_dims, override_ratio=override_ratio), "high"
                await _call_openai(img, output_path, model_id, final_prompt, resolved_size, quality)
            else:
                await _call_google(img, output_path, model_id, final_prompt, image_size, aspect_ratio)

            status = "success"
            progress.print(f"✅ Saved {output_path.name}")

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
                "source":           image_path.name if image_path is not None else None,
                "output":           output_path.name,
                "model_key":        model_key,
                "model_id":         model_id,
                "provider":         PROVIDER_MAPPING[model_key],
                "image_size":       image_size,
                "aspect_ratio":     aspect_ratio,
                "resolved_size":    resolved_size,
                "quality":          quality,
                "prompt":           final_prompt,
                "status":           status,
                "error":            error_msg,
            })

def _next_variant_index(image_path: Path | None, suffix: str) -> int:
    """Return the next available variant index for this image+suffix pair.
    When image_path is None (prompt-only mode), scans CWD for 'generated{suffix}N.png'."""
    directory = image_path.parent if image_path is not None else Path.cwd()
    stem      = image_path.stem   if image_path is not None else "generated"
    pattern = re.compile(
        rf"^{re.escape(stem)}{re.escape(suffix)}(\d+)\.png$",
        re.IGNORECASE,
    )
    max_idx = 0
    for f in directory.iterdir():
        m = pattern.match(f.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

async def async_main(image_files: list[Path], model_keys: list[str], extra_prompt: str, override_prompt: str, variants: int, image_size: str, aspect_ratio: str):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    log_lock  = asyncio.Lock()
    estimates = _load_estimates()

    model_desc   = " + ".join(MODEL_MAPPING[k] for k in model_keys)
    prompt_only  = not image_files
    sources      = image_files if not prompt_only else [None]

    if prompt_only:
        console.print(f"Prompt-only mode. Generating {variants} variant(s) per model — {model_desc}")
    else:
        console.print(f"Found {len(image_files)} image(s). Generating {variants} variant(s) each — {model_desc}")

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
        for img_path in sources:
            for model_key in model_keys:
                model_id = MODEL_MAPPING[model_key]
                suffix = SUFFIX_MAPPING[model_key]
                start_idx = _next_variant_index(img_path, suffix)
                for i in range(start_idx, start_idx + variants):
                    tasks.append(generate_variant(img_path, i, semaphore, log_lock, model_key, model_id, suffix, extra_prompt, override_prompt, image_size, aspect_ratio, progress, estimates))

        console.print(f"Firing off {len(tasks)} requests...")
        await asyncio.gather(*tasks)

    console.print("🎉 All done!")

def main():
    parser = argparse.ArgumentParser(description="Redraw images asynchronously using Gemini Image Models.")
    parser.add_argument("path", nargs="?", default=None,
                        help="Path to a single image or a folder of images. Omit to generate from prompt only.")
    parser.add_argument("--model", choices=["pro", "flash", "gptimage2", "all"], default="pro",
                        help="Choose 'pro' (Nano Banana Pro), 'flash' (Nano Banana 2), 'gptimage2' (GPT Image 2), or 'all' to run all three in parallel.")
    parser.add_argument("--extra-prompt", type=str, default="",
                        help="Additional text to append to the end of the base prompt.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Fully override the base prompt (--extra-prompt is ignored when this is set).")
    parser.add_argument("--1k", action="store_true", dest="use_1k",
                        help="Use 1K resolution output instead of the default 4K.")
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS_PER_IMAGE,
                        help=f"Number of variants to generate per image (default: {DEFAULT_VARIANTS_PER_IMAGE}).")
    parser.add_argument("--aspect-ratio", type=_aspect_ratio_arg, default=None,
                        help="Output aspect ratio as W:H (e.g. 16:9, 4:3, 1:1). Defaults to source image ratio.")
    args = parser.parse_args()

    model_keys    = ["pro", "flash", "gptimage2"] if args.model == "all" else [args.model]
    extra_prompt  = args.extra_prompt
    override_prompt = args.prompt
    image_size    = "1K" if args.use_1k else "4K"
    aspect_ratio  = args.aspect_ratio
    all_suffixes  = set(SUFFIX_MAPPING.values())

    image_files = []
    valid_exts  = {".jpg", ".jpeg", ".png", ".webp"}

    if args.path is not None:
        input_path = Path(args.path)
        if input_path.is_file():
            image_files.append(input_path)
        elif input_path.is_dir():
            image_files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        else:
            console.print("Invalid path provided.")
            return
        image_files = [f for f in image_files if not any(s in f.stem for s in all_suffixes)]
        if not image_files:
            console.print("No valid, unprocessed images found.")
            return
    else:
        # Prompt-only mode: a meaningful prompt is required
        if not override_prompt and not extra_prompt:
            console.print("[yellow]Warning:[/yellow] no source image and no --prompt given — the default redraw prompt will be used as-is.")


    asyncio.run(async_main(image_files, model_keys, extra_prompt, override_prompt, args.variants, image_size, aspect_ratio))

if __name__ == "__main__":
    main()
