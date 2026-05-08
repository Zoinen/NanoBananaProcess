import argparse
import asyncio
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

# Initialize the GenAI client
client = genai.Client()

# Configuration variables
BASE_PROMPT = "redraw this as if it was a professional photo taken on a modern high quality DSLR camera, but keep all the details as close to original as possible"
DEFAULT_VARIANTS_PER_IMAGE = 3
MAX_CONCURRENT_REQUESTS = 15 # Adjust based on your API tier's rate limits

# Model registry based on current Gemini documentation
MODEL_MAPPING = {
    "pro": "gemini-3-pro-image-preview",      # Nano Banana Pro
    "flash": "gemini-3.1-flash-image-preview" # Nano Banana 2
}

SUFFIX_MAPPING = {
    "pro": "_nbp",  # Nano Banana Pro
    "flash": "_nb2" # Nano Banana 2
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

async def generate_variant(image_path: Path, variant_idx: int, semaphore: asyncio.Semaphore, model_id: str, suffix: str, extra_prompt: str, image_size: str):
    """Asynchronously calls the API to redraw the image and saves it."""
    output_path = image_path.with_name(f"{image_path.stem}{suffix}{variant_idx}.png")

    # Combine the base prompt with the extra text if provided
    final_prompt = BASE_PROMPT
    if extra_prompt:
        final_prompt = f"{BASE_PROMPT}. {extra_prompt.strip()}"

    async with semaphore:
        print(f"🔄 Requesting: {image_path.name} -> {output_path.name} (Ratio: ORIGINAL, Res: {image_size}, Model: {model_id})")

        try:
            # Load source image
            def _load_img():
                img = Image.open(image_path)
                img.load()
                return img

            img = await asyncio.to_thread(_load_img)

            # Omitting 'aspect_ratio' forces the model to use the source image's exact ratio
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=[img, final_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        image_size=image_size
                    )
                )
            )

            # Process and save the returned image
            saved = False
            for part in response.parts:
                if hasattr(part, 'as_image') and callable(part.as_image):
                    await save_image_async(part.as_image(), output_path)
                    print(f"✅ Success: Saved {output_path.name}")
                    saved = True
                    break
                elif part.inline_data:
                    await save_image_async(part.inline_data.data, output_path)
                    print(f"✅ Success: Saved {output_path.name}")
                    saved = True
                    break

            if not saved:
                 print(f"❌ Failed: No image returned for {output_path.name}")

        except Exception as e:
            print(f"❌ Error processing {output_path.name}: {e}")

async def async_main(image_files: list[Path], model_keys: list[str], extra_prompt: str, variants: int, image_size: str):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = []
    for img_path in image_files:
        for model_key in model_keys:
            model_id = MODEL_MAPPING[model_key]
            suffix = SUFFIX_MAPPING[model_key]
            for i in range(1, variants + 1):
                tasks.append(generate_variant(img_path, i, semaphore, model_id, suffix, extra_prompt, image_size))

    model_desc = " + ".join(MODEL_MAPPING[k] for k in model_keys)
    print(f"Found {len(image_files)} unique image(s). Generating {variants} variant(s) each with model(s): {model_desc}...")
    print(f"Firing off {len(tasks)} asynchronous API requests...")

    await asyncio.gather(*tasks)
    print("🎉 All asynchronous tasks completed!")

def main():
    parser = argparse.ArgumentParser(description="Redraw images asynchronously using Gemini Image Models.")
    parser.add_argument("path", help="Path to a single image or a folder containing images.")
    parser.add_argument("--model", choices=["pro", "flash", "both"], default="pro",
                        help="Choose 'pro' for Nano Banana Pro, 'flash' for Nano Banana 2, or 'both' to run both in parallel.")
    parser.add_argument("--extra-prompt", type=str, default="",
                        help="Additional text to append to the end of the base prompt.")
    parser.add_argument("--1k", action="store_true", dest="use_1k",
                        help="Use 1K resolution output instead of the default 4K.")
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS_PER_IMAGE,
                        help=f"Number of variants to generate per image (default: {DEFAULT_VARIANTS_PER_IMAGE}).")
    args = parser.parse_args()

    input_path = Path(args.path)
    model_keys = ["pro", "flash"] if args.model == "both" else [args.model]
    extra_prompt = args.extra_prompt
    image_size = "1K" if args.use_1k else "4K"
    all_suffixes = set(SUFFIX_MAPPING.values())

    image_files = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    if input_path.is_file():
        image_files.append(input_path)
    elif input_path.is_dir():
        image_files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    else:
        print("Invalid path provided.")
        return

    image_files = [f for f in image_files if not any(s in f.stem for s in all_suffixes)]

    if not image_files:
        print("No valid, unprocessed images found.")
        return

    asyncio.run(async_main(image_files, model_keys, extra_prompt, args.variants, image_size))

if __name__ == "__main__":
    main()
