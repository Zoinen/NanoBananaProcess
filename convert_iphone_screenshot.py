from pathlib import Path
from typing import Optional
import argparse
import os
import shutil
import subprocess
import sys


WINDOWS_COLOR_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "spool" / "drivers" / "color"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
}


def find_srgb_profile() -> Path:
    candidates = [
        WINDOWS_COLOR_DIR / "sRGB Color Space Profile.icm",
        WINDOWS_COLOR_DIR / "sRGB.icc",
        WINDOWS_COLOR_DIR / "sRGB.icm",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find Windows sRGB ICC/ICM profile.")


def has_embedded_icc(path: Path) -> bool:
    result = subprocess.run(
        ["magick", "identify", "-verbose", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(f"identify failed for {path}:\n{result.stderr}")

    text = result.stdout.lower()
    return "profile-icc" in text or "icc:" in text or "icc profile" in text


def convert_image(
    source: Path,
    destination: Path,
    srgb_profile: Path,
    strip_metadata: bool,
    quality: Optional[int],
) -> None:
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(
            f"Destination looks like a file path, but a folder already exists there:\n"
            f"{destination}\n\n"
            f"Delete that folder or choose a different output filename."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["magick", str(source), "-auto-orient"]

    if has_embedded_icc(source):
        cmd += ["-profile", str(srgb_profile)]

    cmd += ["-depth", "8"]

    if quality is not None and destination.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
        cmd += ["-quality", str(quality)]

    if strip_metadata:
        cmd += ["-strip"]

    suffix = destination.suffix.lower()
    if suffix == ".png":
        output_spec = f"PNG:{destination}"
    elif suffix in {".jpg", ".jpeg"}:
        output_spec = f"JPEG:{destination}"
    elif suffix == ".webp":
        output_spec = f"WEBP:{destination}"
    elif suffix in {".tif", ".tiff"}:
        output_spec = f"TIFF:{destination}"
    else:
        output_spec = str(destination)

    cmd += [output_spec]

    print(f"Converting: {source} -> {destination}")
    subprocess.run(cmd, check=True)


def collect_images(source: Path) -> list[Path]:
    return sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert iPhone Display P3 / 16-bit images to 8-bit sRGB."
    )

    parser.add_argument("source", help="Source image file or folder.")
    parser.add_argument("destination", help="Destination image file or output folder.")

    parser.add_argument(
        "--srgb-profile",
        help="Path to sRGB ICC/ICM profile. Defaults to Windows system sRGB profile.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip metadata after color conversion.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/WebP quality, default: 95.",
    )

    args = parser.parse_args()

    if shutil.which("magick") is None:
        print("Error: ImageMagick 'magick' command was not found in PATH.", file=sys.stderr)
        return 1

    source = Path(args.source)
    destination = Path(args.destination)

    try:
        srgb_profile = Path(args.srgb_profile) if args.srgb_profile else find_srgb_profile()

        if not srgb_profile.exists():
            raise FileNotFoundError(f"sRGB profile not found: {srgb_profile}")

        print("sRGB profile:", srgb_profile)

        if source.is_file():
            final_destination = destination if destination.suffix else destination / source.name
            if not destination.suffix:
                destination.mkdir(parents=True, exist_ok=True)

            convert_image(
                source=source,
                destination=final_destination,
                srgb_profile=srgb_profile,
                strip_metadata=args.strip,
                quality=args.quality,
            )

        elif source.is_dir():
            if destination.suffix:
                raise ValueError("Source is a folder, so destination must also be a folder, not a file.")

            destination.mkdir(parents=True, exist_ok=True)

            images = collect_images(source)
            if not images:
                print("No supported image files found.")
                return 0

            for image in images:
                convert_image(
                    source=image,
                    destination=destination / image.name,
                    srgb_profile=srgb_profile,
                    strip_metadata=args.strip,
                    quality=args.quality,
                )

        else:
            raise FileNotFoundError(f"Source does not exist: {source}")

    except subprocess.CalledProcessError as e:
        print(f"ImageMagick failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
