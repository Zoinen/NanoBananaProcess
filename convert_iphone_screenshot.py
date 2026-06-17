from pathlib import Path
from typing import Optional
import argparse
import os
import shutil
import subprocess
import sys


def _convert_p3_to_srgb_software(
    src: Path,
    dst: Path,
    strip_metadata: bool = False,
    quality: Optional[int] = 95,
) -> None:
    """
    Display P3 → 8-bit sRGB via Pillow + NumPy.
    Used as fallback when no ICC profile file is available.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "NumPy is required for software P3→sRGB conversion. "
            "Install it with: pip install numpy"
        )

    from PIL import Image, ImageOps

    img = Image.open(src)
    img = ImageOps.exif_transpose(img)

    has_alpha = img.mode in ("RGBA", "LA", "PA")
    img = img.convert("RGBA" if has_alpha else "RGB")

    arr = np.array(img, dtype=np.float64) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4] if has_alpha else None

    # sRGB gamma decode (Display P3 uses the same transfer function as sRGB)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # Display P3 → linear sRGB (both D65; row sums = 1)
    M = np.array([
        [ 1.22494018, -0.22494018,  0.00000000],
        [-0.04205695,  1.04205695,  0.00000000],
        [-0.01963755, -0.07863605,  1.09827360],
    ])

    h, w = linear.shape[:2]
    srgb_linear = np.clip((linear.reshape(-1, 3) @ M.T).reshape(h, w, 3), 0.0, 1.0)

    # sRGB gamma encode
    srgb = np.where(
        srgb_linear <= 0.0031308,
        srgb_linear * 12.92,
        1.055 * srgb_linear ** (1.0 / 2.4) - 0.055,
    )
    srgb = np.clip(srgb, 0.0, 1.0)

    out_arr = np.concatenate([srgb, alpha], axis=2) if alpha is not None else srgb
    out_mode = "RGBA" if alpha is not None else "RGB"
    out_img = Image.fromarray((out_arr * 255.0 + 0.5).astype(np.uint8), mode=out_mode)

    dst.parent.mkdir(parents=True, exist_ok=True)

    suffix = dst.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        out_img = out_img.convert("RGB")
        save_kw = {"quality": quality} if quality is not None else {}
        out_img.save(dst, format="JPEG", **save_kw)
    elif suffix == ".webp":
        save_kw = {"quality": quality} if quality is not None else {}
        out_img.save(dst, format="WEBP", **save_kw)
    elif suffix in {".tif", ".tiff"}:
        out_img.save(dst, format="TIFF")
    else:
        out_img.save(dst, format="PNG")

    print(f"Saved: {dst}")


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

    raise FileNotFoundError(
        "Could not find Windows sRGB ICC/ICM profile."
    )


def find_display_p3_profile() -> Optional[Path]:
    patterns = [
        "*Display*P3*.icc",
        "*Display*P3*.icm",
        "*P3*.icc",
        "*P3*.icm",
    ]

    for pattern in patterns:
        matches = sorted(WINDOWS_COLOR_DIR.glob(pattern))
        if matches:
            return matches[0]

    return None


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

    return (
        "profile-icc" in text
        or "icc:" in text
        or "icc profile" in text
    )


def convert_image(
    source: Path,
    destination: Path,
    srgb_profile: Path,
    p3_profile: Optional[Path],
    force_assign_p3: bool,
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

    cmd = [
        "magick",
        str(source),
        "-auto-orient",
    ]

    embedded_icc = has_embedded_icc(source)

    if force_assign_p3:
        if p3_profile is None:
            print(
                "No Display P3 profile found; using software P3→sRGB conversion.",
                file=sys.stderr,
            )
            _convert_p3_to_srgb_software(source, destination, strip_metadata, quality)
            return

        # Remove existing ICC profile, then assign Display P3 as source.
        cmd += [
            "+profile",
            "icc",
            "-profile",
            str(p3_profile),
        ]

    elif not embedded_icc:
        if p3_profile is None:
            print(
                "No Display P3 profile found; using software P3→sRGB conversion.",
                file=sys.stderr,
            )
            _convert_p3_to_srgb_software(source, destination, strip_metadata, quality)
            return

        # Assign Display P3 as source profile.
        cmd += [
            "-profile",
            str(p3_profile),
        ]

    # Convert to sRGB and reduce to 8-bit.
    cmd += [
        "-profile",
        str(srgb_profile),
        "-depth",
        "8",
    ]

    if quality is not None and destination.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
        cmd += [
            "-quality",
            str(quality),
        ]

    if strip_metadata:
        # Strip only after color conversion.
        cmd += ["-strip"]

    suffix = destination.suffix.lower()

    if suffix == ".png":
        output_spec = f"PNG32:{destination}"
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
        description="Convert iPhone Display P3 / 16-bit images to 8-bit sRGB images."
    )

    parser.add_argument(
        "source",
        help="Source image file or folder.",
    )

    parser.add_argument(
        "destination",
        help="Destination image file or output folder.",
    )

    parser.add_argument(
        "--p3-profile",
        help="Path to Display P3 ICC profile.",
    )

    parser.add_argument(
        "--srgb-profile",
        help="Path to sRGB ICC/ICM profile. Defaults to Windows system sRGB profile.",
    )

    parser.add_argument(
        "--force-assign-p3",
        action="store_true",
        help="Ignore embedded ICC profile and assign Display P3 before converting to sRGB.",
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
        p3_profile = Path(args.p3_profile) if args.p3_profile else find_display_p3_profile()

        if not srgb_profile.exists():
            raise FileNotFoundError(f"sRGB profile not found: {srgb_profile}")

        if p3_profile is not None and not p3_profile.exists():
            raise FileNotFoundError(f"Display P3 profile not found: {p3_profile}")

        print("sRGB profile:", srgb_profile)
        print("Display P3 profile:", p3_profile or "not found")

        if source.is_file():
            if destination.suffix:
                # Destination has an extension, so it is a file path.
                final_destination = destination
            else:
                # Destination has no extension, so it is an output folder.
                destination.mkdir(parents=True, exist_ok=True)
                final_destination = destination / source.name

            convert_image(
                source=source,
                destination=final_destination,
                srgb_profile=srgb_profile,
                p3_profile=p3_profile,
                force_assign_p3=args.force_assign_p3,
                strip_metadata=args.strip,
                quality=args.quality,
            )

        elif source.is_dir():
            if destination.suffix:
                raise ValueError(
                    "Source is a folder, so destination must also be a folder, not a file."
                )

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
                    p3_profile=p3_profile,
                    force_assign_p3=args.force_assign_p3,
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
