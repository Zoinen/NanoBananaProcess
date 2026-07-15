"""
Color-match destination image to source image using histogram matching in LAB space.

Usage:
    python color_match.py <source> <destination> [output]

    source      : reference image (untouched)
    destination : image to remap
    output      : output path (default: <destination_dir>/matched_<destination_name>)

All three channels (L, a, b) are matched so both contrast and color align.
"""

import sys
import cv2
import numpy as np
from pathlib import Path


def load_lab(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    assert bgr is not None, f"Could not read {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def channel_cdf(channel: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum().astype(np.float64)
    cdf /= cdf[-1]
    return cdf


def build_lut(src_cdf: np.ndarray, tgt_cdf: np.ndarray) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    tgt_idx = 0
    for src_val in range(256):
        while tgt_idx < 255 and tgt_cdf[tgt_idx] < src_cdf[src_val]:
            tgt_idx += 1
        lut[src_val] = tgt_idx
    return lut


def match_colors(src_path: Path, dst_path: Path, out_path: Path) -> None:
    src_lab = load_lab(src_path)
    dst_lab = load_lab(dst_path)

    result = dst_lab.copy()
    for c in range(3):
        src_cdf = channel_cdf(src_lab[:, :, c])
        dst_cdf = channel_cdf(dst_lab[:, :, c])
        lut = build_lut(dst_cdf, src_cdf)
        result[:, :, c] = lut[dst_lab[:, :, c]]

    bgr_out = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr_out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python color_match.py <source> <destination> [output]")
        sys.exit(1)

    src  = Path(sys.argv[1])
    dst  = Path(sys.argv[2])
    out  = Path(sys.argv[3]) if len(sys.argv) > 3 else dst.parent / f"{dst.stem}_matched{dst.suffix}"

    match_colors(src, dst, out)
