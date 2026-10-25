"""Evaluate the detector against the hand-labeled frames in data/labeled/.

For every labeled frame we run lane.detect() (single frame, no smoothing) and
compare predicted vs. ground-truth x positions on 10 rows between Y_TOP*h and
the bottom of the image. A point is "correct" if |pred - gt| < px pixels.
If a side is not detected, its 10 points all count as wrong.

Example:
    python src/evaluate.py --canny_lo 80 --canny_hi 200 --px 10 --color
Each run appends a block to results/metrics.txt.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import lane

ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = ROOT / "data/labeled"
N_ROWS = 10


def gt_x(points, ys):
    """x on the straight line through the two clicked points, at rows ys."""
    (x1, y1), (x2, y2) = points
    a = (x2 - x1) / (y2 - y1)  # same x = a*y + b form as the detector
    return x1 + a * (ys - y1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canny_lo", type=int, default=50)
    p.add_argument("--canny_hi", type=int, default=150)
    p.add_argument("--px", type=float, default=20, help="pixel tolerance for a correct point")
    p.add_argument("--color", action="store_true", help="HLS white/yellow mask before Canny")
    p.add_argument("--out", default=str(ROOT / "results/metrics.txt"))
    args = p.parse_args()

    lane.CANNY_LO, lane.CANNY_HI, lane.USE_COLOR = args.canny_lo, args.canny_hi, args.color
    labels = json.loads((LABEL_DIR / "labels.json").read_text())

    n_frames = both = correct = total = 0
    errors = []  # absolute pixel errors of points where the side was detected
    for name, lab in sorted(labels.items()):
        img = cv2.imread(str(LABEL_DIR / name))
        if img is None:
            continue
        n_frames += 1
        h = img.shape[0]
        ys = np.linspace(lane.Y_TOP * h, h - 1, N_ROWS)

        L, R = lane.detect(img)
        both += (L is not None and R is not None)
        for pred, side in ((L, "left"), (R, "right")):
            total += N_ROWS
            if pred is None:
                continue  # all points of this side count as wrong
            err = np.abs(lane.x_at(pred, ys) - gt_x(lab[side], ys))
            errors.extend(err)
            correct += int((err < args.px).sum())

    config = (f"canny=({args.canny_lo},{args.canny_hi}) px={args.px:g} "
              f"color={'on' if args.color else 'off'}")
    report = (f"[{config}]\n"
              f"frames:               {n_frames}\n"
              f"both lanes detected:  {100 * both / n_frames:.1f}%\n"
              f"point accuracy:       {100 * correct / total:.1f}%  "
              f"({correct}/{total} points within {args.px:g}px)\n"
              f"mean pixel error:     {np.mean(errors):.1f} px (detected sides only)\n")
    print(report)
    with open(args.out, "a") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
