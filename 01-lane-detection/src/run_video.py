"""Run the lane detector on a video (or a folder of frames) and write an annotated mp4.

Example:
    python src/run_video.py --src data/CarND-Advanced-Lane-Lines/project_video.mp4 \
                            --out results/demo.mp4
"""
import argparse
import glob
import os
from pathlib import Path

import cv2

import lane

RESULTS = Path(__file__).resolve().parent.parent / "results"


def read_frames(src):
    """Yield BGR frames from a video file or from a folder of images."""
    if os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "*"))):
            img = cv2.imread(f)
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(src)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def smooth(prev, new, alpha):
    """Exponential moving average of line coefficients.

    new detection weighted by alpha, history by (1 - alpha). If this frame has
    no detection we keep the previous line, so the overlay does not flicker.
    """
    if new is None:
        return prev
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="video file or folder of frames")
    p.add_argument("--out", default=str(RESULTS / "demo.mp4"))
    p.add_argument("--alpha", type=float, default=0.3, help="EMA weight of new frame")
    p.add_argument("--fps", type=float, default=25, help="fps if --src is a folder")
    p.add_argument("--edges", default=str(RESULTS / "step_edges.jpg"),
                   help="where to save the Canny+ROI image of the first frame")
    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--color", action="store_true", help="HLS white/yellow mask before Canny")
    args = p.parse_args()
    lane.USE_COLOR = args.color

    fps = args.fps
    if not os.path.isdir(args.src):
        fps = cv2.VideoCapture(args.src).get(cv2.CAP_PROP_FPS) or args.fps

    stem = Path(args.out).stem
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    L = R = None
    n = 0
    for n, frame in enumerate(read_frames(args.src)):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            cv2.imwrite(args.edges, lane.edges(frame))  # pipeline illustration

        newL, newR = lane.detect(frame)
        L, R = smooth(L, newL, args.alpha), smooth(R, newR, args.alpha)
        out = lane.draw(frame, L, R)
        writer.write(out)

        if n % args.sample_every == 0:
            cv2.imwrite(str(RESULTS / f"{stem}_frame{n:04d}.jpg"), out)

    if writer is not None:
        writer.release()
    print(f"Processed {n + 1} frames -> {args.out}")


if __name__ == "__main__":
    main()
