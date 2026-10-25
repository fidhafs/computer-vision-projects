"""Hand-label lane lines on ~40 evenly spaced frames of project_video.mp4.

For each frame click 4 points, in this order:
    1. left lane  - bottom      2. left lane  - top
    3. right lane - bottom      4. right lane - top
("top" = anywhere near the far end of the visible lane line, roughly 62% down the image.)

Keys:
    SPACE / ENTER  accept the 4 points and go to the next frame
    r              reset the clicks on this frame
    s              skip this frame (e.g. no clear lane lines)
    q              quit (everything labeled so far is already saved)

Output: data/labeled/frame_XXXX.jpg and data/labeled/labels.json.
Re-running resumes: frames already in labels.json are skipped.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "data/CarND-Advanced-Lane-Lines/project_video.mp4"
OUT_DIR = ROOT / "data/labeled"
NAMES = ["left bottom", "left top", "right bottom", "right top"]

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))


def render(frame):
    """Draw the clicks so far plus an instruction line."""
    img = frame.copy()
    for i, (x, y) in enumerate(clicks):
        color = (255, 0, 0) if i < 2 else (0, 0, 255)  # blue = left, red = right
        cv2.circle(img, (x, y), 6, color, -1)
    if len(clicks) >= 2:
        cv2.line(img, clicks[0], clicks[1], (255, 0, 0), 2)
    if len(clicks) == 4:
        cv2.line(img, clicks[2], clicks[3], (0, 0, 255), 2)
        msg = "SPACE=accept  r=reset  s=skip  q=quit"
    else:
        msg = f"click: {NAMES[len(clicks)]}   (r=reset  s=skip  q=quit)"
    cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=str(DEFAULT_VIDEO))
    p.add_argument("--n", type=int, default=40, help="number of frames to extract")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_file = OUT_DIR / "labels.json"
    labels = json.loads(label_file.read_text()) if label_file.exists() else {}

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Evenly spaced frame indices, skipping the very first/last frame.
    indices = np.linspace(0, total - 1, args.n + 2).astype(int)[1:-1]

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)

    for k, idx in enumerate(indices):
        name = f"frame_{idx:04d}.jpg"
        if name in labels:
            continue  # already labeled in a previous session
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        clicks.clear()
        print(f"[{k + 1}/{len(indices)}] {name}")

        while True:
            cv2.imshow("label", render(frame))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                print(f"Quit. {len(labels)} frames labeled in {label_file}")
                return
            if key == ord("r"):
                clicks.clear()
            if key == ord("s"):
                print("  skipped")
                break
            if key in (13, 32) and len(clicks) == 4:  # ENTER or SPACE
                cv2.imwrite(str(OUT_DIR / name), frame)
                labels[name] = {"left": [list(clicks[0]), list(clicks[1])],
                                "right": [list(clicks[2]), list(clicks[3])]}
                label_file.write_text(json.dumps(labels, indent=1))  # save every frame
                break

    cv2.destroyAllWindows()
    print(f"Done. {len(labels)} frames labeled in {label_file}")


if __name__ == "__main__":
    main()
