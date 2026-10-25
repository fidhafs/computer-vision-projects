# Lane Detection + Lane Departure Warning (Classical CV)

![demo](results/demo.gif)

## Problem Statement

A driver-assistance system needs to know where the lane is and whether the car is drifting out of it. This project does that with a single front-facing dashcam and classical computer vision only, without deep learning or training data.

## Objective

- Detect the left and right lane lines in each video frame.
- Estimate the car's lateral offset from the lane centre and show a **DEPARTURE WARNING** when it gets too large.
- Measure detection accuracy against hand-labeled ground truth and compare a few pipeline settings.

## Dataset

Videos come from the [Udacity CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) repository (1280×720 highway dashcam footage). `project_video.mp4` is the main test video and `challenge_video.mp4` is the harder one.

For evaluation, 40 evenly spaced frames of `project_video.mp4` were hand-labeled with `src/label_frames.py`. On each frame, two points were clicked on each lane line. See [DATASET.md](DATASET.md) for details. The data is not committed to this repo.

## Methodology

The pipeline in `src/lane.py` runs once per frame:

1. **HLS colour mask** (on by default, `--no-color` to disable): keep only white pixels (lightness ≥ 200) and yellow pixels (hue 15–35, saturation ≥ 80).
2. **Grayscale → 5×5 Gaussian blur → Canny edges** with thresholds (80, 200) by default.
3. **Region of interest:** a trapezoid covering the road ahead, from the bottom of the image up to 62% of its height.
4. **Probabilistic Hough transform** to find line segments.
5. **Left/right split by slope:** drop near-horizontal segments (|slope| < 0.4). Negative slope in the left half is the left lane; positive slope in the right half is the right lane.
6. **Line fit:** least squares per side, as `x = a·y + b`. Fitting x as a function of y keeps near-vertical lines stable.
7. **Departure warning:** offset = (image centre − lane centre) / lane width, measured at the bottom row. If |offset| > 0.15, the frame shows a warning.

For video (`src/run_video.py`), the line coefficients are smoothed with an exponential moving average (α = 0.3). A frame with no detection keeps the previous line.

**Evaluation (`src/evaluate.py`):** each labeled frame is run through `detect()` alone, with no smoothing. Predicted and ground-truth x positions are compared on 10 rows between 62% of the height and the bottom of the image. A point counts as correct if it is within the pixel tolerance. If a side is not detected, all 10 of its points count as wrong.

## Tools/Libraries

Python 3.10, OpenCV (`opencv-python`), NumPy (<2), Matplotlib.

## How to Run

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate
pip install -r 01-lane-detection/requirements.txt
cd 01-lane-detection

# 1. get the data (see DATASET.md)
git clone https://github.com/udacity/CarND-Advanced-Lane-Lines.git data/CarND-Advanced-Lane-Lines

# 2. run on a video -> annotated mp4 + sample frames in results/
python src/run_video.py --src data/CarND-Advanced-Lane-Lines/project_video.mp4 --out results/demo.mp4

# 3. (optional) label your own ground truth
python src/label_frames.py

# 4. evaluate (appends to results/metrics.txt); defaults are Canny (80, 200) + colour mask
python src/evaluate.py --px 20
python src/evaluate.py --canny_lo 50 --canny_hi 150 --no-color   # baseline
```

## Results

Evaluated on 40 hand-labeled frames (800 points: 40 frames × 2 sides × 10 rows). Mean pixel error is over detected sides only and does not depend on the tolerance.

| Canny (lo, hi) | HLS mask | Both lanes detected | Accuracy @ 20 px | Accuracy @ 10 px | Mean error |
|---|---|---|---|---|---|
| (50, 150) | off | 97.5% | 88.4% (707/800) | 69.0% (552/800) | 9.1 px |
| (50, 150) | on  | 100.0% | 93.6% (749/800) | 77.8% (622/800) | 7.6 px |
| (80, 200) | off | 95.0% | 87.6% (701/800) | 72.0% (576/800) | 8.9 px |
| (80, 200) | on  | **100.0%** | **94.1% (753/800)** | **78.9% (631/800)** | **7.5 px** |

The best setting is **Canny (80, 200) with the HLS colour mask**. The colour mask has the biggest effect: it adds about 6 points of accuracy at 20 px and about 7–9 points at 10 px, and it brings detection of both lanes to 100%. The stricter Canny thresholds help only slightly once the mask is on. This setting is now the default in `src/lane.py`, and `demo.gif` and `challenge.gif` were generated with it.

**Full videos (no labels, detection only).** Percentage of frames in which both lanes were found by `detect()` alone, without smoothing:

| Video | Canny (50, 150), no mask | Canny (80, 200) + mask |
|---|---|---|
| `project_video.mp4` | 99.4% (1252/1260) | 100.0% (1260/1260) |
| `challenge_video.mp4` | 95.9% (464/484) | 96.1% (465/484) |

The colour mask fixes the main video, but it barely changes the result on the challenge video.

Sample output frame (`project_video.mp4`, frame 1000):

![sample frame](results/demo_frame1000.jpg)

Edge map after Canny + ROI (first frame): [results/step_edges.jpg](results/step_edges.jpg). Output on the challenge video: [results/challenge.gif](results/challenge.gif).

## Limitations

- **Straight-line model only.** Curved lanes are approximated by one straight line. The ground truth is also a straight line through two clicks, so the metric does not penalise curvature errors.
- **Harder roads.** On `challenge_video.mp4`, one lane side is still missed in 19 of 484 frames (3.9%) with the best setting. The colour mask helps little there, because the faded markings, tar seams and concrete barrier are not cleanly separated by fixed white/yellow thresholds. The overlay is held by temporal smoothing in those frames, and there are no labels for this video, so its accuracy is not measured.
- **Hand-tuned constants.** The ROI, Hough parameters, slope cutoff and colour thresholds are fixed for this camera and do not adapt to lighting or mounting.
- **Small evaluation set.** 40 frames from a single video, labeled by one person.
- **No camera calibration or perspective transform.** The offset is a fraction of lane width, not metres.

## Conclusion

A simple Canny + Hough pipeline with an HLS white/yellow mask finds both lane lines in all 40 test frames and in every frame of the main video. 94.1% of points are within 20 px of the hand labels, and the mean error is 7.5 px. That is enough to drive a working lane-departure warning on the main highway video. The approach is weaker on curved lanes and on roads with seams and changing surfaces: on the challenge video it still misses a side in 3.9% of frames. A perspective transform with polynomial fitting, or a learned detector, would be the next step.
