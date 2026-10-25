# Lane Detection and Departure Warning with Classical Computer Vision

## 1. Introduction

Lane-departure warning is one of the simplest driver-assistance features. The system finds the lane lines, works out where the car sits between them, and warns the driver when it drifts. I built one from a single dashcam using only classical image processing: edges, the Hough transform and least-squares fitting. Then I measured how well it works against frames I labeled by hand.

## 2. Dataset

I used two videos from the Udacity CarND-Advanced-Lane-Lines repository [1], both 1280×720. `project_video.mp4` is daylight highway driving. `challenge_video.mp4` has seams, a barrier and a changing road surface. I hand-labeled 40 evenly spaced frames of `project_video.mp4` by clicking two points on each lane line. That gives 40 frames × 2 sides × 10 evaluation rows = 800 test points.

## 3. Method

Each frame goes through these steps:

1. An optional HLS mask keeps only white and yellow pixels.
2. The frame is converted to grayscale, blurred with a 5×5 Gaussian and passed to Canny [2].
3. A trapezoidal region of interest keeps the road ahead, up to 62% of the image height.
4. A probabilistic Hough transform [3] finds line segments.
5. Segments are split into left and right by the sign of their slope. Near-horizontal segments (|slope| < 0.4) are dropped.
6. One line per side is fitted by least squares as x = a·y + b.

Based on the experiments below, I made the HLS mask and Canny (80, 200) the defaults. The departure offset is (image centre − lane centre) / lane width at the bottom row. If |offset| > 0.15, the system shows a warning. In video, I smooth the line coefficients with an exponential moving average (α = 0.3).

To evaluate, I compare predicted and labeled x positions on 10 rows. A point is correct if it is within a pixel tolerance (20 or 10 px). If a side is missed, all 10 of its points count as wrong.

## 4. Experiments & Results

I tried two Canny threshold pairs, with the colour mask on and off, at two tolerances:

| Canny | HLS mask | Both detected | Acc @ 20 px | Acc @ 10 px | Mean error |
|---|---|---|---|---|---|
| (50, 150) | off | 97.5% | 88.4% | 69.0% | 9.1 px |
| (50, 150) | on  | 100.0% | 93.6% | 77.8% | 7.6 px |
| (80, 200) | off | 95.0% | 87.6% | 72.0% | 8.9 px |
| (80, 200) | on  | 100.0% | **94.1%** | **78.9%** | **7.5 px** |

The colour mask gave the biggest improvement. It adds 5.2–6.5 points of accuracy at 20 px and 6.9–8.8 points at 10 px. With it on, no lanes are missed. Stricter Canny thresholds (80, 200) help a little at 10 px. Without the mask, though, they miss more lanes (95.0% both detected versus 97.5%), because faint markings stop producing edges. Going from 20 px to 10 px tolerance cuts accuracy by 15–19 points, so most of the remaining error is between 10 and 20 px.

I also checked the full videos. There are no labels for them, so I only measured how often both sides were detected. The best setting reaches 100.0% on `project_video.mp4` (baseline 99.4%) and 96.1% on `challenge_video.mp4` (baseline 95.9%).

## 5. Discussion

- **Curves.** The model is a straight line. On bends, the fitted line follows the near part of the lane and drifts off it further ahead. My ground truth is also a straight line, so the numbers above don't capture this error.
- **Shadows and surface changes.** Shadow edges and the borders of pale concrete patches are strong grayscale gradients, and Canny and Hough can mistake them for lane segments. The colour mask removes most of them, since they are neither white nor yellow. That fits with the 100% detection rate when the mask is on. With the baseline settings, the left line sat inside the yellow marking on pale concrete (demo frame 600). With the mask on, it follows the marking.
- **Challenge video.** Tar seams running alongside the lane, and the base of the concrete barrier, are long, straight and high-contrast. Inside the ROI, Hough sees them as lane lines. With the baseline settings, the right line locked onto a seam (challenge frame 200). The colour mask removes much of this, but a side is still missed in 19 of 484 frames (3.9%). Without the mask it was 20 frames, so almost the same. Fixed colour thresholds aren't enough for faded markings. The EMA smoothing holds the previous line through a miss, but it can't fix a line that is wrong.

## 6. Conclusion & Future Work

With Canny (80, 200) and the HLS mask, my pipeline found both lanes in every test frame, with a mean error of 7.5 px. That's enough for a working departure warning on clean highway footage. Next, I would like to try:

- Camera calibration and a bird's-eye perspective transform with second-order polynomial fits, to handle curves and give the offset in metres.
- Rejecting outlier segments using the expected lane width and the previous frame.
- A larger labeled set that includes the challenge videos.
- Comparing against a learned lane detector.

## References

1. Udacity, *CarND-Advanced-Lane-Lines*. https://github.com/udacity/CarND-Advanced-Lane-Lines
2. J. Canny, "A Computational Approach to Edge Detection," *IEEE TPAMI*, 8(6), 1986.
3. J. Matas, C. Galambos, J. Kittler, "Robust Detection of Lines Using the Progressive Probabilistic Hough Transform," *CVIU*, 78(1), 2000.
4. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
