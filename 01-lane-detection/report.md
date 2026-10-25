# Lane Detection and Departure Warning with Classical Computer Vision

## 1. Introduction

Lane-departure warning is one of the simplest driver-assistance features. The system finds the lane lines, works out where the car sits between them, and warns the driver when it drifts. This project builds such a system from a single dashcam using only classical image processing (edges, the Hough transform and least-squares fitting). It then measures how well this works against hand-labeled ground truth.

## 2. Dataset

The videos come from the Udacity CarND-Advanced-Lane-Lines repository [1]: `project_video.mp4` (daylight highway) and `challenge_video.mp4` (seams, barrier, varied surface), both 1280×720. Forty evenly spaced frames of `project_video.mp4` were hand-labeled by clicking two points on each lane line. That gives 40 frames × 2 sides × 10 evaluation rows = 800 test points.

## 3. Method

Each frame goes through these steps:

1. An optional HLS mask keeps only white and yellow pixels.
2. The frame is converted to grayscale, blurred with a 5×5 Gaussian and passed to Canny [2].
3. A trapezoidal region of interest keeps the road ahead, up to 62% of the image height.
4. A probabilistic Hough transform [3] finds line segments.
5. Segments are split into left and right by the sign of their slope. Near-horizontal segments (|slope| < 0.4) are dropped.
6. One line per side is fitted by least squares as x = a·y + b.

The departure offset is (image centre − lane centre) / lane width at the bottom row, and |offset| > 0.15 triggers a warning. In video, line coefficients are smoothed with an exponential moving average (α = 0.3).

For evaluation, predicted and labeled x positions are compared on 10 rows. A point is correct if it lies within a pixel tolerance (20 or 10 px). A missed side counts all 10 of its points as wrong.

## 4. Experiments & Results

Two Canny threshold pairs, with the colour mask on and off, at two tolerances:

| Canny | HLS mask | Both detected | Acc @ 20 px | Acc @ 10 px | Mean error |
|---|---|---|---|---|---|
| (50, 150) | off | 97.5% | 88.4% | 69.0% | 9.1 px |
| (50, 150) | on  | 100.0% | 93.6% | 77.8% | 7.6 px |
| (80, 200) | off | 95.0% | 87.6% | 72.0% | 8.9 px |
| (80, 200) | on  | 100.0% | **94.1%** | **78.9%** | **7.5 px** |

The colour mask is the main improvement. It adds 5.2–6.5 points of accuracy at 20 px and 6.9–8.8 points at 10 px, and it removes every missed lane. Stricter Canny thresholds (80, 200) help a little at 10 px. Without the mask, though, they miss more lanes (95.0% both detected versus 97.5%), because faint markings no longer produce edges. Tightening the tolerance from 20 px to 10 px cuts accuracy by 15–19 points, so most of the remaining error is in the 10–20 px range.

## 5. Discussion

- **Curves.** The model is a straight line, so on bends the fitted line follows the near part of the lane and drifts off it further ahead. The ground truth is also a straight line, so this error is not captured by the numbers above.
- **Shadows and surface changes.** Shadow edges and the borders of pale concrete patches are strong grayscale gradients, which Canny and Hough can mistake for lane segments. The colour mask removes most of these because they are neither white nor yellow, which is consistent with the 100% detection rate when the mask is on. Even with good overall numbers, the left line sometimes sits inside the yellow marking on pale concrete (frame 600 of the demo).
- **Challenge video.** Tar seams that run parallel to the lane, and the base of the concrete barrier, are long, straight and high-contrast. Inside the ROI they look exactly like lane lines to Hough. In `challenge_video.mp4`, the right line often locks onto a seam instead of the dashed marking. The EMA smoothing makes the overlay stable but does not fix where it points.

## 6. Conclusion & Future Work

Using Canny (80, 200) with the HLS mask, the pipeline found both lanes in every test frame with a mean error of 7.5 px. That is enough to drive a working departure warning on clean highway footage. Next steps:

- Camera calibration and a bird's-eye perspective transform with second-order polynomial fits, to handle curves and give an offset in metres.
- Rejecting outlier segments by expected lane width and by consistency with the previous frame.
- A larger labeled set that includes the challenge videos.
- Comparison with a learned lane detector.

## References

1. Udacity, *CarND-Advanced-Lane-Lines*. https://github.com/udacity/CarND-Advanced-Lane-Lines
2. J. Canny, "A Computational Approach to Edge Detection," *IEEE TPAMI*, 8(6), 1986.
3. J. Matas, C. Galambos, J. Kittler, "Robust Detection of Lines Using the Progressive Probabilistic Hough Transform," *CVIU*, 78(1), 2000.
4. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
