# Dataset

## Source

**Udacity CarND-Advanced-Lane-Lines**: https://github.com/udacity/CarND-Advanced-Lane-Lines

This is the public repository for Udacity's Self-Driving Car Nanodegree "Advanced Lane Finding" project. It is MIT-licensed; see the `LICENSE` file in that repository.

## Contents

| Item | Description |
|---|---|
| `project_video.mp4` | Main test video: highway driving in daylight, 1280×720. Used for the demo and for the labeled frames. |
| `challenge_video.mp4` | Harder video with tar seams, a concrete barrier and changing road surface. Used for `results/challenge.gif`. |
| `harder_challenge_video.mp4` | Winding road with sharp curves and strong shadows. Not used in this project. |
| `test_images/` | Still frames from the videos. |
| `camera_cal/` | Chessboard images for camera calibration. Not used, because this pipeline does no calibration. |
| `examples/`, `output_images/`, `writeup_template.md`, `example_writeup.pdf` | Udacity course material. |

## Getting the data

`data/` is gitignored, so the dataset is never committed. From `01-lane-detection/`, run:

```bash
git clone https://github.com/udacity/CarND-Advanced-Lane-Lines.git data/CarND-Advanced-Lane-Lines
```

The scripts expect the videos at `data/CarND-Advanced-Lane-Lines/*.mp4`.

## Hand-labeled ground truth

To evaluate the detector, **40 evenly spaced frames of `project_video.mp4` were hand-labeled** with `src/label_frames.py`, skipping the first and last frames. For each frame the annotator clicked four points: the bottom and top of the left lane line, then the bottom and top of the right lane line. The straight line through each pair of points is the ground truth for that side.

The labeling script writes:

```
data/labeled/frame_XXXX.jpg   # the 40 extracted frames
data/labeled/labels.json      # {"frame_XXXX.jpg": {"left": [[x,y],[x,y]], "right": [[x,y],[x,y]]}, ...}
```

Labels are saved after every frame, so the script can be stopped and resumed. To reproduce the labels, run `python src/label_frames.py` after cloning the dataset.
