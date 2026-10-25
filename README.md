# Computer Vision Projects

Five computer vision projects for robotics and driver assistance, from classical image processing to deep learning.

| Preview | Project | CV concept | Key result |
|---|---|---|---|
| <img src="01-lane-detection/results/demo.gif" width="220"> | [01 · Lane Detection + Departure Warning](01-lane-detection/) | HLS colour mask, Canny, Hough transform | 94.1% of points within 20 px, 7.5 px mean error |
| | [02 · Gesture Recognition for Robot Control](02-gesture-robot-control/) | | In progress |
| | [03 · Multi-Object Tracking](03-multi-object-tracking/) | | In progress |
| | [04 · Explainable AI for Object Recognition](04-explainable-ai/) | | In progress |
| | [05 · Vision-Based Pick-and-Place Detection](05-pick-and-place-detection/) | | In progress |

## Setup

```bash
git clone https://github.com/fidhafs/computer-vision-projects.git
cd computer-vision-projects
python -m venv .venv
source .venv/bin/activate

# install the dependencies of the project you want to run, e.g.
pip install -r 01-lane-detection/requirements.txt
```

Each project folder has its own README with dataset and run instructions.

---

Author: Fidha FS, B.Tech Robotics & Automation, Karunya University
