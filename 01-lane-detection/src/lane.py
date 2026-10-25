"""Classical lane detection pipeline (no deep learning).

Steps: HLS white/yellow colour mask (on by default, USE_COLOR) -> grayscale
-> Gaussian blur -> Canny edges -> trapezoid region of interest
-> probabilistic Hough transform -> split segments into left/right by slope
-> fit one straight line per side as x = a*y + b.

Public API:
    Y_TOP                   fraction of image height where the ROI / drawn lanes stop
    edges(frame)            Canny edge map restricted to the ROI
    detect(frame)           -> (L, R), each a numpy array [a, b] or None
    draw(frame, L, R)       -> annotated copy of frame (lane shading, lines, warning)
"""
import cv2
import numpy as np

# ---- Tunable parameters ---------------------------------------------------
Y_TOP = 0.62            # top of the trapezoid ROI, as a fraction of image height
CANNY_LO, CANNY_HI = 80, 200   # best of the evaluated settings (see README)
MIN_SLOPE = 0.4         # ignore near-horizontal segments (|dy/dx| below this)
WARN_THRESHOLD = 0.15   # |offset| above this (fraction of lane width) -> warning
USE_COLOR = True        # if True, keep only white/yellow pixels (HLS) before Canny


def color_mask(frame):
    """Keep only white and yellow pixels using the HLS colour space.

    White = high lightness. Yellow = hue around 15-35 (OpenCV hue is 0-180)
    with reasonable saturation. Everything else is set to black, which removes
    edges from shadows, tar seams and the road/grass border.
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 200, 0), (180, 255, 255))
    yellow = cv2.inRange(hls, (15, 30, 80), (35, 220, 255))
    mask = cv2.bitwise_or(white, yellow)
    return cv2.bitwise_and(frame, frame, mask=mask)


def roi_mask(h, w):
    """Binary mask of a trapezoid covering the road in front of the car."""
    pts = np.array([[
        (int(0.05 * w), h),             # bottom-left
        (int(0.45 * w), int(Y_TOP * h)),  # top-left
        (int(0.55 * w), int(Y_TOP * h)),  # top-right
        (int(0.95 * w), h),             # bottom-right
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def edges(frame):
    """Grayscale -> blur -> Canny, then keep only edges inside the ROI."""
    if USE_COLOR:
        frame = color_mask(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    e = cv2.Canny(blur, CANNY_LO, CANNY_HI)
    h, w = e.shape
    return cv2.bitwise_and(e, roi_mask(h, w))


def fit_line(xs, ys):
    """Least-squares fit x = a*y + b. We fit x as a function of y because
    lane lines are close to vertical, where y = m*x + c would blow up."""
    if len(xs) < 2:
        return None
    return np.polyfit(ys, xs, 1)  # returns [a, b]


def detect(frame):
    """Return (L, R): fitted line coefficients [a, b] for each side, or None."""
    e = edges(frame)
    h, w = e.shape
    segs = cv2.HoughLinesP(e, rho=2, theta=np.pi / 180, threshold=20,
                           minLineLength=20, maxLineGap=100)
    lx, ly, rx, ry = [], [], [], []
    if segs is not None:
        for x1, y1, x2, y2 in segs[:, 0]:
            if x1 == x2:
                continue  # perfectly vertical, slope undefined; skip
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < MIN_SLOPE:
                continue  # almost horizontal: shadows, cracks, car bonnet
            # Image y grows downward, so the left lane rises to the right
            # (negative slope) and the right lane has positive slope.
            if slope < 0 and max(x1, x2) < w / 2:
                lx += [x1, x2]
                ly += [y1, y2]
            elif slope > 0 and min(x1, x2) > w / 2:
                rx += [x1, x2]
                ry += [y1, y2]
    return fit_line(lx, ly), fit_line(rx, ry)


def x_at(line, y):
    """Evaluate x = a*y + b."""
    a, b = line
    return a * y + b


def draw(frame, L, R):
    """Draw the lane overlay and the departure warning on a copy of frame."""
    out = frame.copy()
    h, w = frame.shape[:2]
    y_bot, y_top = h, int(Y_TOP * h)

    if L is not None and R is not None:
        # Shade the lane area between the two lines in green.
        poly = np.array([[
            (int(x_at(L, y_bot)), y_bot), (int(x_at(L, y_top)), y_top),
            (int(x_at(R, y_top)), y_top), (int(x_at(R, y_bot)), y_bot),
        ]], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, poly, (0, 255, 0))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

    # Draw each detected line in red.
    for line in (L, R):
        if line is not None:
            p1 = (int(x_at(line, y_bot)), y_bot)
            p2 = (int(x_at(line, y_top)), y_top)
            cv2.line(out, p1, p2, (0, 0, 255), 8)

    # Departure warning. The camera is mounted at the car's centre, so the
    # image centre is the car position. Offset is measured at the bottom row
    # and normalised by lane width (0 = centred, +-0.5 = on a lane line).
    if L is not None and R is not None:
        xl, xr = x_at(L, y_bot), x_at(R, y_bot)
        lane_width = xr - xl
        offset = (w / 2 - (xl + xr) / 2) / lane_width
        if abs(offset) > WARN_THRESHOLD:
            text, color = "DEPARTURE WARNING", (0, 0, 255)
        else:
            text, color = "IN LANE", (0, 200, 0)
        cv2.putText(out, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        cv2.putText(out, f"offset: {offset:+.2f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        cv2.putText(out, "LANE NOT FOUND", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
    return out
