"""
Turn MediaPipe face landmarks into the numbers we actually store.

TWO BUGS FROM THE OLD RECORDER ARE FIXED HERE:

1. Vertical gaze was divided by ~zero. The old code bounded the eye with
   landmarks [33, 133], which are the two *horizontal* corners -- they sit at
   almost the same height, so (max_y - min_y) was ~0 and norm_y exploded to
   millions. The eye's vertical extent needs the EYELIDS (159 upper, 145 lower).

2. The eyes were cross-wired. In MediaPipe, 468-472 is the RIGHT iris and
   473-477 is the LEFT iris; 33/133/159/145 outline the RIGHT eye and
   362/263/386/374 the LEFT. The old code paired the left iris with the right
   eye's corners, so it measured one iris against the other eye's box.
"""
import numpy as np

# right eye (as seen by the camera's owner: their right)
R_IRIS = [469, 470, 471, 472]
R_CORNER_OUT, R_CORNER_IN, R_LID_UP, R_LID_LOW = 33, 133, 159, 145
# left eye
L_IRIS = [474, 475, 476, 477]
L_CORNER_IN, L_CORNER_OUT, L_LID_UP, L_LID_LOW = 362, 263, 386, 374
# inner ends of the eyebrows -- these drop when someone frowns
BROW_R, BROW_L = 55, 285


def _pt(landmarks, i, shape):
    """Landmark i as (x, y) in pixels. MediaPipe gives 0..1, so scale by frame size."""
    return np.array([landmarks[i].x * shape[1], landmarks[i].y * shape[0]])


def _iris_center(landmarks, idx, shape):
    return np.mean([_pt(landmarks, i, shape) for i in idx], axis=0)


def _eye_norm(landmarks, shape, iris_idx, out_i, in_i, up_i, low_i):
    """Where the iris sits inside its own eye, as 0..1 across and 0..1 down."""
    iris = _iris_center(landmarks, iris_idx, shape)
    x0, x1 = _pt(landmarks, out_i, shape)[0], _pt(landmarks, in_i, shape)[0]
    y0, y1 = _pt(landmarks, up_i, shape)[1], _pt(landmarks, low_i, shape)[1]
    w, h = abs(x1 - x0), abs(y1 - y0)
    # if the eye is closed or the face is lost, refuse to answer instead of
    # dividing by a tiny number and inventing a coordinate.
    if w < 4 or h < 2:
        return None, None
    return (iris[0] - min(x0, x1)) / w, (iris[1] - min(y0, y1)) / h


def _ear(landmarks, shape, out_i, in_i, up_i, low_i):
    """Eye aspect ratio: lid gap / eye width. Drops toward 0 during a blink."""
    width = np.linalg.norm(_pt(landmarks, out_i, shape) - _pt(landmarks, in_i, shape))
    height = np.linalg.norm(_pt(landmarks, up_i, shape) - _pt(landmarks, low_i, shape))
    return float(height / width) if width > 1 else None


def extract(landmarks, shape):
    """All per-frame measurements. Returns a dict; None means 'could not measure'."""
    rx, ry = _eye_norm(landmarks, shape, R_IRIS, R_CORNER_OUT, R_CORNER_IN, R_LID_UP, R_LID_LOW)
    lx, ly = _eye_norm(landmarks, shape, L_IRIS, L_CORNER_OUT, L_CORNER_IN, L_LID_UP, L_LID_LOW)
    norm_x = None if rx is None or lx is None else (rx + lx) / 2
    norm_y = None if ry is None or ly is None else (ry + ly) / 2

    # LEAN-IN: distance between the two outer eye corners, in pixels. Nothing about
    # a face changes this except distance to the camera, so it is a clean proxy.
    eye_span = float(np.linalg.norm(_pt(landmarks, R_CORNER_OUT, shape)
                                    - _pt(landmarks, L_CORNER_OUT, shape)))

    # FROWN: brows drop toward the eyes. Divide by eye_span so leaning in doesn't
    # look like frowning -- both make raw pixel distances bigger.
    brow_gap = (np.linalg.norm(_pt(landmarks, BROW_R, shape) - _pt(landmarks, R_LID_UP, shape))
                + np.linalg.norm(_pt(landmarks, BROW_L, shape) - _pt(landmarks, L_LID_UP, shape))) / 2
    frown = float(brow_gap / eye_span) if eye_span > 1 else None

    ears = [e for e in (_ear(landmarks, shape, R_CORNER_OUT, R_CORNER_IN, R_LID_UP, R_LID_LOW),
                        _ear(landmarks, shape, L_CORNER_OUT, L_CORNER_IN, L_LID_UP, L_LID_LOW))
            if e is not None]

    return {"norm_x": norm_x, "norm_y": norm_y, "eye_span": eye_span,
            "frown": frown, "ear": float(np.mean(ears)) if ears else None}
