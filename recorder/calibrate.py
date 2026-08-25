"""
Calibration: the missing step between 'where the iris sits in the eye' (norm_x/y)
and 'which pixel on the screen' (gaze_x/y).

The old recorder assumed screen_x = norm_x * SCREEN_W -- as if an iris at the far
left corner of the eye means looking at the far left of the monitor. It doesn't:
eyes rotate only a few degrees to cross a screen, heads sit off-centre, and every
face is shaped differently. That assumption is why 95% of your recorded frames
landed off-screen.

Instead: show 9 dots, one at a time. The user looks at each, we record norm_x/y,
and we fit a small polynomial mapping norm -> pixels FOR THIS USER, THIS SITTING.
The mapping is stored with the session, because it is only valid for that sitting.
"""
import numpy as np


def design_matrix(nx, ny):
    """Quadratic terms: enough to bend the mapping, few enough to fit from 9 dots."""
    nx, ny = np.asarray(nx, float), np.asarray(ny, float)
    return np.column_stack([np.ones_like(nx), nx, ny, nx * ny, nx**2, ny**2])


def fit(norm_xy, screen_xy):
    """Least squares from observed iris positions to known dot positions.

    Returns (mapping_dict, mean_pixel_error). The error is your honesty number:
    if it is 120 px, word-level gaze is fiction and you should trust line-level only.
    """
    A = design_matrix(*np.asarray(norm_xy, float).T)
    S = np.asarray(screen_xy, float)
    coef, *_ = np.linalg.lstsq(A, S, rcond=None)
    err = float(np.linalg.norm(A @ coef - S, axis=1).mean())
    return {"kind": "poly2", "coef": coef.tolist()}, err


def apply(mapping, nx, ny):
    """norm -> screen pixels using a fitted mapping."""
    coef = np.asarray(mapping["coef"], float)
    return design_matrix([nx], [ny]) @ coef


def grid_points(screen_w, screen_h, margin=0.12):
    """The 9 dot positions: a 3x3 grid inset from the edges."""
    xs = [margin, 0.5, 1 - margin]
    return [(int(x * screen_w), int(y * screen_h)) for y in xs for x in xs]
