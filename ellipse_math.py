import imageio
import numpy as np
from skimage.draw import ellipse
import os
import cv2
import time
import math
print("Done Importing")

notebook_directory = os.getcwd()

print(f"Notebook directory: {notebook_directory}")

#was going to use float32 given that the final results are in uint8 and the former gives 7
#and float32 gives 7 decimal places of accuracy but the critical use to avoid cancellation error in  the following  c_coeff
"""
    VERY NUMERICALLY CRAZY BAD
        theta = np.deg2rad(angle_deg%360)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        a_coeff = (cos_t / semi_a) ** 2 + (sin_t / semi_b) ** 2
        b_coeff = (sin_t / semi_a) ** 2 + (cos_t / semi_b) ** 2
        c_coeff = 2 * sin_t * cos_t * (1/semi_a**2 - 1/semi_b**2)

"""

    #diff  = (b_hat - a_hat) * (b_hat + a_hat)
    #denom = (a_hat * a_hat) * (b_hat * b_hat)
    #delta  = diff / denom if abs(diff) > _DOUBLE_TINY else 0.0

# the product underflows for the diff and denom  underflow to float32 subnormal
# also reciprical- squares can exceed normal range
# rounding noise may accumulate past 7 decimal places
_DOUBLE_EPS  = np.finfo(np.float64).eps     # ≈ 2.22e‑16
_DOUBLE_TINY = np.finfo(np.float64).tiny    # ≈ 2.23e‑308



def ellipse_params_to_general_form(center_x: float,
                                   center_y: float,
                                   semi_a: float,
                                   semi_b: float,
                                   angle_deg: float,
                                   *,
                                   circle_rel_eps: float | None = None):
    """
    Parameters
    ----------
    - center_x, center_y : float
        - Ellipse centre (h, k).
    - semi_a, semi_b : float
        - Semi‑major / semi‑minor axes *in pixels* (must be > 0).
    - angle_deg : float
        - Rotation angle θ in degrees.
    - circle_rel_eps : float or None, optional
        - Relative tolerance for deciding the ellipse is a circle.  If *None*
        (default), we use 2*machine ε(≈4.4e-16).  A value of, say,  1e‑8 duplicates the common graphics‑library threshold.
    """
    if semi_a <= 0 or semi_b <= 0:
        raise ValueError("semi_a and semi_b must be positive")


    semi_a   = np.float64(semi_a)
    semi_b   = np.float64(semi_b)
    center_x = np.float64(center_x)
    center_y = np.float64(center_y)

    #rel_diff threshold
    rel_eps = (2.0 * _DOUBLE_EPS) if circle_rel_eps is None else np.float64(circle_rel_eps)

    #circle shortcut – if |a - b| <= rel_eps * max(a, b), treat as circle.
    if abs(semi_a - semi_b) <= rel_eps * max(semi_a, semi_b):
        r      = np.float64(0.5) * (semi_a + semi_b)
        inv_r2 = np.float64(1.0) / (r * r)
        return {
            "a": inv_r2,  # A
            "b": inv_r2,  # C (equal for a circle)
            "c": np.float64(0.0),  # B
            "d": np.float64(1.0),
            "h": center_x,
            "k": center_y,
        }

    scale  = np.float64(max(semi_a, semi_b))
    a_hat, b_hat = semi_a / scale, semi_b / scale   #to bound 0 < a_hat, b_hat <= 1

    #b_hat or a_hat just means "normalized and i could be 1 or smaller"
    inv_a2 = np.float64(1.0) / (a_hat * a_hat)
    inv_b2 = np.float64(1.0) / (b_hat * b_hat)

    #c_coeff = 2 * sin_t * cos_t * (1/semi_a**2 - 1/semi_b**2)
    #init used this one but there is cancellation error in the parentheses so i made
    # delta, inv1 and inv2 for numerical stabilityinv_a2 = 1.0 / (a*a)
    #delta used later

    theta = np.deg2rad(angle_deg % 360)
    sin_t, cos_t = math.sin(theta), math.cos(theta)

    theta  = math.radians(angle_deg % 360.0)
    sin_t  = math.sin(theta)
    cos_t  = math.cos(theta)

    # ------------------------------------------------------------------
    # 3Stable product for Δ = 1/a² − 1/b²  (avoids catastrophic cancellation)
    #     Δ = (b_hat − a_hat)(b_hat + a_hat) / (a_hat² b_hat²)
    #     Underflow safeguard: if Δ under‑flows to zero at float64, we can
    #     safely snap B to zero – numerically, the ellipse is essentially
    #     axis‑aligned.
    diff   = (b_hat - a_hat) * (b_hat + a_hat)
    denom  = (a_hat * a_hat) * (b_hat * b_hat)
    delta  = diff / denom if abs(diff) > _DOUBLE_TINY else 0.0
    1/semi_a**2 - 1/semi_b**2

    # coeffs - minimize rounding with fma
    sc = math.fma(sin_t, cos_t, 0.0) if hasattr(math, "fma") else sin_t * cos_t
    B  = 2.0 * sc * delta

    A  = (cos_t * cos_t) * inv_a2 + (sin_t * sin_t) * inv_b2
    C  = (sin_t * sin_t) * inv_a2 + (cos_t * cos_t) * inv_b2

    #undo rescaling - coeff for original pixel units
    scale2 = scale * scale
    A     /= scale2
    B     /= scale2
    C     /= scale2

    return {
        "a": A,
        "b": C,
        "c": B,
        "d": np.float64(1.0),
        "h": center_x,
        "k": center_y,
    }

def generate_uint8_labels(w: int, h: int, cells_data: dict,
                          *, use_vectorized: bool = True) -> np.ndarray:
    """
    Parameters:

    - w, h : int
        - Output image width and height **in pixels**.
    - cells_data : dict
        Expected structure::

            {
                "indices":      [1, 2, ...],              # uint8 IDs (1‑255)
                "fluorescence": [0.42, 0.17, ...],        # ignored here
                "shape":        [(a1, b1), (a2, b2), ...],
                "location":     [(x1, y1), (x2, y2), ...],
                "rotation":     [θ1, θ2, ...]             # degrees
            }

        - ``'indices'`` must contain unique integers ≤ 255.  All lists must be
        the same length.
    - use_vectorized : bool, default ``True``
        If ``True`` masks are generated with :func:`create_ellipse_mask_vectorized`.
        If ``False`` the pixel‑loop variant is used (slower, but easier to
        debug).


    Returns:
    - uint8_labels: np.ndarray of shape (h, w) with dtype=np.uint8
        -------
    uint8_labels : np.ndarray, shape ``(h, w)``, dtype ``np.uint8``
        Background pixels hold 0; each ellipse interior is filled with its
        corresponding ID from ``cells_data['indices']``.

    - Notes
        - All heavy arithmetic is carried out in **float64** via
          :func:`ellipse_params_to_general_form`; conversion to ``uint8`` happens
          only at the final assignment step, ensuring numerical robustness.
        - Raises ``ValueError`` if a cell ID exceeds 255.

    """
    uint8_labels = np.zeros((h, w), dtype=np.uint8)

    indices    = cells_data["indices"]
    shapes     = cells_data["shape"]       # list of (semi_a, semi_b)
    locations  = cells_data["location"]    # list of (x, y)
    rotations  = cells_data["rotation"]    # list of theta in degrees

    print(f"Generating uint8 labels for {len(indices)} cells...")
    print(f"Output array shape: {uint8_labels.shape}, dtype: {uint8_labels.dtype}")
    print(f"Using {'vectorized' if use_vectorized else 'mathematical'} ellipse generation")

    #mask generation method
    mask_fn = create_ellipse_mask_vectorized if use_vectorized else create_ellipse_mask_mathematical


    for cell_id, (semi_a, semi_b), (center_x, center_y), angle_rot_numbers in zip(indices, shapes, locations, rotations):
        if cell_id > 255:
            raise ValueError(f"Cell ID {cell_id} exceeds uint8 range 0‑255")

        coeffs   = ellipse_params_to_general_form(center_x, center_y, semi_a, semi_b, angle_rot_numbers)
        cell_mask = mask_fn(w, h, coeffs)
        uint8_labels[cell_mask] = cell_id

        pixel_count = np.sum(cell_mask)
        print(f"  Cell {cell_id}: center=({center_x},{center_y}), shape=({semi_a},{semi_b}), "
              f"rotation={angle_rot_numbers} number, pixels={pixel_count}")

    unique_labels = np.unique(uint8_labels)
    print(f"\nUint8 label summary:")
    print(f"  Unique values: {unique_labels}")
    print(f"  Background pixels (0): {np.sum(uint8_labels == 0)}")
    print(f"  Total labeled pixels: {np.sum(uint8_labels > 0)}")

    return uint8_labels


def _implicit_value(A: float, C: float, B: float, dx, dy):
    #compute A*dx² + C*dy² + B*dx*dy element‑wise (broadcast friendly)
    return A * dx * dx + C * dy * dy + B * dx * dy

def create_ellipse_mask_mathematical(w: int, h: int, coeffs: dict):
    """Pixel‑by‑pixel mask using the implicit form."""
    A = coeffs["a"]; C = coeffs["b"]; B = coeffs["c"]
    h0 = coeffs["h"]; k0 = coeffs["k"]

    mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        dy = y - k0
        for x in range(w):
            dx = x - h0
            if _implicit_value(A, C, B, dx, dy) <= 1.0:
                mask[y, x] = True
    return mask



def create_ellipse_mask_vectorized(w: int, h: int, coeffs: dict):
    """Vectorised Boolean mask via broadcasting."""
    A = coeffs["a"]; C = coeffs["b"]; B = coeffs["c"]
    h0 = coeffs["h"]; k0 = coeffs["k"]

    y_grid, x_grid = np.ogrid[:h, :w]
    dx = x_grid - h0
    dy = y_grid - k0
    return _implicit_value(A, C, B, dx, dy) <= 1.0


def generate_uint8_labels(w: int, h: int, cells_data: dict,
                          *, use_vectorized: bool = True) -> np.ndarray:
    """Generate a uint8 label image for a collection of elliptical cells."""

    uint8_labels = np.zeros((h, w), dtype=np.uint8)

    indices    = cells_data["indices"]
    shapes     = cells_data["shape"]       # list of (semi_a, semi_b)
    locations  = cells_data["location"]    # list of (x, y)
    rotations  = cells_data["rotation"]    # list of θ in degrees

    mask_fn = create_ellipse_mask_vectorized if use_vectorized else create_ellipse_mask_mathematical

    for cell_id, (semi_a, semi_b), (cx, cy), angle in zip(indices, shapes, locations, rotations):
        if cell_id > 255:
            raise ValueError(f"Cell ID {cell_id} exceeds uint8 range 0‑255")

        coeffs   = ellipse_params_to_general_form(cx, cy, semi_a, semi_b, angle)
        cell_msk = mask_fn(w, h, coeffs)
        uint8_labels[cell_msk] = cell_id

    return uint8_labels





