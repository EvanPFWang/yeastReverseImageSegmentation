import imageio
import time
import imageio.v2 as imageio  #imageio‑v3 friendly import
import numpy as np
from typing import Tuple, Dict, Any

from skimage.draw import ellipse
from noise  import  pnoise2
import os
import cv2

import math
print("Done Importing")



notebook_directory = os.getcwd()

print(f"Notebook directory: {notebook_directory}")

#was going to use float32 given that the final results are in uint8 and the former gives 7
#and float32 gives 7 decimal places of accuracenter_y but the critical use to avoid cancellation error
# in  the following  c_coeff since semi_a**2 would lose much info after being rounded
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

#the product underflows for the diff and denom  underflow to float32 subnormal
#also reciprical- squares can exceed normal range
#rounding noise may accumulate past 7 decimal places
_DOUBLE_EPS  = np.finfo(np.float64).eps     #≈ 2.22e‑16
_DOUBLE_TINY = np.finfo(np.float64).tiny    #≈ 2.23e‑308
_SINGLE_EPS = np.finfo(np.float32).eps
_SINGLE_TINY = np.finfo(np.float32).tiny


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

    #
    #
    semi_a   = np.float32(semi_a)
    semi_b   = np.float32(semi_b)
    center_x = np.float32(center_x)
    center_y = np.float32(center_y)

    #rel_diff threshold
    rel_eps = (2.0 * _SINGLE_EPS) if circle_rel_eps is None else np.float32(circle_rel_eps)

    #circle shortcut – if |a - b| <= rel_eps * max(a, b), treat as circle.
    if abs(semi_a - semi_b) <= rel_eps**0.8 * max(semi_a, semi_b):  #keeps 1024 pix drift ≲2.9e-6
        r      = np.float32(0.5) * (semi_a + semi_b)
        inv_r2 = np.float32(1.0) / (r * r)
        return {
            "semi_a":r,
            "semi_b":r,
            "A": inv_r2,  #A
            "B": inv_r2,  #C (equal for a circle)
            "C": np.float32(0.0),  #B
            "k": center_x,
            "l": center_y,
        }

    scale  = np.float64(max(semi_a, semi_b))
    a_hat, b_hat = np.float64(semi_a) / scale, np.float64(semi_b) / scale   #to bound 0 < a_hat, b_hat <= 1

    #b_hat or a_hat just means "normalized and i could be 1 or smaller"
    inv_a2 = np.float64(1.0) / (a_hat * a_hat)
    inv_b2 = np.float64(1.0) / (b_hat * b_hat)

    #c_coeff = 2 * sin_t * cos_t * (1/semi_a**2 - 1/semi_b**2)
    #init used this one but there is cancellation error in the parentheses so i made
    #delta, inv1 and inv2 for numerical stabilityinv_a2 = 1.0 / (a*a)
    #delta used later

    theta = np.deg2rad(angle_deg % 360.0)
    sin_t, cos_t = math.sin(theta), math.cos(theta)

    #------------------------------------------------------------------
    #Stable product for Δ = 1/a² − 1/b²  (avoids catastrophic cancellation)
    #   Δ = (b_hat − a_hat)(b_hat + a_hat) / (a_hat² b_hat²)
    #   Underflow safeguard: if Δ under‑flows to zero at float64, we can
    #   safely snap B to zero – numerically, the ellipse is essentially
    #   axis‑aligned.
    diff   = (b_hat - a_hat) * (b_hat + a_hat)
    denom  = (a_hat * a_hat) * (b_hat * b_hat)
    delta  = diff / denom if abs(diff) > _DOUBLE_TINY else 0.0
    #   1/semi_a**2 - 1/semi_b**2

    #coeffs - minimize rounding with fma
    sc = math.fma(sin_t, cos_t, 0.0) if hasattr(math, "fma") else sin_t * cos_t

    C  = 2.0 * sc * delta

    A  = (cos_t * cos_t) * inv_a2 + (sin_t * sin_t) * inv_b2
    B  = (sin_t * sin_t) * inv_a2 + (cos_t * cos_t) * inv_b2

    #undo rescaling - coeff for original pixel units
    scale2 = scale * scale
    A     /= scale2
    B     /= scale2
    C     /= scale2

    return {
        "semi_a": np.float32(semi_a),   #semi-axis lengths since having to recalculate these in my mask
                                        #masker causes immersion error from sqrt() truncation
        "semi_b": np.float32(semi_b),
        "A": A,
        "C": C,
        "B": B,
        "k": center_x,
        "l": center_y,
    }

#def generate_uint8_labels(w: int, h: int, cells_data: dict,
# *, use_vectorized: bool = True) -> np.ndarray:
"""
Parameters:

- w, h : int
    - Output image width and height **in pixels**.
- cells_data : dict
    Expected structure::

        {
            "indices":      [1, 2, ...],              #uint8 IDs (1‑255)
            "fluorescence": [0.42, 0.17, ...],        #ignored here
            "shape":        [(a1, b1), (a2, b2), ...],
            "location":     [(x1, y1), (x2, y2), ...],
            "rotation":     [θ1, θ2, ...]             #degrees
        }

    - ``'indices'`` must contain unique integers <=  255.  All lists must be
    the same length.

Returns:
- uint8_labels: np.ndarray of shape (h, w) with dtype=np.uint8
    -------
uint8_labels : np.ndarray, shape ``(h, w)``, dtype ``np.uint8``
    Background pixels hold 0; each ellipse interior is filled with its
    corresponding ID from ``cells_data['indices']``.

- Notes
    - All heavy arithmetic is carried out in **float64** via
      :func:`ellipse_params_to_general_form`; conversion to ``uint8`` happens
      only at the final assignment step, ensuring numerical robustness.
    - Raises ``ValueError`` if a cell ID exceeds 255.

"""



def _implicit_value(A: float, B: float, C: float, dx, dy):
    #compute A*dx² + C*dy² + B*dx*dy element‑wise (broadcast friendly)
    return A * dx * dx + B * dy * dy + C * dx * dy #<=D but D is assumed to be 1
def center_offset(canvas_shape: Tuple, raster_shape:    Tuple):
    """
    Return offset to place the raster centre at the centre of the canvas.

    Parameters
    ----------
    canvas_shape : tuple[int, int]
        (height, width) of the big array, e.g. (2048, 2048)
    raster_shape : tuple[int, int]
        (height, width) of the smaller raster

    Returns
    -------
    row_off, col_off : int
        Offset to the top‑left corner where the raster should be pasted
        so its centre aligns with the canvas centre.
    """
    ch, cw = canvas_shape
    rh, rw = raster_shape

    row_off = (ch - rh) // 2
    col_off = (cw - rw) // 2
    return row_off, col_off


def create_ellipse_mask_vectorized_perturbed(w: int, h: int, coeffs: dict,
                                             jitter: np.float32 = 0.07,
                                             noise_scale: int = 64,
                                             offset:    tuple[int,int]=(0,0),
                                             seed: int | None = None):

    """
    As `create_ellipse_mask_vectorized`, but boundary is perturbed by Perlin
    noise.  `jitter` ≈ relative amplitude; `noise_scale` controls feature size.
    """
    #2048x2048
    #r(θ) = ab / sqrt((b cosθ)^2 + (a sinθ)^2)
#    rng = np.random.default_rng(seed)#       Recover    pixel    semi‑axes
    semi_a,  semi_b = coeffs["semi_a"], coeffs["semi_b"]
    xSqr_coeff  =   coeffs["A"]
    ySqr_coeff = coeffs["B"]


    #offset_row, offset_col = offset[0], offset[1]
    center_x, center_y = coeffs["k"]+row_offset, col_offset+coeffs["l"]
    #h is height and w is width so coordinate grids


    eps =   np.finfo(np.float32).eps
    scale   =   max(semi_a,semi_b)
    a32_hat,    b32_hat =   semi_a/scale,    semi_b/scale
    y_grid, x_grid = np.ogrid[:h, :w]
    #
    
    yy, xx = np.indices((2048, 2048))

    dx, dy = xx - center_x, yy - center_y
    theta = np.arctan2(dy, dx,  dtype=np.float32)

    # Ideal ellipse radius for every pixel direction
    #r_ideal = (a * b) / np.sqrt((b * np.cos(theta)) ** 2 +
    #                            (a * np.sin(theta)) ** 2)
    #scale gets taken out so all we deal with are the hats
    #fix with
    denom = np.hypot(b32_hat*np.cos(theta), a32_hat*np.sin(theta))
    r_ideal =   (a32_hat * b32_hat) /   denom*scale
    # 2‑D Perlin noise field ∈ [‑1,1]


    noise = np.clip(np.vectorize(lambda y0, x0: pnoise2(x0/noise_scale,
                                                     y0/noise_scale,
                                                     repeatx=w, repeaty=h,
                                                     base=seed or 0),
                               otypes=[np.float32])(yy, xx),
                  -1.0, 1.0)
    delta = jitter * noise
    r_px = np.hypot(dx, dy)
    return r_px <= r_ideal * (1.0 + delta   +   eps) #to keep label pixels lying on the actual boundary
    #returns bool mask as labels for this cell over whole 2048


def generate_uint8_labels(w: int, h: int, cells_data: dict,
                          *, use_vectorized: bool = True)\
        -> tuple[np.ndarray,tuple[int,int],tuple[int,int]]:
    """Generate a uint8 label image for a collection of elliptical cells."""

    canvas_shape = (2048, 2048)
    raster_shape = (h, w)

    #uint8_labels = np.zeros((h, w), dtype=np.uint8)
    uint8_labels = np.zeros(canvas_shape, dtype=np.uint8)

    row_off, col_off = center_offset(canvas_shape, raster_shape)


    indices    = cells_data["indices"]
    shapes     = cells_data["shape"]       #list of (semi_a, semi_b)
    locations  = cells_data["location"]    #list of (x, y)
    rotations  = cells_data["rotation"]    #list of θ in degrees

    roi = uint8_labels[row_off: row_off + raster_shape[1],
          col_off: col_off + raster_shape[0]]#sincelooking for w then h

    mask_fn = create_ellipse_mask_vectorized_perturbed
    for cell_id, (semi_a, semi_b), (center_x, center_y), angle in zip(indices, shapes, locations, rotations):
        if cell_id > 255:
            raise ValueError(f"Cell ID {cell_id} exceeds uint8 range 0‑255")
        coeffs   = ellipse_params_to_general_form(center_x, center_y, semi_a, semi_b, angle)
        cell_msk = mask_fn(raster_shape[0], raster_shape[1], coeffs)[row_off: row_off + raster_shape[1],
          col_off: col_off + raster_shape[0]]
        roi[cell_msk] = cell_id
        #draw all cells on canvas
    return uint8_labels,    raster_shape,   (row_off, col_off)


def generate_uint8_labels_cv2(w: int,   h: int, cells_data: dict)\
        -> tuple[np.ndarray,tuple[int,int],tuple[int,int]]:
    """
    Alternative generator for very large batches where
    Python‑level loops become the bottleneck.
    Note that OpenCV uses the *endpoint‑inclusive* angle
    convention (startAngle, endAngle), so we pass 0‑to‑360.


    Rasterise filled ellipses directly with OpenCV and write the cell ID
    into a uint‑8 mask.

    Parameters
    - w, h : int
        - Output width × height in pixels.
    - cells_data : dict
        - Must contain the keys ``'indices'``, ``'shape'``, ``'location'``,
        - and ``'rotation'`` (same schema as the coefficient‑based pipeline).

    Returns
    - uint8_labels : np.ndarray, shape ``(k, l)``, dtype ``uint8``
            - Background pixels are 0; each ellipse interior is the
            corresponding ID from ``cells_data['indices']``.
    """
    #allocate target image (background = 0)

    canvas_h, canvas_w = 2048, 2048
    raster_h, raster_w = h, w
    canvas_shape = (canvas_h, canvas_w)
    raster_shape = (raster_h, raster_w)

    #uint8_labels = np.zeros((h, w), dtype=np.uint8)
    uint8_labels = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    row_off, col_off = center_offset(canvas_shape, raster_shape)


    indices,    shapes, locations,  rotations   = cells_data["indices"],  cells_data["shape"] ,   cells_data["location"], cells_data["rotation"]
    #roi slices window into `uint8_labels`, so cv2 draws in place
    roi = uint8_labels[row_off: row_off + raster_h,
                 col_off: col_off + raster_w]
    #draw each ellipse in‑place
    for cell_id, (a, b), (center_x, center_y), angle_rot_numbers in zip(
        indices, shapes, locations, rotations
    ):
        if cell_id > 255:
            raise ValueError(f"Cell ID {cell_id} exceeds uint8 range 0‑255")

        center = (int(round(center_x)), int(round(center_y)))   # integer pixel coords
        axes   = (int(round(a)),  int(round(b)))    # integer semi‑axes

        # thickness = –1 (cv2.FILLED) -> fill the ellipse interior
        cv2.ellipse(
            img       = roi,
            center    = center,
            axes      = axes,
            angle     = float(angle_rot_numbers),  # CCW degrees
            startAngle= 0,
            endAngle  = 360,
            color     = int(cell_id),  # write the ID value
            thickness = -1
        )

    return uint8_labels,    raster_shape,   (row_off, col_off)

if __name__ == "__main__":
    p = ellipse_params_to_general_form(10, 5, 5, 3, 30)
    print(p)
    W, H = 512, 512  #canvas size (px)
    center_x, center_y = W / 2.0, H / 2.0  #centre
    semi_a, semi_b = 140.0, 80.0  #axes lengths
    theta_deg = 30.0  #rotation (deg)

    coeffs = ellipse_params_to_general_form(center_x, center_y,
                                           semi_a, semi_b, theta_deg)

    row_offset, col_offset =   center_offset((2048, 2048), (512,512))
    mask_vec = create_ellipse_mask_vectorized_perturbed(W, H, coeffs,0.08,29, (row_offset, col_offset), max(W,H)%min(W,H))



    #console report
    print("Ellipse parameters :",
          f"centre=({center_x:.1f},{center_y:.1f})  a={semi_a}  b={semi_b}  θ={theta_deg}°")

    print("Saved mask.png")

    #write 8‑bit image
    imageio.imwrite("mask.png", (mask_vec.astype(np.uint8) * 255))