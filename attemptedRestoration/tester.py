import imageio
import time
import imageio.v2 as imageio  #imageio‑v3 friendly import
import numpy as np
from    noise   import  pnoise2
from skimage.draw import ellipse
import os
import cv2

import math
import ellipseExamplesIo
import ellipseMath

print("Done Importing")



notebook_directory = os.getcwd()


def create_ellipse_mask_vectorized_perturbed(w: int, h: int, coeffs: dict,
                                             jitter: np.float32 = 0.07,
                                             noise_scale: int = 64,
                                             seed: int | None = None):
    #r(θ) = ab / sqrt((b cosθ)^2 + (a sinθ)^2)
    rng = np.random.default_rng(seed)#       Recover    pixel    semi‑axes    from implicit coefficients
    a = coeffs["a"]
    b = coeffs["b"]
    center_x, center_y = coeffs["k"], coeffs["l"]
    #h is height and w is width so coordinate grids


    eps =   np.finfo(np.float32).eps
    scale   =   max(a,b)
    a32_hat,    b32_hat =   a/scale,    b/scale
    y, x = np.ogrid[:h, :w]
    dx, dy = x - center_x, y - center_y
    r_px = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    # Ideal ellipse radius for every pixel direction
    #r_ideal = (a * b) / np.sqrt((b * np.cos(theta)) ** 2 +
    #                            (a * np.sin(theta)) ** 2)

    #fix with
    denom = np.hypot(b32_hat*np.cos(theta), a32_hat*np.sin(theta))
    r_ideal =   (a32_hat * b32_hat) /   denom*scale

    # 2‑D Perlin noise field ∈ [‑1,1]
    n = np.vectorize(lambda yy, xx:
                     pnoise2(xx / noise_scale, yy / noise_scale,
                             repeatx=w, repeaty=h, base=seed or 0))
    delta = jitter * n(*np.indices((h, w)))

    return r_px <= r_ideal * (1.0 + delta) #to keep label pixels lying on the actual boundary
"""
mask_bool = create_ellipse_mask_vectorized_perturbed(
                128, 256, coeffs,
                roughness=0.1,      # set 0 for perfect ellipse
                jitter=0.15,         # relative edge amplitude
                noise_scale=28,      # bigger → smoother bumps
                seed=42)             # reproducible masks
labels[mask_bool] = cell_id"""