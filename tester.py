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
from scipy import ndimage
from perlin_numpy import generate_perlin_noise_2d
from ellipseMath import _fit_periods



from ellipseExamplesIo import visualize_uint8_labels
import matplotlib.pyplot as plt, numpy as np





import time, matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import ndimage                                     #PSF blur
from perlin_numpy import generate_perlin_noise_2d             #Perlin

import ellipseMath, ellipseExamplesIo
from ellipseMath import     (_fit_periods,
                            ellipse_params_to_general_form,
                            create_ellipse_mask_vectorized_perturbed2,
                             center_offset)
from ellipseExamplesIo import   (visualize_uint8_labels,
                                save_uint8_labels,
                                n_spaced_lab,
                                order_for_gradient,
                                gradient_palette,
                                colormap_for_cells,
                                palette_to_strip,
                                colormap_for_cells,
                                canvas_slicer)


_FALSE_YELLOW   =   tuple(int("#CFCF00".lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

notebook_directory = os.getcwd()

import numpy as np
from numpy.random import default_rng
rng_global = default_rng()

#-------------------------------------------------------------------------
# Helper 1 ─ choose where & how the bud attaches
#-------------------------------------------------------------------------
def add_bud_random_rotation(parent_center, parent_axes, *,
                             bud_ratio: float = 0.6,
                             bud_offset: float = 0.15,
                             rng=rng_global):
    """
    Return (bud_center_yx, bud_axes, bud_rot_deg).

    The bud centre is placed tangentially at a random angle on the mother
    ellipse, then shifted *outward* by `bud_offset * max(a_p,b_p)`.
    Bud semi‑axes = `bud_ratio * parent_axes`.
    Rotation ∈ [0°,180°) to break alignment symmetry.

    Notes
    -----
    • `bud_ratio`  ≈ 0.5–0.7 matches measured daughter : mother volume
      ratios 0.5‑0.7 in *S. cerevisiae*:.
    • `bud_offset` ≥ `bud_ratio·a_p/max` guarantees the bud does **not**
      cross the mother’s far side.
    """
    center_y, center_x   = parent_center
    a_p, b_p = parent_axes

    #random attachment angle on mother perimeter
    φ = rng.uniform(0.0, 2*np.pi)

    y_bound = center_y + a_p * np.sin(φ)
    x_bound = center_x + b_p * np.cos(φ)

    #shift bud centre slightly outward
    shift   = bud_offset * max(a_p, b_p)
    bud_center_y  = y_bound + shift * np.sin(φ)
    bud_center_x  = x_bound + shift * np.cos(φ)

    #scale axes & random own rotation
    a_bud, b_bud = a_p * bud_ratio, b_p * bud_ratio
    bud_rot_deg  = rng.uniform(0.0, 180.0)

    return (bud_center_y, bud_center_x), (a_bud, b_bud), bud_rot_deg






def ellipse_mask_rot_jitter(h, w, center, axes, angle_deg: float,
                            *, jitter=0.05, noise_scale=64,
                            seed=None, repeat=True):
    """
    Boolean mask of a rotated ellipse with optional Perlin boundary jitter.

    Parameters
    ----------
    angle_deg : float
        CCW rotation of the bud ellipse.
    jitter : float
        Relative amplitude of Perlin noise perturbation (0 → no wobble).
    noise_scale : int
        Larger -> coarser bumps (Perlin period ≈ `noise_scale` px).
    """
    yy, xx = np.indices((h, w), dtype=np.float32)
    center_y, center_x = center
    a,  b  = axes

    #convert world offsets -> body frame (x',y')  (StackOverflow rotation)
    dy, dx = yy - center_y, xx - center_x
    φ      = np.deg2rad(angle_deg, dtype=np.float32)
    cosφ, sinφ = np.cos(φ), np.sin(φ)
    x_p = dx * cosφ + dy * sinφ
    y_p = -dx * sinφ + dy * cosφ

    #Per‑pixel jitter of axes via Perlin field
    if jitter:

        res_y = _fit_periods(h, noise_scale)  #guarantees h % res_y == 0
        res_x = _fit_periods(w, noise_scale)
        noise = generate_perlin_noise_2d((h, w), (res_y, res_x),
                                         tileable=(repeat, repeat)).astype(np.float32)
        a_eff = a * (1 + jitter * noise)
        b_eff = b * (1 + jitter * noise)
    else:
        a_eff, b_eff = a, b

    return (x_p / a_eff)**2 + (y_p / b_eff)**2 <= 1.0




#small injector: optionally spawn a bud for each parent cell
def _maybe_add_bud(parent_mask, parent_center, parent_axes, *,
                   rng, prob=0.4):
    """
    With probability `prob`, generate a bud mask and merge *outside*
    the parent.  Returns merged_mask.
    """
    if rng.random() > prob:
        return parent_mask  #leave as‑is

    bud_center, bud_axes, bud_rot = add_bud_random_rotation(
        parent_center, parent_axes, rng=rng
    )
    h, w = parent_mask.shape
    bud_mask = ellipse_mask_rot_jitter(h, w, bud_center, bud_axes,
                                       bud_rot, jitter=0.05,
                                       noise_scale=96, seed=rng.integers(1<<31))
    #keep only protruding cap
    bud_only = bud_mask & ~parent_mask
    return parent_mask | bud_only

#hook into generate_uint8_labels
def generate_uint8_labels_with_buds(w: int, h: int, cells_data: dict,
                                    *, rng=None, bud_prob=0.4):
    """
    As original but inserts buds on a subset of mothers.
    """
    rng = rng or np.random.default_rng()
    canvas_shape = (2048, 2048)
    uint8_labels = np.zeros(canvas_shape, dtype=np.uint8)
    row_off, col_off = center_offset(canvas_shape, (h, w))

    #loop over parent cells
    for cell_id, (a, b), (center_x, center_y), angle in zip(cells_data["indices"],
                                               cells_data["shape"],
                                               cells_data["location"],
                                               cells_data["rotation"]):
        coeffs = ellipse_params_to_general_form(center_x, center_y, a, b, angle)
        parent = create_ellipse_mask_vectorized_perturbed2(
            w, h, coeffs, 0.07, 64, (row_off, col_off)
        )
        parent = _maybe_add_bud(parent, (center_y+row_off, col_off+center_x), (a, b),
                                rng=rng, prob=bud_prob)
        uint8_labels[parent] = cell_id
    return uint8_labels
"""
mask_bool = create_ellipse_mask_vectorized_perturbed(
                128, 256, coeffs,
                roughness=0.1,      #set 0 for perfect ellipse
                jitter=0.15,         #relative edge amplitude
                noise_scale=28,      #bigger → smoother bumps
                seed=42)             #reproducible masks
labels[mask_bool] = cell_id"""


def render_fluor_image(label_map: np.ndarray,*, sigma_px=1.2,
                       bitdepth=16, gamma=0.7, rng=None):
    """Return uint16 synthetic fluorescence image I₀."""
    rng = rng or np.random.default_rng()
    img_f = np.zeros_like(label_map, dtype=np.float32)

    # per-cell gradient + nucleolus
    for cid in np.unique(label_map):
        if cid == 0:          #background
            continue
        mask = label_map == cid

        #cell centre & max radius
        yy, xx = np.where(mask)
        center_y, center_x = np.median(yy), np.median(xx)
        r_max  = np.sqrt(((yy - center_y)**2 + (xx - center_x)**2).max())

        #radial profile
        rr = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        F0 = rng.uniform(0.6, 1.0)
        img_f[yy, xx] = F0 * (1. - (rr / (r_max + 1e-6))**gamma)

        #--- dark nucleolus ---
        nuc_axes = (0.3 * r_max, 0.3 * r_max)
        nuc_rot  = rng.uniform(0, 180.)
        nuc_mask = ellipse_mask_rot_jitter(*mask.shape,
                                           center=(center_y, center_x),
                                           axes=nuc_axes,
                                           angle_deg=nuc_rot,
                                           jitter=0.03,
                                           noise_scale=64,
                                           repeat=True,
                                           seed=rng)
        img_f[nuc_mask] *= rng.uniform(0.4, 0.6)

    #---- optics + noise ----
    img_f = ndimage.gaussian_filter(img_f, sigma_px)
    #scale to photo-electron counts
    counts = img_f * 5000.0
    counts = rng.poisson(counts)
    counts = counts + rng.normal(0, 50, counts.shape)  #read noise
    counts = np.clip(counts, 0, 2**bitdepth - 1).astype(np.uint16)
    cv2.imshow('uint16 Image', counts)
    return counts.astype(np.uint16)


rng = default_rng(42)

#parameters
H = W = 2048
SIGMA_PSF = 1.2           #px  (optics blur)
BITDEPTH  = 16
GAMMA     = 0.7
BUD_PROB  = 1.0           #always add a bud for demo
NUCL_RATIO = 0.3          #inner dark ellipse as % of mother radius

def generate_uint8_labels_with_buds(w, h, cells_data, *, rng, bud_prob=0.4):
    canvas_shape = (H, W)
    labels = np.zeros(canvas_shape, dtype=np.uint8)
    row_off, col_off = center_offset(canvas_shape, (h, w))

    for cid, (a, b), (center_x, center_y), angle in zip(cells_data["indices"],
                                            cells_data["shape"],
                                            cells_data["location"],
                                            cells_data["rotation"]):
        coeffs = ellipse_params_to_general_form(center_x, center_y, a, b, angle)
        parent = create_ellipse_mask_vectorized_perturbed2(
            w, h, coeffs, 0.7, 39, (row_off, col_off))
        parent = _maybe_add_bud(parent, (center_y, center_x), (a, b),
                                rng=rng, prob=bud_prob)
        labels[parent] = cid
    return labels

def _maybe_add_bud(parent_mask, parent_center, parent_axes, *, rng, prob):
    if rng.random() > prob:
        return parent_mask
    b_cen, b_axes, b_rot = add_bud_random_rotation(parent_center,
                                                   parent_axes, rng=rng)
    bud = ellipse_mask_rot_jitter(*parent_mask.shape, b_cen, b_axes, b_rot,
                                  jitter=0.05, noise_scale=96,
                                  seed=rng.integers(1<<31))
    return parent_mask | (bud & ~parent_mask)

def crop_box(img,dimensions,offset):
    """    Crop a TIFF to a box    """
    h_img, w_img,box_h,box_w,row_off,col_off \
        = 2048, 2048,  dimensions[0], dimensions[1], offset[0], offset[1]
    cropped = img[row_off : row_off + box_h, col_off : col_off + box_w]

    # Preserve original metadata & bit depth
    return cropped
def false_yellow_overlay(I16: np.ndarray, *, bitdepth=16,   gain=4.0) -> np.ndarray:
    """
    map uint16 fluorescence frame to false-yellow RGB (0xCFCF00).
    black background stays black
    """
    I_norm = I16.astype(np.float32) / (2 ** bitdepth - 1)  #0–1
    I_norm = np.clip(I_norm * gain, 0, 1)  #boost & clip
    rgb_y = np.array([207, 207, 0], dtype=np.float32) / 255.
    return (I_norm[..., None] * rgb_y).astype(np.uint8)

if __name__ == "__main__":
    t0 = time.time()
    w,h = 128,200
    toy = {
        "indices": list(range(1, 10)),
        "fluorescence": [100, 120, 80, 150, 90, 110, 130, 140, 95],
        "size": [15, 18, 14, 20, 16, 17, 19, 18, 15],
        "shape": [(8, 19), (10, 13), (7, 6), (11, 7), (13, 8),
                  (9, 9), (11, 8), (45, 9), (18, 7)],
        "location": [(30, 25), (30, 50), (40, 80), (60, 30),
                     (85, 70), (110, 223), (110, 85), (110, 110), (80, 110)],
        "rotation": [0, 15, -20, 30, 0, 45, -10, 0, 25],
    }

    mask                    = generate_uint8_labels_with_buds(w, h, toy, rng=rng,
                                           bud_prob=BUD_PROB)
    row_offset,col_offset   = center_offset((2048, 2048), (h, w))
    centered_start          =   (row_offset, col_offset)
    I0                      = render_fluor_image(mask, sigma_px=SIGMA_PSF,
                              bitdepth=BITDEPTH, gamma=GAMMA, rng=rng)
    cropped_uint8_labels    = canvas_slicer(mask, (h, w), (row_offset, col_offset))
    cropped_IO              =  crop_box(I0,(h,w),centered_start)

    #save uncropped and uncropped masks

    IO_metadata         =    save_uint8_labels(mask, (h,w), centered_start,"dump0\mask_tester_mask")
    cropped_IO_metadata =   save_uint8_labels(cropped_uint8_labels, (h,w), centered_start,"dump0\cropped_tester_mask")

    #save raw uint16 + false-yellow PNG
    imageio.imwrite("demo_fluor.tiff", cropped_IO)                     # training

    fluors = np.array(cropped_IO_metadata["fluorescence"], dtype=np.float32)
    min_adu, max_adu = fluors.min(), fluors.max()

    # define a stretch & false‑yellow function in this scope:
    def false_yellow_ADU(I16, *, min_adu, max_adu, gamma=0.7):
        """
        Linearly stretch I16 from [min_adu..max_adu] → [0..1],
        apply optional gamma, then false‑color into 0xCFCF00.
        """
        I = I16.astype(np.float32)
        I_norm = (I - min_adu) / (max_adu - min_adu)
        I_norm = np.clip(I_norm, 0, 1)
        # optional gamma‑correction (brighten central cores)
        I_norm = I_norm ** (1 / gamma) if gamma != 1.0 else I_norm



    vis_yellow = (false_yellow_overlay(cropped_IO) * 255).astype(np.uint8)  # display
    imageio.imwrite("vis_yellow.png", vis_yellow)



    vis_rgb = visualize_uint8_labels(cropped_uint8_labels,cropped_IO_metadata,None)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(vis_rgb) ;          ax[0].set_title("Label RGB") ; ax[0].axis("off")
    ax[1].imshow(vis_yellow)   ;          ax[1].set_title("False-Yellow I₀") ; ax[1].axis("off")
    plt.tight_layout() ; plt.show()

    print(f"Saved demo_fluor.tiff (uint16) and vis_yellow.png – "
          f"run time {time.time() - t0:.2f}s")




    plt.imshow(vis_rgb); plt.axis("off")