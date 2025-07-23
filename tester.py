import imageio
import time
import imageio.v2 as imageio  #imageio‑v3 friendly import
import numpy as np
from    noise   import  pnoise2
from skimage.draw import ellipse
from    scipy.signal import fftconvolve
import psfmodels    as psfm
from scipy.ndimage import distance_transform_edt,   center_of_mass, gaussian_filter, sobel
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





import time
import  matplotlib.pyplot as plt
from numpy.random import default_rng
import tifffile
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
#Helper 1 ─ choose where & how the bud attaches
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
    phi = rng.uniform(0.0, 2*np.pi)

    y_bound = center_y + a_p * np.sin(phi)
    x_bound = center_x + b_p * np.cos(phi)

    #shift bud centre slightly outward
    shift   = bud_offset * max(a_p, b_p)
    bud_center_y  = y_bound + shift * np.sin(phi)
    bud_center_x  = x_bound + shift * np.cos(phi)

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
    tiny32 = np.finfo(np.float32).tiny
    #convert world offsets -> body frame (x',y')  (StackOverflow rotation)
    dy, dx = yy - center_y, xx - center_x
    phi      = np.deg2rad(angle_deg, dtype=np.float32)
    cosphi, sinphi = np.cos(phi), np.sin(phi)
    x_p = dx * cosphi + dy * sinphi
    y_p = -dx * sinphi + dy * cosphi
    eps32 = np.finfo(np.float32).eps
    #Per‑pixel jitter of axes via Perlin field
    if jitter:

        res_y = _fit_periods(h, noise_scale)  #guarantees h % res_y == 0
        res_x = _fit_periods(w, noise_scale)
        noise = generate_perlin_noise_2d((h, w), (res_y, res_x),
                                         tileable=(repeat, repeat)).astype(np.float32)

        #more numerically stable for extra checks
        scale = math.fma(jitter, noise,1.0)  if hasattr(math, "fma") else jitter * noise+1.0
        scale = np.where(scale >= 0,
                         np.maximum(scale, eps32),  #positive branch
                         np.minimum(scale, -eps32)).astype(np.float32)  #negative branch
        a_eff = a * scale
        b_eff = b * scale

        a_eff = np.where(a_eff >= 0,
                         np.maximum(a_eff, eps32),  #positive branch
                         np.minimum(a_eff, -eps32)).astype(np.float32)  #negative branch
        b_eff = np.where(b_eff >= 0,
                     np.maximum(b_eff, eps32),  #positive branch
                     np.minimum(b_eff, -eps32)).astype(np.float32)  #negative branch

    else:
        a_eff = np.where(a >= 0,
                         np.maximum(a, eps32),  #positive branch
                         np.minimum(a, -eps32)).astype(np.float32)  #negative branch
        b_eff = np.where(b >= 0,
                         np.maximum(b, eps32),  #positive branch
                         np.minimum(b, -eps32)).astype(np.float32)  #negative branch

    inv_x = np.divide(x_p, a_eff, where=(a_eff != 0),
                      out=np.zeros_like(x_p))
    inv_y = np.divide(y_p, b_eff, where=(b_eff != 0),
                      out=np.zeros_like(y_p))
    mask = (inv_x ** 2 + inv_y ** 2) <= 1.0
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
    for cell_id, (a, b), (center_x, center_y), angle in zip(cells_data["indices"],cells_data["shape"],
                                               cells_data["location"],cells_data["rotation"]):
        coeffs = ellipse_params_to_general_form(center_x, center_y, a, b, angle)
        parent = create_ellipse_mask_vectorized_perturbed2(
            w, h, coeffs, 0.7, 39, (row_off, col_off)
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



def render_fluor_image(label_map: np.ndarray,
                       metadata: dict,
                       *,
                       # ---------- GLOBAL -------------
                       sigma_px: float = 1.0,
                       bitdepth: int   = 16,
                       gamma: float    = 1.2,
                       rng=None,
                       # ----------   RIM  -------------
                       rim_dark_amp: float      = -0.03,
                       rim_edge_sigma_px: float = 0.1,
                       rim_edge_softness: float = 0.1,
                       rim_perlin_period: int   = 0.01,
                       rim_perlin_blur_px: int  = -10,
                       # ---------- NUCLEOLUS ----------
                       nuc_core_mul: float = -9,
                       nuc_grad_amp: float = -9,#0.13
                       nuc_decay_px: float = 2,
                       # ---------- GRAD NOISE ---------
                       grad_dark_amp: float = 0,#-0.08,
                       # ---------- RANGE --------------
                       peak_fraction: float = 0.7):
    """
    Flexible yeast‑fluorescence renderer.

    Each **amp** parameter <0 only *darkens*.  Set it to 0 to disable
    the corresponding noise.
    """

    h, w = label_map.shape
    img  = np.zeros((h, w), dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # fluorescence per cell -------------------------------------------------
    # ------------------------------------------------------------------
    fluors = np.asarray(metadata["fluorescence"], dtype=np.float32)
    fluors /= fluors.max() + 1e-6

    nuc_masks = []

    for cid in np.unique(label_map):
        if cid == 0:
            continue
        mask = (label_map == cid)
        F0   = fluors[cid-1]

        # radial profile
        din  = distance_transform_edt(mask).astype(np.float32)
        yy, xx = np.nonzero(mask)
        r_max  = din[yy, xx].max()
        r_norm = din[yy, xx] / (r_max + 1e-6)
        img[yy, xx] += F0 * (1.0 - r_norm**gamma)

        # nucleolus geometry
        cy, cx = center_of_mass(mask)
        a_nuc  = rng.uniform(0.28, 0.34) * r_max
        b_nuc  = a_nuc * rng.uniform(0.65, 0.90)
        nuc_mask = ellipse_mask_rot_jitter(
            h, w, (cy, cx), (a_nuc, b_nuc),
            angle_deg=rng.uniform(0, 180.),
            jitter=0.4, noise_scale=48,
            repeat=True, seed=rng.integers(1<<31))
        img[nuc_mask] *= nuc_core_mul            # base dark
        nuc_masks.append(nuc_mask)

    # ------------------------------------------------------------------
    # rim‑darkening w/ Perlin
    # ------------------------------------------------------------------
    if rim_dark_amp != 0.0:
        dist_out = distance_transform_edt(label_map == 0).astype(np.float32)
        dist_in  = distance_transform_edt(label_map != 0).astype(np.float32)
        d_signed = dist_out
        d_signed[label_map != 0] = -dist_in[label_map != 0]

        rim_w = np.exp(-((np.abs(d_signed)-0.5)/rim_edge_sigma_px)**2)
        rim_w *= 1.0 / (1.0 + (dist_out / rim_edge_softness)**2)

        res_y = _fit_periods(h, rim_perlin_period)
        res_x = _fit_periods(w, rim_perlin_period)
        perlin = generate_perlin_noise_2d((h, w), (res_y, res_x),
                                          tileable=(False, False)
                                         ).astype(np.float32)
        perlin = gaussian_filter(perlin, rim_perlin_blur_px)
        img   *= 1.0 + rim_dark_amp * rim_w * perlin

    # ------------------------------------------------------------------
    # optics PSF
    # ------------------------------------------------------------------
    if hasattr(psfm, "make_psf"):
        psf = psfm.make_psf([0.0], nx=65, dxy=0.1, NA=1.40,
                            ns=1.33, wvl=0.52).astype(np.float32)[0]
        psf /= psf.sum()
        img  = fftconvolve(img, psf, mode='same')
    else:
        img  = gaussian_filter(img, sigma_px)

    # ------------------------------------------------------------------
    # nucleolus distance fall‑off
    # ------------------------------------------------------------------
    if nuc_grad_amp != 0.0:
        for nm in nuc_masks:
            dist_nuc = distance_transform_edt(~nm).astype(np.float32)
            decay = np.exp(-(dist_nuc / nuc_decay_px)**2)
            img[nm] += nuc_grad_amp * (1.0 - decay[nm])

    # ------------------------------------------------------------------
    # gradient‑linked dark noise
    # ------------------------------------------------------------------
    if grad_dark_amp != 0.0:
        gx = sobel(img, axis=0)
        gy = sobel(img, axis=1)
        grad = np.hypot(gx, gy)
        grad /= grad.max() + 1e-6
        dark_noise = rng.standard_normal((h, w)).astype(np.float32)
        img *= 1.0 + grad_dark_amp * grad * np.abs(dark_noise)

    # ------------------------------------------------------------------
    # normalise & sample photon statistics
    # ------------------------------------------------------------------
    img -= img.min()
    img /= img.max() / peak_fraction + 1e-6

    counts = rng.poisson(img * ((1<<bitdepth)-1)).astype(np.int32)
    counts += rng.normal(0, 50, counts.shape).astype(np.int32)
    counts  = np.clip(counts, 0, (1<<bitdepth)-1).astype(np.uint16)

    return counts
rng = default_rng(42)

#parameters
H = W = 2048
SIGMA_PSF = 1.2           #px  (optics blur)
BITDEPTH  = 16
GAMMA     = 0.7
BUD_PROB  = 1.0           #always add a bud for demo
NUCL_RATimage0 = 0.3          #inner dark ellipse as % of mother radius

def generate_uint8_labels_with_buds(w, h, cells_data, *, rng, bud_prob=0.4):
    canvas_shape = (H, W)
    labels = np.zeros(canvas_shape, dtype=np.uint8)
    row_off, col_off = center_offset(canvas_shape, (h, w))

    indices    = cells_data["indices"]
    shapes     = cells_data["shape"]       #list of (semi_a, semi_b)
    locations  = cells_data["location"]    #list of (x, y)
    rotations  = cells_data["rotation"]    #list of θ in degrees

    for cell_id, (semi_a, semi_b), (center_x, center_y), angle in zip(indices, shapes,    locations, rotations):
        if cell_id > 255:
            raise ValueError(f"Cell ID {cell_id} exceeds uint8 range 0‑255")
        coeffs = ellipse_params_to_general_form(center_x, center_y, semi_a, semi_b, angle)
        parent = create_ellipse_mask_vectorized_perturbed2(
            w, h, coeffs, 0.7, 39, (row_off, col_off))
        parent = _maybe_add_bud(parent, (center_y, center_x), (semi_a, semi_b),
                                rng=rng, prob=bud_prob)
        labels[parent] = cell_id
    return labels

def _maybe_add_bud(parent_mask, parent_center, parent_axes, *, rng, prob):
    if rng.random() > prob:
        return parent_mask
    b_cen, b_axes, b_rot = add_bud_random_rotation(parent_center,
                                                   parent_axes, rng=rng)
    bud = ellipse_mask_rot_jitter(*parent_mask.shape, b_cen, b_axes, b_rot,
                                  jitter=0.54, noise_scale=70,
                                  seed=rng.integers(1<<31))
    return parent_mask | (bud & ~parent_mask)

def crop_box(img,dimensions,offset):
    """    Crop a TIFF to a box    """
    h_img, w_img,box_h,box_w,row_off,col_off \
        = 2048, 2048,  dimensions[0], dimensions[1], offset[0], offset[1]
    cropped = img[row_off : row_off + box_h, col_off : col_off + box_w]

    #Preserve original metadata & bit depth
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
        "fluorescence": [1000, 1200, 800, 1500, 900, 1100, 1300, 1400, 950],
        "size": [15, 18, 14, 20, 16, 17, 19, 18, 15],
        "shape": [(8, 19), (10, 13), (7, 6), (11, 7), (13, 8),
                  (9, 9), (11, 8), (45, 9), (18, 7)],
        "location": [(30, 25), (30, 75), (40, 80), (60, 30),
                     (85, 70), (110, 223), (110, 85), (110, 110), (80, 110)],
        "rotation": [0, 15, -20, 30, 0, 45, -10, 0, 25],
    }
    row_offset,col_offset   = center_offset((2048, 2048), (h, w))

    mask                    = generate_uint8_labels_with_buds(w, h, toy, rng=rng,
                                           bud_prob=BUD_PROB)

    cropped_uint8_labels    = canvas_slicer(mask, (h, w), (row_offset, col_offset))
    cropped_image0_metadata =   save_uint8_labels(cropped_uint8_labels, (h,w), (row_offset, col_offset),toy,"dump0\cropped_tester_mask")

    image0_metadata         =    save_uint8_labels(mask, (h,w), (row_offset, col_offset),toy,"dump0\mask_tester_mask")



    #image0                      = render_fluor_image(mask,image0_metadata,
                      #        bitdepth=BITDEPTH, gamma=GAMMA, rng=rng)
    image1 = render_fluor_image(mask,image0_metadata,bitdepth=BITDEPTH, gamma=GAMMA, rng=rng)
    cropped_image0              =  crop_box(image1,(h,w),(row_offset, col_offset))

    #save uncropped and uncropped masks


    imageio.imwrite("demo_fluor.tiff", cropped_image0)                     #training

    #Compute ADU min/max
    fluors = np.array(cropped_image0_metadata["fluorescence"], dtype=np.float32)
    min_adu, max_adu = fluors.min(), fluors.max()

    #define a stretch & false‑yellow function in this scope:
    def false_yellow_ADU(I16, *, min_adu, max_adu, gamma=0.7):
        """
        Linearly stretch I16 from [min_adu..max_adu] → [0..1],
        apply optional gamma, then false‑color into 0xCFCF00.
        """
        I = I16.astype(np.float32)
        I_norm = np.clip((I - min_adu) / (max_adu - min_adu),0,1)
        I_norm = I_norm ** (1 / gamma) if gamma != 1.0 else I_norm
        rgb_y = np.array([207, 207, 0], dtype=np.float32) / 255.0
        img = (I_norm[..., None] * rgb_y*(2**16-1)).astype(np.uint16)
        return img


    #ADU‑normalizer
    vis_yellow = (false_yellow_ADU(cropped_image0, min_adu=min_adu, max_adu=max_adu, gamma=GAMMA)).astype(np.uint16)
    #save raw uint16 + false-yellow PNG
    imageio.imwrite("vis_yellow.tiff", vis_yellow, format='TIFF')


    vis_rgb = visualize_uint8_labels(cropped_uint8_labels,cropped_image0_metadata,None)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(vis_rgb) ;          ax[0].set_title("Label RGB") ; ax[0].axis("off")
    ax[1].imshow(vis_yellow)   ;          ax[1].set_title("False-Yellow I₀") ; ax[1].axis("off")
    plt.tight_layout() ; plt.show()

    print(f"Saved demo_fluor.tiff (uint16) and vis_yellow.png – "
          f"run time {time.time() - t0:.2f}s")




    plt.imshow(vis_rgb); plt.axis("off")