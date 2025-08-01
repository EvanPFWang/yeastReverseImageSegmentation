# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS + VISUALISER (slot this in once, replace the old versions)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.cluster import MiniBatchKMeans
from skimage.color import lab2rgb
import numpy as np
import matplotlib.pyplot as plt

def gradient_palette(n_colors: int, *, seed: int | None = None) -> np.ndarray:
    """
    Return an (n_colors,3) uint8 array spanning a perceptually uniform
    gradient in CIELab → RGB.  Darker tones come first so label 1 appears
    darker than label n.
    """
    rng = np.random.default_rng(seed)
    # sample a wide Lab cube then K-means cluster
    lab = rng.uniform([20, -40, -40], [90, 40, 40], size=(4096, 3)).astype(np.float32)
    km  = MiniBatchKMeans(n_clusters=n_colors, batch_size=1024,
                          random_state=seed).fit(lab)
    centres = km.cluster_centers_
    order   = np.argsort(centres[:, 0])          # sort by lightness L*
    rgb     = lab2rgb(centres[order][None])[0]   # -> 0-1 float
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def make_colormap(unique_cell_ids: np.ndarray,
                  *, background: tuple[int, int, int] = (0, 0, 0)
                  ) -> np.ndarray:
    """
    Build a palette where *row i* is the colour for *cell_id i*.
    `unique_cell_ids` must include **all** positive IDs present.
    Row 0 is the background colour.
    """
    n = unique_cell_ids.max() if unique_cell_ids.size else 0
    palette = np.zeros((n + 1, 3), dtype=np.uint8)
    palette[0] = background
    if n:
        palette[1:] = gradient_palette(n)
    return palette

def visualize_uint8_labels(
        coll_masks: np.ndarray,
        metadata: dict | list | None = None,
        palette: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a stack **(N,H,W)** of binary masks to an RGB visualisation
    **and** return the palette used.

    Returns
    -------
    rgb_vis : (H,W,3) uint8  – label image in colour
    palette : (max_id+1,3) uint8 – row i = colour for cell_id i
    """
    if coll_masks.ndim != 3:
        raise ValueError("coll_masks must have shape (N,H,W)")
    masks = coll_masks.astype(bool, copy=False)
    N, H, W = masks.shape

    # numeric label map
    label_img = (masks * (np.arange(1, N + 1,
                                    dtype=np.uint16)[:, None, None])).max(axis=0)

    # build / extend palette
    if palette is None:
        if isinstance(metadata, dict):
            cell_ids = np.asarray(metadata.get("unique_labels",
                                               np.arange(1, N + 1)), dtype=int)
        else:
            cell_ids = np.arange(1, N + 1, dtype=int)
        palette = make_colormap(cell_ids)
    if palette.shape[0] <= label_img.max():
        extra = gradient_palette(label_img.max() - palette.shape[0] + 1)
        palette = np.vstack([palette, extra])

    rgb_vis = palette[label_img]          # LUT
    return rgb_vis, palette

def palette_to_strip(palette: np.ndarray, h: int, thickness: int = 20
                     ) -> np.ndarray:
    """
    Turn *palette* (n,3) → vertical strip image (h, thickness, 3) for legends.
    """
    taps   = palette.shape[0]
    knots  = np.linspace(0, h - 1, taps).astype(np.float32)
    base_y = np.arange(h, dtype=np.float32)
    col    = np.empty((h, 1, 3), dtype=np.uint8)
    for ch in range(3):
        col[:, 0, ch] = np.interp(base_y, knots, palette[:, ch]).round()
    return np.flipud(np.tile(col, (1, thickness, 1)))   # darkest at top

def show_labels_and_palette(label_rgb: np.ndarray,
                            palette: np.ndarray,
                            *,
                            cell_ids: list[int] | None = None,
                            thickness: int = 32,
                            figsize=(8, 4)):
    """
    Quick preview: label RGB on the left, palette bar + IDs on the right.
    """
    if cell_ids is None:
        cell_ids = list(range(1, palette.shape[0]))
    strip = palette_to_strip(palette, label_rgb.shape[0], thickness)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=figsize
    )
    # label image
    ax0.imshow(label_rgb)
    ax0.set_axis_off()
    ax0.set_title("Label RGB")

    # palette strip
    ax1.imshow(strip)
    ax1.set_axis_off()
    for i, cid in enumerate(cell_ids):
        y = int(label_rgb.shape[0] * (i + 0.5) / len(cell_ids))
        ax1.text(thickness + 4, y, str(cid),
                 va='center', ha='left', fontsize=8, color='white')
    ax1.set_title("Palette")
    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
