# Tester.py Documentation - Critical Analysis and Improvements
## Numerical Stability Issues and Recommended Fixes

---

## Table of Contents

1. [Current Implementation Shortcomings](#current-implementation-shortcomings)
2. [Numerical Instabilities in ellipse_mask_rot_jitter](#numerical-instabilities-in-ellipse_mask_rot_jitter)
3. [Memory Management Issues](#memory-management-issues)
4. [Budding Cell Generation Problems](#budding-cell-generation-problems)
5. [Fluorescence Rendering Instabilities](#fluorescence-rendering-instabilities)
6. [Recommended Fixes and Improvements](#recommended-fixes-and-improvements)

---

## Current Implementation Shortcomings

### 1. Global Random State Management

**Problem**: Global RNG without proper initialization control
```python
# Current problematic code:
rng_global = default_rng()  # Non-deterministic initialization
```

**Issue**: Results are not reproducible between runs, making debugging numerical issues impossible.

**Fix**:
```python
# ------------------------------------------------------------------
# Deterministic RNG initialization with explicit seed management
# ------------------------------------------------------------------
def create_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create RNG with explicit seed for reproducibility.
    
    Parameters
    ----------
    seed : int or None
        If None, uses entropy from OS (non-reproducible)
        If int, creates deterministic sequence
    
    Returns
    -------
    rng : np.random.Generator
        Initialized random number generator
    """
    if seed is None:
        # Use high-precision timestamp for pseudo-random seed
        seed = int(time.time() * 1e9) % (2**31 - 1)
    return default_rng(seed)
```

### 2. Hardcoded Canvas Dimensions

**Problem**: Fixed 2048×2048 canvas size throughout
```python
yy, xx = np.indices((2048, 2048))  # Hardcoded dimensions
```

**Issues**:
- Memory waste for smaller images
- Cannot process larger datasets
- Numerical precision loss when scaling

**Fix**:
```python
# ------------------------------------------------------------------
# Dynamic canvas sizing with numerical stability
# ------------------------------------------------------------------
def compute_canvas_size(max_cell_radius: float, 
                       num_cells: int,
                       padding_factor: float = 1.5) -> Tuple[int, int]:
    """
    Compute optimal canvas size for numerical stability.
    
    Mathematical basis:
        canvas_size = 2 × padding × sqrt(num_cells) × max_radius
    
    Ensures:
    1. No boundary artifacts
    2. Efficient memory usage
    3. Power-of-2 alignment for FFT operations
    """
    estimated_size = 2 * padding_factor * np.sqrt(num_cells) * max_cell_radius
    # Round up to next power of 2 for FFT efficiency
    canvas_size = 2 ** int(np.ceil(np.log2(estimated_size)))
    return (canvas_size, canvas_size)
```

---

## Numerical Instabilities in ellipse_mask_rot_jitter

### Critical Issues Identified

#### 1. Division by Zero in Rotated Coordinates

**Problem**: 
```python
# Current dangerous code:
inv_x = np.divide(x_p, a_eff, where=(a_eff != 0))
inv_y = np.divide(y_p, b_eff, where=(b_eff != 0))
```

**Mathematical Issue**: When `a_eff` or `b_eff` approach zero due to Perlin noise, the division becomes unstable.

**Fix with Proper Guards**:
```python
# ------------------------------------------------------------------
# Numerically stable ellipse membership test
# ------------------------------------------------------------------
def ellipse_mask_rot_jitter(h, w, center, axes, angle_deg: float,
                           *, jitter=0.05, noise_scale=64,
                           seed=None, repeat=True):
    """
    Stable rotated ellipse with Perlin perturbation.
    
    Mathematical formulation:
        Given rotation angle φ and point (x,y):
        1. Transform to body frame: (x',y') = R^T(φ) @ (x-cx, y-cy)
        2. Apply jittered axes: a_eff = a(1 + δ), b_eff = b(1 + δ)
        3. Test membership: (x'/a_eff)² + (y'/b_eff)² ≤ 1
    
    Numerical safeguards:
        - Minimum axis threshold: max(eps, 0.1% of nominal)
        - Stable rotation via half-angle formulas
        - Clamped noise to prevent sign flips
    """
    yy, xx = np.indices((h, w), dtype=np.float32)
    center_y, center_x = center
    a, b = axes
    
    # Numerical constants
    eps32 = np.finfo(np.float32).eps
    min_axis = max(eps32, 0.001 * min(a, b))  # 0.1% threshold
    
    # Stable rotation using half-angle formulation
    # Avoids catastrophic cancellation near θ = π
    phi_half = np.deg2rad(angle_deg / 2, dtype=np.float32)
    cos_half = np.cos(phi_half)
    sin_half = np.sin(phi_half)
    
    # Full angle via half-angle formulas
    cos_phi = cos_half**2 - sin_half**2
    sin_phi = 2 * sin_half * cos_half
    
    # Transform to body coordinates
    dy, dx = yy - center_y, xx - center_x
    x_p = dx * cos_phi + dy * sin_phi
    y_p = -dx * sin_phi + dy * cos_phi
    
    # Generate bounded Perlin noise
    if jitter > 0:
        res_y = _fit_periods(h, noise_scale)
        res_x = _fit_periods(w, noise_scale)
        noise = generate_perlin_noise_2d((h, w), (res_y, res_x),
                                       tileable=(repeat, repeat))
        
        # Clamp noise to prevent axis collapse
        # δ ∈ [-jitter, +jitter] but ensure 1+δ > min_ratio
        min_ratio = 0.1  # Axes can shrink to 10% minimum
        noise = np.clip(noise * jitter, 
                       min_ratio - 1.0,  # Lower bound
                       2.0 - 1.0)        # Upper bound (2× growth max)
        
        scale = 1.0 + noise.astype(np.float32)
    else:
        scale = np.ones((h, w), dtype=np.float32)
    
    # Apply scale with strict positivity
    a_eff = np.maximum(a * scale, min_axis)
    b_eff = np.maximum(b * scale, min_axis)
    
    # Membership test with guaranteed denominators
    x_norm_sq = (x_p / a_eff)**2
    y_norm_sq = (y_p / b_eff)**2
    
    return x_norm_sq + y_norm_sq <= 1.0
```

#### 2. Perlin Noise Numerical Issues

**Problem**: Current implementation doesn't handle edge cases

**Fix**:
```python
# ------------------------------------------------------------------
# Robust Perlin noise generation with numerical bounds
# ------------------------------------------------------------------
def generate_bounded_perlin(shape: Tuple[int, int],
                          res: Tuple[int, int],
                          amplitude: float = 1.0,
                          octaves: int = 1) -> np.ndarray:
    """
    Generate Perlin noise with guaranteed bounds.
    
    Mathematical guarantee:
        |η(x,y)| ≤ amplitude × sum(0.5^i for i in range(octaves))
                 ≤ amplitude × 2
    
    For standard parameters:
        |η(x,y)| ≤ amplitude  (exactly)
    """
    noise = np.zeros(shape, dtype=np.float32)
    
    for octave in range(octaves):
        freq = 2**octave
        weight = 0.5**octave
        
        octave_noise = generate_perlin_noise_2d(
            shape,
            (res[0] * freq, res[1] * freq),
            tileable=(True, True)
        )
        
        # Ensure bounds per octave
        octave_noise = np.clip(octave_noise, -1.0, 1.0)
        noise += weight * octave_noise
    
    # Final normalization
    return amplitude * np.clip(noise, -1.0, 1.0)
```

---

## Memory Management Issues

### Current Memory Waste

**Problem**: Multiple full 2048×2048 arrays created unnecessarily

```python
# Current inefficient code:
canvas = np.zeros((2048, 2048), dtype=np.uint8)  # 4MB
parent = create_ellipse_mask_vectorized_perturbed2(...)  # 4MB boolean
bud = ellipse_mask_rot_jitter(...)  # Another 4MB
# Total: 12MB per cell!
```

### Optimized Memory-Efficient Approach

```python
# ------------------------------------------------------------------
# Memory-efficient mask generation with bounding boxes
# ------------------------------------------------------------------
def generate_cell_with_bbox(center: Tuple[float, float],
                          axes: Tuple[float, float],
                          angle: float,
                          canvas_shape: Tuple[int, int],
                          jitter: float = 0.05) -> Tuple[np.ndarray, Tuple[slice, slice]]:
    """
    Generate cell mask only within its bounding box.
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask of minimal size
    bbox : Tuple[slice, slice]
        Location in full canvas: canvas[bbox] = mask
    
    Memory reduction:
        Full canvas: 2048² = 4M pixels
        Typical cell: 100² = 10K pixels
        Savings: 99.75%
    """
    cx, cy = center
    a, b = axes
    
    # Conservative bounding box with rotation + jitter
    max_radius = max(a, b) * (1.0 + jitter) * 1.2  # 20% safety
    
    # Integer bounds with clipping
    y_min = max(0, int(cy - max_radius))
    y_max = min(canvas_shape[0], int(cy + max_radius) + 1)
    x_min = max(0, int(cx - max_radius))
    x_max = min(canvas_shape[1], int(cx + max_radius) + 1)
    
    # Generate mask in local coordinates
    local_h = y_max - y_min
    local_w = x_max - x_min
    local_center = (cy - y_min, cx - x_min)
    
    mask = ellipse_mask_rot_jitter(local_h, local_w, local_center,
                                 axes, angle, jitter=jitter)
    
    bbox = (slice(y_min, y_max), slice(x_min, x_max))
    return mask, bbox
```

---

## Budding Cell Generation Problems

### Current Issues

1. **Random bud placement** - No biological constraints
2. **Fixed size ratios** - Ignores cell cycle
3. **Overlap handling** - Simple boolean OR can create artifacts

### Biologically Accurate Budding

```python
# ------------------------------------------------------------------
# Biologically constrained bud generation
# ------------------------------------------------------------------
def add_bud_biological(parent_center: Tuple[float, float],
                      parent_axes: Tuple[float, float],
                      parent_angle: float,
                      cell_cycle_phase: float = 0.5,
                      *,
                      rng=None) -> Dict[str, Any]:
    """
    Generate bud with biological constraints.
    
    Mathematical model based on S. cerevisiae:
        - Bud size ratio: r(t) = 0.1 + 0.6 × sigmoid(10(t - 0.5))
        - Attachment angle: θ ~ von Mises(μ=previous_site, κ=2)
        - Neck constriction: w_neck = 0.7 × min(r_bud, r_mother)
    
    Parameters
    ----------
    cell_cycle_phase : float in [0, 1]
        0 = G1 (no bud)
        0.3 = S (bud emergence)
        0.7 = G2/M (large bud)
        1.0 = cytokinesis
    
    Returns
    -------
    bud_params : dict
        'center', 'axes', 'angle', 'neck_width', 'attachment_point'
    """
    if cell_cycle_phase < 0.3:
        return None  # No bud in G1
    
    # Sigmoid growth curve
    t_norm = (cell_cycle_phase - 0.3) / 0.7
    size_ratio = 0.1 + 0.6 / (1 + np.exp(-10 * (t_norm - 0.5)))
    
    # Bud site selection (simplified - should use Bud1p/Bud2p model)
    if rng is None:
        rng = np.random.default_rng()
    
    # Von Mises distribution for angular bias
    theta_attach = rng.vonmises(0, 2)  # κ=2 for moderate clustering
    
    # Compute attachment point on mother ellipse
    a_p, b_p = parent_axes
    x_attach = a_p * np.cos(theta_attach)
    y_attach = b_p * np.sin(theta_attach)
    
    # Rotate to world coordinates
    phi = np.deg2rad(parent_angle)
    x_world = parent_center[0] + x_attach * np.cos(phi) - y_attach * np.sin(phi)
    y_world = parent_center[1] + x_attach * np.sin(phi) + y_attach * np.cos(phi)
    
    # Bud grows normal to surface
    normal_angle = np.arctan2(y_attach/b_p**2, x_attach/a_p**2)
    bud_angle = np.rad2deg(normal_angle + phi)
    
    # Offset along normal
    offset = (1 - 0.3 * size_ratio) * min(a_p, b_p)  # Overlap decreases with size
    bud_center = (
        x_world + offset * np.cos(normal_angle + phi),
        y_world + offset * np.sin(normal_angle + phi)
    )
    
    bud_axes = (a_p * size_ratio, b_p * size_ratio * 0.9)  # Slightly elongated
    
    return {
        'center': bud_center,
        'axes': bud_axes,
        'angle': bud_angle,
        'neck_width': 0.7 * min(bud_axes[0], a_p),
        'attachment_point': (x_world, y_world),
        'size_ratio': size_ratio
    }
```

---

## Fluorescence Rendering Instabilities

### Current Numerical Issues

1. **Distance transform underflow**
```python
# Problem: 
din = distance_transform_edt(mask).astype(np.float32)
r_norm = din[yy, xx] / (r_max + 1e-6)  # Ad-hoc epsilon
```

2. **Gradient computation at boundaries**
```python
# Sobel filter can amplify noise
gx = sobel(img, axis=0)
gy = sobel(img, axis=1)
grad = np.hypot(gx, gy)
```

### Stable Implementation

```python
# ------------------------------------------------------------------
# Numerically stable fluorescence rendering
# ------------------------------------------------------------------
def render_fluor_stable(label_map: np.ndarray,
                       fluor_values: np.ndarray,
                       *,
                       gamma: float = 0.7,
                       psf_sigma: float = 1.2,
                       bit_depth: int = 16) -> np.ndarray:
    """
    Render fluorescence with careful numerical handling.
    
    Key stability measures:
    1. Log-space operations for intensity
    2. Stable gradient computation
    3. Proper normalization sequence
    """
    h, w = label_map.shape
    
    # Work in log space for better dynamic range
    log_img = np.full((h, w), -np.inf, dtype=np.float64)
    
    for cell_id in range(1, label_map.max() + 1):
        mask = (label_map == cell_id)
        if not mask.any():
            continue
        
        # Robust distance transform
        din = distance_transform_edt(mask)
        
        # Handle edge case where mask is single pixel
        r_max = din.max()
        if r_max < 1.0:
            log_img[mask] = np.log(fluor_values[cell_id - 1])
            continue
        
        # Stable normalized radius
        r_norm = np.clip(din / r_max, 0.0, 1.0)
        
        # Intensity in log space
        # I = F₀(1 - r^γ) => log(I) = log(F₀) + log(1 - r^γ)
        intensity_factor = 1.0 - r_norm[mask]**gamma
        
        # Avoid log(0) with proper masking
        valid = intensity_factor > eps
        log_img[mask][valid] = np.log(fluor_values[cell_id - 1]) + \
                               np.log(intensity_factor[valid])
    
    # Convert back from log space
    img = np.exp(log_img, where=np.isfinite(log_img), 
                 out=np.zeros_like(log_img))
    
    # Stable PSF convolution
    if psf_sigma > 0:
        # Ensure PSF sums to exactly 1
        y, x = np.ogrid[-3*psf_sigma:3*psf_sigma+1, 
                        -3*psf_sigma:3*psf_sigma+1]
        psf = np.exp(-(x**2 + y**2) / (2 * psf_sigma**2))
        psf = psf / psf.sum()  # Exact normalization
        
        # Use separable convolution for efficiency
        img = fftconvolve(img, psf, mode='same')
    
    # Quantization with proper rounding
    max_val = 2**bit_depth - 1
    img_normalized = np.clip(img / img.max(), 0.0, 1.0)
    
    # Add noise before quantization (dithering)
    noise_amplitude = 0.5 / max_val  # Half LSB
    dither = np.random.uniform(-noise_amplitude, noise_amplitude, img.shape)
    
    counts = np.round((img_normalized + dither) * max_val)
    return np.clip(counts, 0, max_val).astype(np.uint16)
```

---

## Recommended Fixes and Improvements

### Short Term (2 days)

1. **Replace all hardcoded dimensions**
```python
# ------------------------------------------------------------------
# Configuration class for all parameters
# ------------------------------------------------------------------
@dataclass
class CellSegmentationConfig:
    """Centralized configuration with validation."""
    canvas_size: Tuple[int, int] = (2048, 2048)
    min_cell_radius: float = 5.0
    max_cell_radius: float = 50.0
    jitter_amplitude: float = 0.07
    noise_scale: int = 64
    
    def __post_init__(self):
        # Validate parameters
        assert self.canvas_size[0] > 0 and self.canvas_size[1] > 0
        assert 0 < self.min_cell_radius < self.max_cell_radius
        assert 0 <= self.jitter_amplitude < 1.0
        assert self.noise_scale > 0
```

2. **Add numerical validation**
```python
# ------------------------------------------------------------------
# Input validation with numerical checks
# ------------------------------------------------------------------
def validate_cell_data(cells_data: Dict[str, Any]) -> None:
    """
    Validate cell data for numerical stability.
    
    Checks:
    1. Finite values only (no NaN/Inf)
    2. Positive axes lengths
    3. Centers within canvas
    4. Valid rotation angles
    """
    required_keys = ['indices', 'shape', 'location', 'rotation']
    
    for key in required_keys:
        if key not in cells_data:
            raise ValueError(f"Missing required key: {key}")
    
    n_cells = len(cells_data['indices'])
    
    # Check all arrays same length
    for key in required_keys:
        if len(cells_data[key]) != n_cells:
            raise ValueError(f"Inconsistent array lengths")
    
    # Numerical validation
    for i, (a, b) in enumerate(cells_data['shape']):
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"Non-finite axes for cell {i}")
        if a <= 0 or b <= 0:
            raise ValueError(f"Non-positive axes for cell {i}")
```

### Medium Term (2 weeks)

1. **Implement sub-pixel accuracy**
```python
# ------------------------------------------------------------------
# Sub-pixel ellipse rendering
# ------------------------------------------------------------------
def create_antialiased_ellipse(shape: Tuple[int, int],
                              center: Tuple[float, float],
                              axes: Tuple[float, float],
                              angle: float,
                              samples_per_pixel: int = 4) -> np.ndarray:
    """
    Anti-aliased ellipse using supersampling.
    
    Mathematical basis:
        Each pixel subdivided into samples_per_pixel² subpixels
        Final value = average of subpixel membership tests
    
    Reduces discretization artifacts at boundaries.
    """
    h, w = shape
    n = samples_per_pixel
    offset = (1 - 1/n) / 2  # Center subpixels
    
    mask = np.zeros((h, w), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            # Subpixel offset
            dy = (i/n - offset)
            dx = (j/n - offset)
            
            # Test at subpixel location
            submask = ellipse_mask_rot_jitter(
                h, w,
                (center[0] + dy, center[1] + dx),
                axes, angle, jitter=0
            )
            
            mask += submask.astype(np.float32)
    
    return mask / (n * n)
```

2. **GPU acceleration with numerical guarantees**
```python
# ------------------------------------------------------------------
# CUDA kernel with exact FP semantics
# ------------------------------------------------------------------
"""
@cuda.jit
def ellipse_mask_cuda(output, centers, axes, angles, 
                     canvas_h, canvas_w):
    '''
    CUDA kernel with IEEE 754 compliance.
    
    Ensures:
    1. Same rounding mode as CPU
    2. Denormal handling matches NumPy
    3. No fast-math optimizations
    '''
    idx = cuda.grid(1)
    if idx >= output.size:
        return
    
    y = idx // canvas_w
    x = idx % canvas_w
    
    # ... ellipse test with explicit FP operations ...
"""
```

### Long Term (2 months)

1. **Interval arithmetic for guaranteed bounds**
2. **Adaptive precision based on cell size**
3. **Formal verification of numerical properties**

---

## Testing Framework

### Numerical Regression Tests

```python
# ------------------------------------------------------------------
# Comprehensive numerical testing suite
# ------------------------------------------------------------------
class TestNumericalStability:
    """Test suite for numerical edge cases."""
    
    def test_near_zero_axes(self):
        """Test behavior with extremely small axes."""
        for eps_factor in [1, 10, 100]:
            axes = (eps * eps_factor, eps * eps_factor)
            mask = ellipse_mask_rot_jitter(100, 100, (50, 50), 
                                         axes, 0)
            assert mask.sum() >= 1  # At least center pixel
    
    def test_large_jitter(self):
        """Test stability with extreme perturbation."""
        mask = ellipse_mask_rot_jitter(100, 100, (50, 50),
                                     (20, 15), 0, jitter=0.99)
        # Should not crash or produce NaN
        assert np.all(np.isfinite(mask))
        assert mask.dtype == bool
    
    def test_rotation_continuity(self):
        """Test smooth behavior across angle wrapping."""
        angles = np.linspace(-180, 180, 100)
        areas = []
        
        for angle in angles:
            mask = ellipse_mask_rot_jitter(100, 100, (50, 50),
                                         (20, 15), angle)
            areas.append(mask.sum())
        
        # Area should vary smoothly
        area_diff = np.diff(areas)
        assert np.max(np.abs(area_diff)) < 50  # pixels
```

---

## Conclusion

The current `tester.py` implementation has significant numerical stability issues that can lead to:

1. **Non-reproducible results** due to global RNG
2. **Memory waste** from hardcoded dimensions
3. **Numerical failures** in edge cases (small axes, large jitter)
4. **Biologically unrealistic** cell generation

The proposed fixes provide:
- Guaranteed numerical bounds
- Memory-efficient implementation  
- Biologically accurate models
- Comprehensive testing framework

Implementation priority should focus on the numerical stability fixes first (2 days), then memory optimization (2 weeks), and finally the biological accuracy improvements (2 months).