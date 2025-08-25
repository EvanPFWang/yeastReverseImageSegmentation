from __future__ import annotations

import math
import numpy as np
from typing import Callable, Iterable, Sequence, Tuple, Optional

def kahan_sum(values: Iterable[float]) -> float:
    """compute sum of sequence of floats w/
     Kahan compensated summation algo
    """
    total = 0.0
    c = 0.0  #running compensation for lost low‑order bits
    for value in values:
        y = value - c       #apply compensation
        t = total + y       #add adjusted value torunning total
        c = (t - total) - y  #compute new compensation
        total = t
    return total


def pairwise_sum(values: Sequence[float]) -> float:

    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    #recursively sumleft and right halves
    mid = n // 2
    return pairwise_sum(values[:mid]) + pairwise_sum(values[mid:])


def safe_subtract(x: float, y: float, threshold: float = 1e-12) -> float:

    #compute relative magnitude of difference.
    denom = max(abs(x), abs(y), 1.0)
    if abs(x - y) <= threshold * denom and (x + y) != 0.0:
        #useidentity (x^2 - y^2) / (x + y) to avoid cancellation
        return (x * x - y * y) / (x + y)
    return x - y


def stable_sqrt1_minus_delta(delta: float) -> float:

    #Guard against negative values inside square root due to rounding
    one_minus_delta = 1.0 - delta
    if one_minus_delta < 0:
        #if delta slightly exceeds 1 due to rounding, zero clamp
        one_minus_delta = 0.0
    sqrt_term = math.sqrt(one_minus_delta)
    #if delta tiny,denominator is close to 2; direct formula is stable.
    denom = 1.0 + sqrt_term
    if denom != 0.0:
        return delta / denom
    #fallback todirect formula (won't occur for real delta <= 1).
    return 1.0 - sqrt_term


def quadratic_roots_stable(a: float, b: float, c: float) -> Tuple[float, float]:
    if a == 0:
        raise ZeroDivisionError("Coefficient 'a' must be non‑zero for a quadratic equation")
    discriminant = b * b - 4.0 * a * c
    #use complex sqrt for negative discriminant to handle complex roots
    sqrt_disc = math.sqrt(discriminant) if discriminant >= 0 else complex(0.0, math.sqrt(-discriminant))
    #determinesign of b forstable formula.  If b == 0, choose +1
    sign_b = 1.0 if b >= 0 else -1.0
    #computefirst root usingstable formula
    denom = -b - sign_b * sqrt_disc
    #avoid division by zero if denom is zero (happens when discriminant == 0)
    if denom == 0:
        #discriminant zero implies repeated root; both roots equal -b/(2a)
        root = -b / (2.0 * a)
        return (root, root)
    x1 = denom / (2.0 * a)
    #computesecond root usingrelationship x1 * x2 = c/a
    x2 = (2.0 * c) / denom
    return (x1, x2)


def safe_divide(numerator: float, denominator: float, eps: float = 1e-15) -> float:
    if abs(denominator) < eps:
        raise ZeroDivisionError(f"Denominator {denominator!r} is too close to zero to safely divide")
    return numerator / denominator


def derivative_central(f: Callable[[float], float], x: float, eps: float | None = None) -> float:
    """approx deriv of scalar function w/ central diff"""
    if eps is None:
        eps = np.finfo(float).eps
    #choose h proportional to sqrt(eps) and magn of x
    h = math.sqrt(eps) * max(abs(x), 1.0)
    return (f(x + h) - f(x - h)) / (2.0 * h)


def logsumexp(values: Sequence[float]) -> float:
    """compute log(sum[exp(values)]) in numerically stable way"""
    if len(values) == 0:
        return -math.inf
    m = max(values)
    #if m -inf, all values -inf and result is -inf
    if m == -math.inf:
        return -math.inf
    total = sum(math.exp(v - m) for v in values)
    return m + math.log(total)


def relative_error(true_value: float, approx_value: float) -> float:
    if true_value == 0:
        return abs(approx_value - true_value)
    return abs(approx_value - true_value) / abs(true_value)


def absolute_error(true_value: float, approx_value: float) -> float:
    return abs(approx_value - true_value)


def series_sum_until_convergence(term_generator: Callable[[int], float], tol: float = 1e-12, max_terms: int = 10_000) -> Tuple[float, int]:
    """sum inf (or long fin) series until succ terms are below tolerance"""
    total = 0.0
    compensation = 0.0
    for i in range(max_terms):
        term = term_generator(i)
        if abs(term) < tol:
            #converged: stop adding further terms
            return total, i
        #Kahan compensated summation to accumulateterm
        y = term - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    #return even convergence not achieved avoid infinite loops.
    return total, max_terms


_F64 = np.float64
_C128 = np.complex128

#helpers
def _to_f64(x):
    return np.asarray(x, dtype=_F64)

def _move_axis_last(x, axis):
    return np.moveaxis(x, axis, -1) if axis is not None else np.ravel(x)

#acc reductions (sum/mean/var/dot)

def pairwise_sumnp(x, axis: Optional[int] = None, dtype=_F64):
    """pairwise (tree) reduction for improved sum acc works on scalars/arrays O(log n) passes"""
    a = np.asarray(x, dtype=dtype)
    if axis is None:
        a = a.ravel()
        if a.size == 0:
            return dtype(0)
        n = a.size
        #pad to next power-of-two
        pow2 = 1 << (int(np.ceil(np.log2(n))))
        if pow2 != n:
            a = np.pad(a, (0, pow2 - n), mode='constant')
        #iter halving
        while a.size > 1:
            a = a.reshape(-1, 2).sum(axis=1, dtype=dtype)
        return a[0]
    else:
        a = np.moveaxis(a, axis, -1)
        n = a.shape[-1]
        if n == 0:
            return np.zeros(a.shape[:-1], dtype)
        pow2 = 1 << (int(np.ceil(np.log2(n))))
        if pow2 != n:
            pad_width = [(0,0)]*(a.ndim-1) + [(0, pow2 - n)]
            a = np.pad(a, pad_width, mode='constant')
        while a.shape[-1] > 1:
            a = a.reshape(*a.shape[:-1], -1, 2).sum(axis=-1, dtype=dtype)
        return a[..., 0]

def sorted_sum_small_to_large(x, axis: Optional[int] = None, dtype=_F64):
    """vectorized w/ argsort/take_along_axis and sum by increasing |value| to reduce cancellation """
    a = np.asarray(x, dtype=dtype)
    if axis is None:
        a = a.ravel()
        if a.size == 0:
            return dtype(0)
        idx = np.argsort(np.abs(a), kind='stable')
        return np.sum(a[idx], dtype=dtype)
    else:
        #move axis -> last for clean indexing
        aT = np.moveaxis(a, axis, -1)
        idx = np.argsort(np.abs(aT), axis=-1, kind='stable')
        ordered = np.take_along_axis(aT, idx, axis=-1)
        s = np.sum(ordered, axis=-1, dtype=dtype)
        return s

def stable_mean(x, axis: Optional[int] = None, method: str = "pairwise"):
    """
    mean computed from a more stable sum
    method: "pairwise" or "sorted"
    """
    a = np.asarray(x, dtype=_F64)
    if axis is None:
        n = a.size
        if n == 0:
            return _F64(np.nan)
        total = pairwise_sumnp(a) if method == "pairwise" else sorted_sum_small_to_large(a)
        return total / n
    else:
        n = a.shape[axis]
        total = pairwise_sumnp(a, axis=axis) if method == "pairwise" else sorted_sum_small_to_large(a, axis=axis)
        return total / n

def stable_var_two_pass(x, axis: Optional[int] = None, ddof: int = 0):
    """
    two-pass (mean then centered squares) variance in float64
    more stable than naive single-pass; works on 1D/broadcasted axes
    """
    a = np.asarray(x, dtype=_F64)
    m = stable_mean(a, axis=axis, method="pairwise")
    #center and accumulate in f64
    centered = a - (m if axis is None else np.expand_dims(m, axis=axis))
    ss = pairwise_sumnp(centered*centered, axis=axis)
    denom = (a.size if axis is None else a.shape[axis]) - ddof
    return ss / denom

def stable_dot(x, y, axis: Optional[int] = None):
    """
    stable dot w/ pairwise sum of elementwise product (1D-friendly)
    if axis None, flattens and dots; else reduces along "axis"
    """
    a = _to_f64(x); b = _to_f64(y)
    prod = a * b
    return pairwise_sumnp(prod, axis=axis)

#stable exp/log domain primitives

def logsumexpnp(x, axis: Optional[int] = None, keepdims: bool = False):
    """stable log-sum-exp w/ ufunc reduction: logaddexp.reduce (pairwise in NumPy)"""
    return np.logaddexp.reduce(_to_f64(x), axis=axis, keepdims=keepdims)

def softmax_stable(x, axis: int = -1):
    """stable softmax by subtracting the max"""
    x = _to_f64(x)
    m = np.max(x, axis=axis, keepdims=True)
    z = np.exp(x - m)
    denom = np.sum(z, axis=axis, keepdims=True)
    return z / denom

def sigmoid_stable(x):
    """numerically stable logistic σ(x)"""
    x = _to_f64(x)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def softplus_stable(x):
    """log(1 + exp(x)) computed stabl"""
    x = _to_f64(x)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def log1mexp(x):
    """
    log(1 - exp(x)) for x <= 0 computed stably.
    use if x < -log(2) -> log1p(-exp(x)); else -> log(-expm1(x))
    """
    x = _to_f64(x)
    cutoff = -np.log(2.0)
    return np.where(x < cutoff, np.log1p(-np.exp(x)), np.log(-np.expm1(x)))

def pow1pm1(x, y):
    """compute (1+x)^y - 1 stably via expm1(y*log1p(x))"""
    x,y = _to_f64(x), _to_f64(y)
    return np.expm1(y * np.log1p(x))

def cosm1(x):
    """cos(x) - 1 computed stably: -2 sin^2(x/2)"""
    return -2.0 * np.sin(0.5 * _to_f64(x)) ** 2

def sinc_stable(x):
    """
    sin(x)/x with a series expansion near 0 (|x| small).
    """
    x = _to_f64(x)
    ax = np.abs(x)
    small = ax < 1e-6
    #1 - x^2/6 + x^4/120 is plenty for float64 near 0
    series = 1.0 - (x*x)/6.0 + (x*x*x*x)/120.0
    return np.where(small, series, np.sin(x)/x)

#Products via log-domain & sign tracking

def log_abs_prod(x, axis: Optional[int] = None, keepdims: bool = False):
    """
    sum log(abs(x)) (useful avoids overflow/underflow when forming products)
    returns -inf if any zeros present along axis
    """
    a = _to_f64(x)
    if axis is None:
        if np.any(a == 0):
            return _F64(-np.inf)
        return np.sum(np.log(np.abs(a)), dtype=_F64)
    else:
        zero = np.any(a == 0, axis=axis, keepdims=keepdims)
        s = np.sum(np.log(np.abs(np.where(a==0, 1.0, a))), axis=axis, dtype=_F64, keepdims=keepdims)
        return np.where(zero, -np.inf, s)

def prod_sign(x, axis: Optional[int] = None, keepdims: bool = False):
    """sign of the product along axis (0 if any zero present)"""
    a = _to_f64(x)
    if axis is None:
        if np.any(a == 0): return _F64(0.0)
        neg = np.sum(a < 0)
        return _F64(-1.0 if (neg % 2) else 1.0)
    else:
        zero = np.any(a == 0, axis=axis, keepdims=keepdims)
        neg = np.sum(a < 0, axis=axis, keepdims=keepdims)
        sgn = np.where(neg % 2 == 0, 1.0, -1.0)
        return np.where(zero, 0.0, sgn).astype(_F64)

def safe_prod_from_logs(x, axis: Optional[int] = None, keepdims: bool = False):
    """
    product reconstructed from log-domain + sign tracking
    may overflow on exponentiation if magn is enormous; prefer keeping logs
    """
    sgn = prod_sign(x, axis=axis, keepdims=keepdims)
    lg = log_abs_prod(x, axis=axis, keepdims=keepdims)
    #where log is -inf (zeros), product is exactly 0
    return np.where(np.isneginf(lg), 0.0, sgn * np.exp(lg))

#stable norms / scaling

def scaled_l2_norm(x, axis: Optional[int] = None, keepdims: bool = False):
    """overflow/underflow-safe L2 norm via scaling by max |x|"""
    a = _to_f64(x)
    if axis is None:
        scale = np.max(np.abs(a))
        if scale == 0: return _F64(0.0)
        return scale * np.sqrt(np.sum((a/scale)**2, dtype=_F64))
    else:
        scale = np.max(np.abs(a), axis=axis, keepdims=True)
        out = np.where(scale == 0, 0.0, (scale * np.sqrt(np.sum((a/scale)**2, axis=axis, dtype=_F64, keepdims=True))).astype(_F64))
        return out if keepdims else np.squeeze(out, axis=axis)

#quadratic roots (vectorized + corrected stable formula)

def quadratic_roots_stablenp(a, b, c) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized, numerically stable quadratic roots
    uses: q = -0.5*(b + sign(b)*sqrt(b^2 - 4ac)), x1 = q/a, x2 = c/q
    handles complex discriminants via complex128
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    #promote to complex if needed
    disc = b*b - 4.0*a*c
    sqrt_disc = np.where(np.isrealobj(disc) & (disc >= 0),
                          np.sqrt(disc.astype(_F64)),
                          np.sqrt(disc.astype(_C128)))
    sign_b = np.where(b >= 0, 1.0, -1.0).astype(_F64)
    q = -0.5 * (b + sign_b * sqrt_disc)
    #repeated root mask where q == 0 (within float tolerance) -> -b/(2a)
    #use absolute for complex as well
    q_is_zero = np.isclose(np.abs(q), 0.0)
    repeated = (-b) / (2.0*a)
    x1 = np.where(q_is_zero, repeated, q / a)
    x2 = np.where(q_is_zero, repeated, c / q)
    return x1, x2

#finite diff (cancellation-safe)

def derivative_complex_step(f: Callable[[float], complex], x: float, h: float = 1e-20) -> float:
    """
    complex-step derivative: Im(f(x + i*h))/h no subtraction -> no cancellation
    req f to accept complex input; returns real derivative
    """
    return np.imag(f(x + 1j*h)) / h

def derivative_5point(f: Callable[[float], float], x: float, eps: float | None = None) -> float:
    """
    5-point centered finite-diff derivative: O(h^4) trunc error
    h approx eps^(1/5)*scale for balance (scale = max(|x|,1))
    """
    if eps is None:
        eps = np.finfo(float).eps
    scale = max(abs(x), 1.0)
    h = (eps ** 0.2) * scale
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12.0*h)

#robust differences / divides (vectorized)

def safe_subtractnp(x, y, threshold: float = 1e-12):
    """
    vectorized cancellation-avoiding subtract using (x^2 - y^2)/(x + y) when |x-y| small.
    """
    x = _to_f64(x); y = _to_f64(y)
    denom = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1.0)
    close = np.abs(x - y) <= threshold * denom
    alt = (x*x - y*y) / (x + y)
    return np.where(close & (x + y != 0), alt, x - y)

def guarded_divide(x, y, eps: float = 1e-15):
    """
    elementwise safe divide: clamps very small denominators to ±eps to avoid overflow
    """
    x = _to_f64(x); y = _to_f64(y)
    y_safe = np.where(np.abs(y) < eps, np.sign(y) * eps, y)
    return x / y_safe

#probabilities/log-probabilities helpers

def normalize_log_probs(logits, axis: int = -1):
    """
    ret log-probabilities: logits - logsumexp(logits)
    """
    logits = _to_f64(logits)
    lse = logsumexpnp(logits, axis=axis, keepdims=True)
    return logits - lse

def entropy_from_probs(p, axis: int = -1):
    """
    shannon entropy H(p) = -sum p*log(p), stable small p w/ x*log(x)->0
    uses np.where avoids nan at p=0
    """
    p = _to_f64(p)
    plogp = np.where(p > 0, p * np.log(p), 0.0)
    return -np.sum(plogp, axis=axis)

#ULP utilities

def nextafter_n(x, n: int):
    """
    move x by n ULPs toward +inf (n>0) or -inf (n<0)
    """
    x = _to_f64(x)
    if n == 0:
        return x
    direction = np.inf if n > 0 else -np.inf
    out = x.copy()
    for _ in range(abs(n)):
        out = np.nextafter(out, direction)
    return out

def ulp_distance(a, b):
    """
    approx ULP distance by counting nextafter steps (vectorized via while-loop per element is expensive);
    here we return |a-b| / spacing(midpoint) as a practical proxy.
    """
    a = _to_f64(a); b = _to_f64(b)
    mid = 0.5*(a + b)
    return np.abs(a - b) / np.spacing(mid)



__all__ = [
    "kahan_sum",
    "pairwise_sum",
    "safe_subtract",
    "stable_sqrt1_minus_delta",
    "quadratic_roots_stable",
    "safe_divide",
    "derivative_central",
    "logsumexp",
    "relative_error",
    "absolute_error",
    "series_sum_until_convergence",
    #reductions
    "pairwise_sumnp", "sorted_sum_small_to_large", "stable_mean", "stable_var_two_pass", "stable_dot",
    #exp/log
    "logsumexpnp", "softmax_stable", "sigmoid_stable", "softplus_stable", "log1mexp", "pow1pm1", "cosm1",
    "sinc_stable",
    #products
    "log_abs_prod", "prod_sign", "safe_prod_from_logs",
    #norms
    "scaled_l2_norm",
    #roots
    "quadratic_roots_stablenp",
    #finite diffs
    "derivative_complex_step", "derivative_5point",
    #safe ops
    "safe_subtractnp", "guarded_divide",
    #probabilities
    "normalize_log_probs", "entropy_from_probs",
    #ulp
    "nextafter_n", "ulp_distance",
]