"""
Fitting functions for various distributions and decay models.

This module provides functions for fitting statistical distributions to data,
including mixture models, decay functions, and standard statistical distributions.
"""

import warnings

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit, minimize
from scipy.special import gamma as gamma_function
from scipy.stats import weibull_min

# ============================================================================
# Helper Functions
# ============================================================================

def _calculate_aic_bic(log_likelihood: float, n_params: int, n_samples: int) -> tuple[float, float]:
    """
    Calculate AIC and BIC from log-likelihood.

    Args:
        log_likelihood: Log-likelihood value.
        n_params: Number of parameters in the model.
        n_samples: Number of samples.

    Returns:
        Tuple of (AIC, BIC).
    """
    if not np.isfinite(log_likelihood):
        return float("inf"), float("inf")
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    return float(aic), float(bic)


def _create_histogram_density(
    values: np.ndarray, n_bins: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create histogram and convert to density.

    Args:
        values: Array of values to histogram.
        n_bins: Number of bins.

    Returns:
        Tuple of (bin_centers, density, bin_widths).
    """
    counts, bin_edges = np.histogram(values, bins=n_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Convert to density
    total_counts = np.sum(counts)
    density = counts / (total_counts * bin_widths) if total_counts > 0 else np.zeros_like(counts)

    # Remove bins with zero density
    mask = density > 0
    bin_centers = bin_centers[mask]
    density = density[mask]

    return bin_centers, density, bin_widths


def _calculate_r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        observed: Observed values.
        fitted: Fitted values.

    Returns:
        R-squared value.
    """
    ss_res = np.sum((observed - fitted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _filter_to_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    """
    Filter values up to a given percentile.

    Args:
        values: Array of values.
        percentile: Percentile cutoff (e.g., 99.0 for 99th percentile).

    Returns:
        Filtered array.
    """
    cutoff_value = np.percentile(values, percentile)
    return values[values <= cutoff_value]


def fit_beta_distance(
    distances: np.ndarray,
    *,
    n_bins: int = 100,
    percentile_cutoff: float = 99.0,
) -> dict:
    """
    Fit Beta(a, b) to **scaled structural distances** (``distance_scaled`` in pairwise data)
    via MLE after linear scaling to (0, 1).

    Each distance ``d`` is mapped to ``u = (d - lo) / (hi - lo)`` where ``lo`` and ``hi``
    are the min and max of the same subset used for the histogram (values up to
    ``percentile_cutoff``).

    Reports:
        - ``a``, ``b``: fitted Beta shape parameters
        - ``m = a / (a + b)`` (mean on the scaled axis)
        - ``f = a + b`` (concentration / sample-size analogue)
        - ``variance = m (1 - m) / (f + 1)`` (Beta theoretical variance)
        - ``std``: square root of ``variance``
        - ``cv = std / m`` (coefficient of variation of the Beta mean; ``m > 0``)

    Args:
        distances: Array of nonnegative scaled distances (e.g. ``distance_scaled``; finite).
        n_bins: Histogram bin count for R² against binned counts.
        percentile_cutoff: Truncate long tail before min/max scaling (default 99th percentile).

    Returns:
        Dictionary with keys above; ``a``, ``b``, etc. may be ``None`` if fitting fails.
    """
    base: dict = {
        "a": None,
        "b": None,
        "m": None,
        "f": None,
        "variance": None,
        "std": None,
        "cv": None,
        "r_squared": None,
        "scale_lo": None,
        "scale_hi": None,
        "n_values_used": 0,
        "percentile_cutoff": float(percentile_cutoff),
    }

    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return base

    filtered = _filter_to_percentile(values, percentile_cutoff)
    if len(filtered) < 3:
        filtered = values
    n = int(len(filtered))
    base["n_values_used"] = n

    lo = float(np.min(filtered))
    hi = float(np.max(filtered))
    base["scale_lo"] = lo
    base["scale_hi"] = hi
    if hi <= lo:
        return base

    u = (filtered - lo) / (hi - lo)
    u = np.clip(u, 1e-9, 1.0 - 1e-9)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a_hat, b_hat, _, _ = scipy_stats.beta.fit(u, floc=0, fscale=1)
    except Exception:
        return base

    if (
        not np.isfinite(a_hat)
        or not np.isfinite(b_hat)
        or a_hat <= 0
        or b_hat <= 0
    ):
        return base

    m = float(a_hat / (a_hat + b_hat))
    f_sum = float(a_hat + b_hat)
    variance = float(m * (1.0 - m) / (f_sum + 1.0))
    std = float(np.sqrt(variance)) if variance >= 0 else None
    cv = float(std / m) if std is not None and m > 1e-15 else None

    # R²: observed vs expected bin counts under Beta on the distance scale
    counts, bin_edges = np.histogram(filtered, bins=n_bins)
    expected = np.zeros_like(counts, dtype=float)
    span = hi - lo
    for i in range(len(counts)):
        e0, e1 = float(bin_edges[i]), float(bin_edges[i + 1])
        u0 = np.clip((e0 - lo) / span, 0.0, 1.0)
        u1 = np.clip((e1 - lo) / span, 0.0, 1.0)
        if u1 <= u0:
            prob = 0.0
        else:
            prob = float(
                scipy_stats.beta.cdf(u1, a_hat, b_hat)
                - scipy_stats.beta.cdf(u0, a_hat, b_hat)
            )
        expected[i] = n * prob
    r_squared = float(_calculate_r_squared(counts.astype(float), expected))

    return {
        "a": float(a_hat),
        "b": float(b_hat),
        "m": m,
        "f": f_sum,
        "variance": variance,
        "std": std,
        "cv": cv,
        "r_squared": r_squared,
        "scale_lo": lo,
        "scale_hi": hi,
        "n_values_used": n,
        "percentile_cutoff": float(percentile_cutoff),
    }


# Backward-compatible alias
fit_beta_lipschitz = fit_beta_distance


# ============================================================================
# Distribution Functions
# ============================================================================


def fit_normal_uniform_mixture(values: np.ndarray) -> dict:
    """
    Fit a mixture of Normal and Uniform distributions to the values.
    
    The mixture model is: p * Normal(μ, σ) + (1-p) * Uniform(a, b)
    where p is the mixing proportion.
    
    Args:
        values: Array of values to fit.
    
    Returns:
        Dictionary with fit parameters: p (mixing proportion), mu, sigma (Normal),
        a, b (Uniform bounds), and fit quality metrics.
    """
    # Filter out NaN and inf values
    valid_mask = np.isfinite(values)
    if not np.all(valid_mask):
        values = values[valid_mask]
        if len(values) == 0:
            raise ValueError("No valid (finite) values to fit")
    
    if len(values) < 2:
        raise ValueError("Need at least 2 values to fit mixture model")
    
    def mixture_pdf(x, p, mu, sigma, a, b):
        """Mixture PDF: p * Normal(μ, σ) + (1-p) * Uniform(a, b)"""
        try:
            normal_part = p * scipy_stats.norm.pdf(x, mu, sigma)
            uniform_part = (1 - p) * scipy_stats.uniform.pdf(x, loc=a, scale=b - a)
            result = normal_part + uniform_part
            # Check for invalid values
            if np.any(~np.isfinite(result)):
                return np.full_like(result, 1e-10)
            return result
        except (ValueError, OverflowError):
            return np.full_like(x, 1e-10)
    
    def negative_log_likelihood(params):
        """Negative log-likelihood for optimization"""
        p, mu, sigma, a, b = params
        
        # Ensure valid parameter ranges
        if not (0 <= p <= 1):
            return np.inf
        if sigma <= 0 or not np.isfinite(sigma):
            return np.inf
        if a >= b or not (np.isfinite(a) and np.isfinite(b)):
            return np.inf
        if a > np.min(values) or b < np.max(values):
            return np.inf
        
        try:
            pdf_vals = mixture_pdf(values, p, mu, sigma, a, b)
            # Avoid log(0) and check for invalid values
            pdf_vals = np.maximum(pdf_vals, 1e-10)
            if not np.all(np.isfinite(pdf_vals)):
                return np.inf
            log_vals = np.log(pdf_vals)
            if not np.all(np.isfinite(log_vals)):
                return np.inf
            return -np.sum(log_vals)
        except (ValueError, OverflowError):
            return np.inf
    
    # Initial parameter estimates
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_mean = float(np.mean(values))
    data_std = float(np.std(values))
    
    # Ensure data_std is positive and finite
    if data_std <= 0 or not np.isfinite(data_std):
        data_std = 1.0
    
    # Initial guess: p=0.7 (mostly normal), normal params from data, uniform covers full range
    initial_params = [0.7, data_mean, data_std, data_min, data_max]
    
    # Bounds: p in [0, 1], sigma > 0, a < b, and reasonable ranges
    bounds = [
        (0.0, 1.0),  # p
        (data_min - 3 * data_std, data_max + 3 * data_std),  # mu
        (0.1 * data_std, 5 * data_std),  # sigma
        (data_min - 2 * data_std, data_min + 0.5 * data_std),  # a
        (data_max - 0.5 * data_std, data_max + 2 * data_std),  # b
    ]
    
    # Optimize with error handling
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
        except (ValueError, OverflowError):
            # If optimization fails catastrophically, use fallback
            result = type('obj', (object,), {'success': False, 'x': initial_params})()
    
    if not result.success:
        # Fallback to simpler estimates if optimization fails
        p_est = 0.5
        mu_est = data_mean
        sigma_est = data_std
        a_est = data_min
        b_est = data_max
    else:
        p_est, mu_est, sigma_est, a_est, b_est = result.x
    
    # Calculate AIC and BIC for model comparison
    n_params = 5
    n_samples = len(values)
    
    # Calculate log-likelihood with error handling
    try:
        log_likelihood = -negative_log_likelihood([p_est, mu_est, sigma_est, a_est, b_est])
        if not np.isfinite(log_likelihood):
            log_likelihood = 0.0
    except (ValueError, OverflowError):
        log_likelihood = 0.0
    
    aic, bic = _calculate_aic_bic(log_likelihood, n_params=5, n_samples=n_samples)
    
    return {
        "p": float(p_est),  # Mixing proportion (weight of Normal component)
        "mu": float(mu_est),  # Normal mean
        "sigma": float(sigma_est),  # Normal std
        "a": float(a_est),  # Uniform lower bound
        "b": float(b_est),  # Uniform upper bound
        "n_values": int(n_samples),
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "optimization_success": bool(result.success),
    }


# ============================================================================
# Decay and Distribution Functions
# ============================================================================

def kww_decay(x: np.ndarray, a: float, k: float, b: float) -> np.ndarray:
    """
    Kohlrausch-Williams-Watts decay function: a * exp(-k * x**b).
    
    Args:
        x: Input values.
        a: Amplitude parameter.
        k: Rate parameter.
        b: Shape parameter.
    
    Returns:
        Array of decay values.
    """
    # Avoid issues with x=0 and b<1
    x_safe = np.maximum(x, 1e-10)
    return a * np.exp(-k * np.power(x_safe, b))


# ============================================================================
# Fitting Functions
# ============================================================================

def generalized_gaussian(x: np.ndarray, a: float, u: float, s: float, b: float) -> np.ndarray:
    """
    Generalized Gaussian function: a * exp(-(|x-u|/s)^b)
    
    This is a generalization of the normal Gaussian distribution where the power
    is b instead of 2. When b=2, this reduces to a normal Gaussian.
    
    Args:
        x: Input values
        a: Amplitude parameter
        u: Location parameter (mean/center)
        s: Scale parameter
        b: Shape parameter (power, b=2 gives normal Gaussian)
    
    Returns:
        Array of function values
    """
    return a * np.exp(-np.power(np.abs(x - u) / s, b))


def fit_generalized_gaussian(
    values: np.ndarray,
    n_bins: int = 50,
    percentile_cutoff: float = 99.0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Fit a generalized Gaussian distribution a*exp(-(|x-u|/s)^b) to the values.
    
    Args:
        values: Array of values to fit.
        n_bins: Number of bins for histogram.
        percentile_cutoff: Percentile cutoff for fitting (excludes outliers).
    
    Returns:
        Tuple of (fit_params, bin_centers, fitted_values) where:
        - fit_params: dict with 'a', 'u', 's', 'b', and fitting statistics
        - bin_centers: array of bin centers used for fitting
        - fitted_values: array of fitted values at bin_centers
    """
    # Filter values up to percentile cutoff
    filtered_values = _filter_to_percentile(values, percentile_cutoff)
    
    if len(filtered_values) < 10:
        # Not enough data, return default values
        return {
            "a": 0.0,
            "u": float(np.mean(values)) if len(values) > 0 else 0.0,
            "s": 0.0,
            "b": 2.0,
            "r_squared": 0.0,
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": len(filtered_values),
        }, np.array([]), np.array([])
    
    # Create histogram and convert to density
    bin_centers, density, _ = _create_histogram_density(filtered_values, n_bins)
    
    if len(bin_centers) < 5:
        return {
            "a": 0.0,
            "u": float(np.mean(values)) if len(values) > 0 else 0.0,
            "s": 0.0,
            "b": 2.0,
            "r_squared": 0.0,
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": len(filtered_values),
        }, np.array([]), np.array([])
    
    # Initial parameter estimates
    data_mean = float(np.mean(filtered_values))
    data_std = float(np.std(filtered_values))
    a0 = np.max(density)
    u0 = data_mean
    s0 = data_std
    b0 = 2.0  # Start with b=2 (normal Gaussian)
    
    try:
        popt, _ = curve_fit(
            generalized_gaussian,
            bin_centers,
            density,
            p0=[a0, u0, s0, b0],
            bounds=([0, -np.inf, 0.01, 0.1], [np.inf, np.inf, np.inf, 10.0]),
            maxfev=10000,
        )
        a, u, s, b = popt
        
        # Calculate R-squared
        fitted = generalized_gaussian(bin_centers, a, u, s, b)
        r_squared = _calculate_r_squared(density, fitted)
        
        fit_params = {
            "a": float(a),
            "u": float(u),
            "s": float(s),
            "b": float(b),
            "r_squared": float(r_squared),
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": int(len(filtered_values)),
        }
        
        fitted_values = generalized_gaussian(bin_centers, a, u, s, b)
        
    except Exception:
        # Fitting failed, return default values
        fit_params = {
            "a": 0.0,
            "u": float(np.mean(values)) if len(values) > 0 else 0.0,
            "s": 0.0,
            "b": 2.0,
            "r_squared": 0.0,
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": int(len(filtered_values)),
        }
        fitted_values = np.zeros_like(bin_centers)
    
    return fit_params, bin_centers, fitted_values


def fit_kww_decay(
    values: np.ndarray,
    n_bins: int = 50,
    percentile_cutoff: float = 99.0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Fit Kohlrausch-Williams-Watts decay a*exp(-k*x**b) to the distribution of values.
    
    Args:
        values: Array of values to fit.
        n_bins: Number of bins for histogram.
        percentile_cutoff: Percentile cutoff for fitting (excludes outliers).
    
    Returns:
        Tuple of (fit_params, bin_centers, fitted_values) where:
        - fit_params: dict with 'a', 'k', 'b', and fitting statistics
        - bin_centers: array of bin centers used for fitting
        - fitted_values: array of fitted values at bin_centers
    """
    # Filter values up to percentile cutoff
    filtered_values = _filter_to_percentile(values, percentile_cutoff)
    
    if len(filtered_values) < 10:
        # Not enough data, return default values
        return {
            "a": 0.0,
            "k": 0.0,
            "b": 1.0,
            "r_squared": 0.0,
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": len(filtered_values),
        }, np.array([]), np.array([])
    
    # Create histogram and convert to density
    bin_centers, density, _ = _create_histogram_density(filtered_values, n_bins)
    
    if len(bin_centers) < 5:
        return {
            "a": 0.0,
            "k": 0.0,
            "b": 1.0,
            "r_squared": 0.0,
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": len(filtered_values),
        }, np.array([]), np.array([])
    
    # Initial parameter estimates
    a0 = np.max(density)
    k0 = 1.0
    b0 = 0.5  # Start with b<1 for heavy-tailed distributions
    
    try:
        popt, _ = curve_fit(
            kww_decay, 
            bin_centers, 
            density,
            p0=[a0, k0, b0],
            bounds=([0, 0, 0.01], [np.inf, np.inf, 2.0]),  # Allow b between 0.01 and 2.0
            maxfev=10000,
        )
        a, k, b = popt
        
        # Calculate R-squared
        fitted = kww_decay(bin_centers, a, k, b)
        r_squared = _calculate_r_squared(density, fitted)
        
        fit_params = {
            "a": float(a),
            "k": float(k),
            "b": float(b),
            "r_squared": float(r_squared),
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": int(len(filtered_values)),
        }
        
        fitted_values = kww_decay(bin_centers, a, k, b)
        
    except Exception:
        # Fitting failed, return default values
        fit_params = {
            "a": 0.0,
            "k": 0.0,
            "b": 1.0,
            "r_squared": 0.0,
            "percentile_cutoff": percentile_cutoff,
            "n_values_used": int(len(filtered_values)),
        }
        fitted_values = np.zeros_like(bin_centers)
    
    return fit_params, bin_centers, fitted_values


def kww_theoretical_statistics(
    a: float | None,
    k: float | None,
    b: float | None,
) -> dict:
    """
    Theoretical summaries from KWW (stretched exponential) parameters ``a``, ``k``, ``b``
    as in the fitted decay **density** model ``a * exp(-k * r^b)`` (see ``kww_decay``).

    Interpreting the **Weibull** survival ``S(r) = exp(-k r^b)`` (same ``k``, ``b`` as in the exponent):

    * **95th percentile:** ``r95 = (-ln(0.05 / a) / k)^(1/b)`` — user-specified form; requires
      ``a > 0.05`` so that ``-ln(0.05/a) > 0``, and ``k > 0``, ``b > 0``.
    * **Mean:** ``E[r] = Gamma(1 + 1/b) / k^(1/b)`` (Weibull mean with shape ``b`` and scale ``k^(-1/b)``).

    These are reported alongside empirical histogram percentiles / mean for comparison.

    Args:
        a: Amplitude from ``fit_kww_decay``.
        k: Rate-like parameter from ``fit_kww_decay``.
        b: Shape parameter from ``fit_kww_decay``.

    Returns:
        Dictionary with ``r95``, ``mean_Er`` (float or None), ``success``, optional ``error``,
        and ``formulas`` for documentation.
    """
    formulas = {
        "r95": "(-ln(0.05/a)/k)^(1/b)",
        "mean_Er": "Gamma(1+1/b) / k^(1/b)",
    }
    out: dict = {
        "r95": None,
        "mean_Er": None,
        "formulas": formulas,
        "success": False,
        "error": None,
    }
    if a is None or k is None or b is None:
        out["error"] = "missing_parameters"
        return out
    try:
        a = float(a)
        k = float(k)
        b = float(b)
    except (TypeError, ValueError):
        out["error"] = "non_numeric_parameters"
        return out

    if k <= 0 or b <= 0:
        out["error"] = "k_and_b_must_be_positive"
        return out

    # Mean of Weibull with survival exp(-k r^b): E[R] = Gamma(1+1/b) / k^(1/b)
    try:
        mean_er = float(gamma_function(1.0 + 1.0 / b) / (k ** (1.0 / b)))
        out["mean_Er"] = mean_er
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        out["error"] = f"mean_computation_failed: {e}"
        return out

    out["success"] = True

    # r95 = (-ln(0.05/a)/k)^(1/b); need a > 0.05 for -ln(0.05/a) > 0 with typical a
    if a <= 0:
        out["r95_note"] = "a_must_be_positive"
        return out
    if a <= 0.05:
        out["r95_note"] = "formula_requires_a_gt_0_05_for_positive_-ln(0.05/a)"
        return out
    try:
        inner = -np.log(0.05 / a) / k
        if inner <= 0 or not np.isfinite(inner):
            out["r95_note"] = "inner_-ln(0.05/a)/k_must_be_positive"
            return out
        out["r95"] = float(inner ** (1.0 / b))
    except (ValueError, FloatingPointError) as e:
        out["r95_note"] = str(e)
    return out


def fit_normal(values: np.ndarray) -> dict:
    """
    Fit a Normal (Gaussian) distribution to the values.
    
    Args:
        values: Array of values to fit.
    
    Returns:
        Dictionary with fit parameters: mu (mean), sigma (standard deviation),
        and fit quality metrics.
    """
    if len(values) < 2:
        return {
            "mu": float(np.mean(values)) if len(values) > 0 else 0.0,
            "sigma": 0.0,
            "n_values": len(values),
            "log_likelihood": 0.0,
            "aic": 0.0,
            "bic": 0.0,
        }
    
    # Use scipy's fit method for maximum likelihood estimation
    mu, sigma = scipy_stats.norm.fit(values)
    
    # Calculate log-likelihood
    log_likelihood = np.sum(scipy_stats.norm.logpdf(values, mu, sigma))
    
    # Calculate AIC and BIC
    aic, bic = _calculate_aic_bic(log_likelihood, n_params=2, n_samples=len(values))
    
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "n_values": int(len(values)),
        "log_likelihood": float(log_likelihood),
        "aic": aic,
        "bic": bic,
    }


def fit_exponential(values: np.ndarray) -> dict:
    """
    Fit an Exponential distribution to the values.
    
    The exponential distribution PDF is: lambda * exp(-lambda * x) for x >= 0
    where lambda = 1 / scale is the rate parameter.
    
    Args:
        values: Array of values to fit (should be non-negative).
    
    Returns:
        Dictionary with fit parameters: lambda (rate parameter), scale (1/lambda),
        and fit quality metrics.
    """
    if len(values) < 2:
        return {
            "lambda": 0.0,
            "scale": 0.0,
            "n_values": len(values),
            "log_likelihood": 0.0,
            "aic": 0.0,
            "bic": 0.0,
        }
    
    # Ensure non-negative values for exponential distribution
    values_positive = np.maximum(values, 0)
    
    # Use scipy's fit method for maximum likelihood estimation
    # expon.fit returns (loc, scale) where scale = 1/lambda
    _, scale = scipy_stats.expon.fit(values_positive, floc=0)  # Fix loc=0
    
    # Rate parameter lambda = 1 / scale
    lambda_param = 1.0 / scale if scale > 0 else 0.0
    
    # Calculate log-likelihood
    log_likelihood = np.sum(scipy_stats.expon.logpdf(values_positive, loc=0, scale=scale))
    
    # Calculate AIC and BIC
    aic, bic = _calculate_aic_bic(log_likelihood, n_params=1, n_samples=len(values))
    
    return {
        "lambda": float(lambda_param),
        "scale": float(scale),
        "n_values": int(len(values)),
        "log_likelihood": float(log_likelihood),
        "aic": aic,
        "bic": bic,
    }


def fit_weibull(values: np.ndarray) -> dict:
    """
    Fit a Weibull distribution to the values.
    
    The Weibull distribution PDF is: weibull_min.pdf(x, c=power, scale=scale)
    where c is the shape parameter (power) and scale is the scale parameter.
    
    Args:
        values: Array of values to fit (should be non-negative).
    
    Returns:
        Dictionary with fit parameters: c (shape/power parameter), scale (scale parameter),
        and fit quality metrics.
    """
    if len(values) < 2:
        return {
            "c": 0.0,
            "scale": 0.0,
            "n_values": len(values),
            "log_likelihood": 0.0,
            "aic": 0.0,
            "bic": 0.0,
        }
    
    # Ensure non-negative values for Weibull distribution
    values_positive = np.maximum(values, 0)
    
    # Use scipy's fit method for maximum likelihood estimation
    # weibull_min.fit returns (c, loc, scale) where c is the shape parameter
    c, _, scale = weibull_min.fit(values_positive, floc=0)  # Fix loc=0
    
    # Calculate log-likelihood
    log_likelihood = np.sum(weibull_min.logpdf(values_positive, c=c, loc=0, scale=scale))
    
    # Calculate AIC and BIC
    aic, bic = _calculate_aic_bic(log_likelihood, n_params=2, n_samples=len(values))
    
    return {
        "c": float(c),
        "scale": float(scale),
        "n_values": int(len(values)),
        "log_likelihood": float(log_likelihood),
        "aic": aic,
        "bic": bic,
    }


def sigmoid(x: np.ndarray, c: float, a: float, k: float, x0: float) -> np.ndarray:
    """
    Sigmoid function: c + a / (1 + exp(k * (x - x0))).

    Args:
        x: Input values.
        c: Baseline offset.
        a: Amplitude.
        k: Steepness of the curve.
        x0: Midpoint of the sigmoid.

    Returns:
        Sigmoid values.
    """
    return c + a / (1 + np.exp(k * (x - x0)))


def fit_sigmoid(
    x: np.ndarray, y: np.ndarray, decreasing: bool = True, max_x: float | None = None
) -> dict[str, float]:
    """
    Fit a sigmoid function to the data.

    Args:
        x: X values (e.g., similarity thresholds or percent ranks).
        y: Y values (e.g., fraction or n_pairs).
        decreasing: If True, fit a decreasing sigmoid (k > 0). If False, fit an increasing sigmoid (k < 0).
                    Default is True.
        max_x: Maximum x value for bounds. If None, uses x.max().

    Returns:
        Dictionary with fitted parameters: c, a, k, x0.
        Returns NaN values if fitting fails.
    """
    if len(x) < 4 or len(y) < 4:  # Need at least 4 points for 4 parameters
        return {"c": float("nan"), "a": float("nan"), "k": float("nan"), "x0": float("nan")}

    # Initial parameter guesses
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    x_min, x_max = np.nanmin(x), np.nanmax(x)

    # Adjust initial guess for 'a' based on decreasing/increasing
    if decreasing:
        p0 = [y_min, y_max - y_min, 1, np.mean(x)]  # c, a, k, x0
        bounds = ([0, 0, 0, x_min], [np.inf, np.inf, np.inf, max_x if max_x is not None else x_max])
    else:  # Increasing sigmoid
        p0 = [y_min, y_max - y_min, -1, np.mean(x)]  # c, a, k, x0 (k is negative for increasing)
        bounds = ([0, 0, -np.inf, x_min], [np.inf, np.inf, 0, max_x if max_x is not None else x_max])

    try:
        popt, _ = curve_fit(
            sigmoid, x, y, p0=p0, maxfev=10000,
            bounds=bounds,
            nan_policy='omit'  # Handle NaN values in y
        )
        return {
            "c": float(popt[0]),
            "a": float(popt[1]),
            "k": float(popt[2]),
            "x0": float(popt[3]),
        }
    except (RuntimeError, ValueError):
        # If fitting fails, return NaN values
        return {"c": float("nan"), "a": float("nan"), "k": float("nan"), "x0": float("nan")}


# ============================================================================
# Scaled distance distribution comparison (BIC)
# ============================================================================

def _fit_gaussian_for_bic(values: np.ndarray) -> dict:
    """Gaussian (Normal) MLE; 2 parameters (mu, sigma)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return _failed_fit("gaussian", "too_few_samples")
    mu, sigma = scipy_stats.norm.fit(v)
    if sigma <= 0 or not np.isfinite(sigma):
        return _failed_fit("gaussian", "invalid_sigma")
    ll = float(np.sum(scipy_stats.norm.logpdf(v, mu, sigma)))
    aic, bic = _calculate_aic_bic(ll, n_params=2, n_samples=len(v))
    return {
        "distribution": "gaussian",
        "success": True,
        "n_params": 2,
        "parameters": {"mu": float(mu), "sigma": float(sigma)},
        "log_likelihood": ll,
        "aic": float(aic),
        "bic": float(bic),
        "n_samples": int(len(v)),
    }


def _fit_beta_for_bic(values: np.ndarray) -> dict:
    """Beta on [0, 1]; 2 parameters (a, b) with floc=0, fscale=1."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return _failed_fit("beta", "too_few_samples")
    # Interior (0, 1) for stable MLE
    x = np.clip(v, 1e-8, 1.0 - 1e-8)
    try:
        a, b, _loc, _scale = scipy_stats.beta.fit(x, floc=0, fscale=1)
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
            return _failed_fit("beta", "invalid_parameters")
        ll = float(np.sum(scipy_stats.beta.logpdf(x, a, b, loc=0, scale=1)))
        aic, bic = _calculate_aic_bic(ll, n_params=2, n_samples=len(x))
        return {
            "distribution": "beta",
            "success": True,
            "n_params": 2,
            "parameters": {"a": float(a), "b": float(b)},
            "log_likelihood": ll,
            "aic": float(aic),
            "bic": float(bic),
            "n_samples": int(len(x)),
        }
    except Exception as e:
        return _failed_fit("beta", str(e))


def _fit_gamma_for_bic(values: np.ndarray) -> dict:
    """Gamma with fixed loc=0; 2 parameters (shape, scale)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.maximum(v, 1e-12)
    if len(v) < 2:
        return _failed_fit("gamma", "too_few_samples")
    try:
        shape, loc, scale = scipy_stats.gamma.fit(v, floc=0)
        if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
            return _failed_fit("gamma", "invalid_parameters")
        ll = float(np.sum(scipy_stats.gamma.logpdf(v, shape, loc=0, scale=scale)))
        aic, bic = _calculate_aic_bic(ll, n_params=2, n_samples=len(v))
        return {
            "distribution": "gamma",
            "success": True,
            "n_params": 2,
            "parameters": {"shape": float(shape), "scale": float(scale)},
            "log_likelihood": ll,
            "aic": float(aic),
            "bic": float(bic),
            "n_samples": int(len(v)),
        }
    except Exception as e:
        return _failed_fit("gamma", str(e))


def _fit_lognormal_for_bic(values: np.ndarray) -> dict:
    """Log-normal with fixed loc=0; 2 parameters (shape s, scale)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.maximum(v, 1e-12)
    if len(v) < 2:
        return _failed_fit("lognormal", "too_few_samples")
    try:
        shape, loc, scale = scipy_stats.lognorm.fit(v, floc=0)
        if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
            return _failed_fit("lognormal", "invalid_parameters")
        ll = float(np.sum(scipy_stats.lognorm.logpdf(v, shape, loc=0, scale=scale)))
        aic, bic = _calculate_aic_bic(ll, n_params=2, n_samples=len(v))
        return {
            "distribution": "lognormal",
            "success": True,
            "n_params": 2,
            "parameters": {"shape": float(shape), "scale": float(scale)},
            "log_likelihood": ll,
            "aic": float(aic),
            "bic": float(bic),
            "n_samples": int(len(v)),
        }
    except Exception as e:
        return _failed_fit("lognormal", str(e))


def _fit_uniform_for_bic(values: np.ndarray) -> dict:
    """Uniform on [loc, loc + scale]; 2 parameters."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return _failed_fit("uniform", "too_few_samples")
    try:
        lo, sc = scipy_stats.uniform.fit(v)
        if not np.isfinite(lo) or not np.isfinite(sc) or sc <= 0:
            return _failed_fit("uniform", "invalid_parameters")
        ll = float(np.sum(scipy_stats.uniform.logpdf(v, lo, sc)))
        aic, bic = _calculate_aic_bic(ll, n_params=2, n_samples=len(v))
        return {
            "distribution": "uniform",
            "success": True,
            "n_params": 2,
            "parameters": {"loc": float(lo), "scale": float(sc)},
            "log_likelihood": ll,
            "aic": float(aic),
            "bic": float(bic),
            "n_samples": int(len(v)),
        }
    except Exception as e:
        return _failed_fit("uniform", str(e))


def _failed_fit(name: str, reason: str) -> dict:
    return {
        "distribution": name,
        "success": False,
        "error": reason,
        "n_params": 0,
        "parameters": {},
        "log_likelihood": None,
        "aic": None,
        "bic": None,
        "n_samples": 0,
    }


def fit_scaled_distances_bic_comparison(values: np.ndarray) -> dict:
    """
    Fit Gaussian, Beta, Gamma, log-normal, and Uniform distributions to scaled distances.

    Uses maximum likelihood and BIC = k * ln(n) - 2 * log_likelihood (lower is better).
    All models use 2 free parameters where applicable (fair comparison).

    Args:
        values: 1D array of scaled distances (typically in [0, 1]).

    Returns:
        Dictionary with:
        - ``fits``: mapping distribution name -> fit dict (log_likelihood, bic, parameters, ...).
        - ``best_by_bic``: name of the distribution with lowest finite BIC.
        - ``best_fit``: copy of the winning fit dict.
    """
    fitters = {
        "gaussian": _fit_gaussian_for_bic,
        "beta": _fit_beta_for_bic,
        "gamma": _fit_gamma_for_bic,
        "lognormal": _fit_lognormal_for_bic,
        "uniform": _fit_uniform_for_bic,
    }
    fits: dict[str, dict] = {}
    for key, fn in fitters.items():
        fits[key] = fn(values)

    best_name: str | None = None
    best_bic = float("inf")
    for name, fd in fits.items():
        if not fd.get("success"):
            continue
        bic = fd.get("bic")
        if bic is None or not np.isfinite(bic):
            continue
        if bic < best_bic:
            best_bic = bic
            best_name = name

    best_fit = dict(fits[best_name]) if best_name else None

    return {
        "n_samples": int(np.sum(np.isfinite(values))),
        "fits": fits,
        "best_by_bic": best_name,
        "best_bic": float(best_bic) if best_name else None,
        "best_fit": best_fit,
    }


def evaluate_scaled_distance_fit_pdf(
    distribution_name: str,
    parameters: dict,
    x: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the PDF of a named fitted distribution at points ``x``.

    Args:
        distribution_name: One of ``gaussian``, ``beta``, ``gamma``, ``lognormal``, ``uniform``.
        parameters: Parameter dict as returned in each fit's ``parameters`` field.
        x: Points where to evaluate the PDF.

    Returns:
        PDF values (same shape as ``x``).
    """
    x = np.asarray(x, dtype=float)
    name = distribution_name.lower()
    if name == "gaussian":
        return np.asarray(scipy_stats.norm.pdf(x, parameters["mu"], parameters["sigma"]), dtype=float)
    if name == "beta":
        return np.asarray(
            scipy_stats.beta.pdf(x, parameters["a"], parameters["b"], loc=0, scale=1), dtype=float
        )
    if name == "gamma":
        return np.asarray(
            scipy_stats.gamma.pdf(x, parameters["shape"], loc=0, scale=parameters["scale"]),
            dtype=float,
        )
    if name == "lognormal":
        return np.asarray(
            scipy_stats.lognorm.pdf(x, parameters["shape"], loc=0, scale=parameters["scale"]),
            dtype=float,
        )
    if name == "uniform":
        return np.asarray(scipy_stats.uniform.pdf(x, parameters["loc"], parameters["scale"]), dtype=float)
    raise ValueError(f"Unknown distribution: {distribution_name}")


def fit_power_law_cumulative_cliff_fraction(
    percent: np.ndarray,
    F_n: np.ndarray,
) -> dict:
    """
    Fit cumulative cliff fraction F(n) ≈ a · n^s, where n is percent rank (1–100).

    Uses nonlinear least squares. R² is computed on the points used for fitting
    (finite F(n), n > 0).

    Args:
        percent: Percent ranks (e.g. 1, 2, …, 100).
        F_n: F(n) = (cliff pairs in top n% most similar) / (total cliff pairs).

    Returns:
        Dictionary with ``success``, ``a``, ``s``, ``r_squared``, and optional ``error``.
    """
    percent = np.asarray(percent, dtype=float)
    F_n = np.asarray(F_n, dtype=float)
    mask = np.isfinite(percent) & np.isfinite(F_n) & (percent > 0)
    if mask.sum() < 2:
        return {"success": False, "error": "insufficient_points", "a": None, "s": None, "r_squared": None}

    n_use = percent[mask]
    F_use = F_n[mask]

    def model(n: np.ndarray, a: float, s: float) -> np.ndarray:
        return a * np.power(np.maximum(n, 1e-9), s)

    # Initial guess from log-linear regression on strictly positive F
    pos = F_use > 1e-15
    if pos.sum() >= 2:
        logn = np.log(n_use[pos])
        logF = np.log(F_use[pos])
        coef = np.polyfit(logn, logF, 1)
        s0 = float(coef[0])
        a0 = float(np.exp(coef[1]))
    else:
        a0, s0 = 1e-4, 1.0

    try:
        popt, _pcov = curve_fit(
            model,
            n_use,
            F_use,
            p0=[a0, s0],
            bounds=([1e-30, -20], [1e30, 30]),
            maxfev=50000,
        )
        a, s = float(popt[0]), float(popt[1])
        F_pred = model(n_use, a, s)
        ss_res = float(np.sum((F_use - F_pred) ** 2))
        ss_tot = float(np.sum((F_use - np.mean(F_use)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
        return {
            "success": True,
            "a": a,
            "s": s,
            "r_squared": r2,
            "n_points_fitted": int(mask.sum()),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "a": None,
            "s": None,
            "r_squared": None,
        }


def gini_from_power_law_exponent(s: float) -> float | None:
    """
    Gini coefficient derived from power-law exponent: G = (s - 1) / (s + 1).

    Undefined when s = -1.
    """
    if s is None or not np.isfinite(s) or abs(s + 1.0) < 1e-12:
        return None
    return float((s - 1.0) / (s + 1.0))

