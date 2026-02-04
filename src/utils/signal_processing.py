"""Signal processing for turn detection"""

import numpy as np
from scipy.optimize import curve_fit


def find_turn_selector(v, target_periods=10, x=None, R_turn_m=23.0, g=9.80665):
    """Find turns using FFT-based period detection.

    Returns
    -------
    x : ndarray
    fit_curve : ndarray
    turn_selector : boolean ndarray
    phi : ndarray  – lean angle (rad), zero on straights
    r_squared : float – R² of the sine fit (0–1)
    """
    v = np.asarray(v, dtype=float)
    n = len(v)

    if x is None:
        x = np.arange(n)
    x = np.asarray(x, dtype=float)

    # Edge case: constant or near-constant speed — no turns detectable
    v_range = np.ptp(v)
    if v_range < 1e-6:
        fit_curve = np.full_like(v, np.mean(v))
        turn_selector = np.zeros(n, dtype=bool)
        phi = np.zeros(n, dtype=float)
        return x, fit_curve, turn_selector, phi, 0.0

    dx = np.median(np.diff(x))
    y = v - np.mean(v)

    freqs = np.fft.rfftfreq(n, d=dx)
    spectrum = np.abs(np.fft.rfft(y))

    if len(freqs) < 3:
        raise ValueError("Not enough points for FFT")

    T_total = (x[-1] - x[0]) + dx
    periods_for_freqs = freqs * T_total
    idx_candidates = np.argsort(np.abs(periods_for_freqs - target_periods))

    close_idxs = [idx for idx in idx_candidates
                  if abs(periods_for_freqs[idx] - target_periods) <= 1]

    if not close_idxs:
        idx = idx_candidates[0]
    else:
        idx = max(close_idxs, key=lambda i: spectrum[i])

    chosen_freq = freqs[idx]

    def sine_func(x, A, phi_param, C):
        return A * np.sin(2 * np.pi * chosen_freq * x + phi_param) + C

    A0 = (np.percentile(v, 95) - np.percentile(v, 5)) / 2
    p0 = [A0, 0, np.mean(v)]

    try:
        popt, _ = curve_fit(sine_func, x, v, p0=p0, maxfev=10000)
    except Exception as e:
        raise RuntimeError("Sine fit failed") from e

    fit_curve = sine_func(x, *popt)

    # R-squared of sine fit
    ss_res = np.sum((v - fit_curve) ** 2)
    ss_tot = np.sum((v - np.mean(v)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # No quality gate - always proceed with turn detection
    # The UI will display a warning if R² is low

    A_est, phi_est, C_est = popt
    dfit_dx = A_est * 2 * np.pi * chosen_freq * np.cos(2 * np.pi * chosen_freq * x + phi_est)
    turn_selector = dfit_dx < 0

    # Lean angle: phi = arctan(v² / (g * R)), zero on straights
    phi = np.zeros(n, dtype=float)
    if R_turn_m > 0:
        phi[turn_selector] = np.arctan((v[turn_selector] ** 2) / (g * R_turn_m))

    return x, fit_curve, turn_selector, phi, r_squared
