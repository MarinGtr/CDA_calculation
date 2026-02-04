"""CdA estimation with Monte Carlo error propagation"""

import numpy as np


def r2_weight(r2, threshold=0.2, steepness=20):
    """Sigmoid weighting based on R² quality.

    Smooth weighting function that penalizes Mode 3 when R² is low.

    - r2 >= 0.4: weight ≈ 1.0 (full contribution)
    - r2 = 0.2: weight = 0.5 (transition point)
    - r2 <= 0.05: weight ≈ 0.0 (minimal contribution)

    Parameters
    ----------
    r2 : float
        R² value from sine fit (0 to 1)
    threshold : float
        R² value at which weight = 0.5 (default 0.2)
    steepness : float
        Controls how sharp the transition is (default 20)

    Returns
    -------
    float
        Weight between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-steepness * (r2 - threshold)))


def compute_interval_cda(mode1_cda, mode2_cda, mode2_se, mode3_cda, mode3_se, r2):
    """
    Combine Mode 1, Mode 2, and Mode 3 for a single interval.

    Mode 1 & Mode 2: Full weight (Mode 1 uses Mode 2's SE as proxy)
    Mode 3: Penalized by R² quality

    Parameters
    ----------
    mode1_cda : float
        CdA estimate from Mode 1 (simple average)
    mode2_cda : float
        CdA estimate from Mode 2 (dynamic + turns)
    mode2_se : float
        Standard error from Mode 2
    mode3_cda : float
        CdA estimate from Mode 3 (straights only)
    mode3_se : float
        Standard error from Mode 3
    r2 : float
        R² value from sine fit

    Returns
    -------
    dict
        CdA: combined CdA estimate
        SE: combined standard error
        weight_mode1: normalized weight for Mode 1
        weight_mode2: normalized weight for Mode 2
        weight_mode3: normalized weight for Mode 3
        r2_penalty: R² penalty applied to Mode 3
    """
    # Handle invalid inputs
    if not np.isfinite(mode2_se) or mode2_se <= 0:
        mode2_se = 0.01  # fallback
    if not np.isfinite(mode3_se) or mode3_se <= 0:
        mode3_se = 0.01  # fallback

    # Mode 2 base weight from SE
    w2_base = 1.0 / (mode2_se ** 2)

    # Mode 1 gets equal weight to Mode 2
    w1_base = w2_base

    # Mode 3 with R² penalty
    w3_base = 1.0 / (mode3_se ** 2)
    r2_penalty = r2_weight(r2, threshold=0.2)
    w3_adjusted = w3_base * r2_penalty

    # Normalize
    w_total = w1_base + w2_base + w3_adjusted
    w1 = w1_base / w_total
    w2 = w2_base / w_total
    w3 = w3_adjusted / w_total

    # Weighted estimate
    cda_interval = w1 * mode1_cda + w2 * mode2_cda + w3 * mode3_cda
    se_interval = np.sqrt(1.0 / w_total)

    return {
        'CdA': round(cda_interval, 4),
        'SE': round(se_interval, 5),
        'weight_mode1': round(w1, 3),
        'weight_mode2': round(w2, 3),
        'weight_mode3': round(w3, 3),
        'r2_penalty': round(r2_penalty, 3),
    }


def compute_final_cda(interval_results):
    """Combine all interval CdAs using inverse-variance weighting.

    Parameters
    ----------
    interval_results : list of dict
        Each dict must have 'CdA' and 'SE' keys from compute_interval_cda

    Returns
    -------
    dict
        CdA_final: final combined CdA estimate
        SE_final: final combined standard error
        CI_95: 95% confidence interval [lower, upper]
        interval_weights: normalized weights for each interval
        n_intervals: number of intervals combined
    """
    cdas = np.array([r['CdA'] for r in interval_results])
    ses = np.array([r['SE'] for r in interval_results])

    # Handle edge cases
    if len(cdas) == 0:
        return {
            'CdA_final': np.nan,
            'SE_final': np.nan,
            'CI_95': [np.nan, np.nan],
            'interval_weights': [],
            'n_intervals': 0,
        }

    weights = 1.0 / (ses ** 2)
    weights_norm = weights / np.sum(weights)

    final_cda = np.sum(weights_norm * cdas)
    final_se = np.sqrt(1.0 / np.sum(weights))

    return {
        'CdA_final': round(final_cda, 4),
        'SE_final': round(final_se, 5),
        'CI_95': [round(final_cda - 1.96 * final_se, 4),
                  round(final_cda + 1.96 * final_se, 4)],
        'interval_weights': [round(w, 3) for w in weights_norm],
        'n_intervals': len(cdas),
    }


def estimate_cda_simple_average(v_mps, P_watts, rho, m_kg, Crr, eta,
                                 phi_rad=None, g=9.80665):
    """Mode 1: CdA from mean values with averaged Crr_eff.

    Uses averaged effective rolling resistance that accounts for lean in turns:
    Crr_eff = Crr / cos(phi) per sample, then averaged.
    """
    v_mean = np.mean(v_mps)
    P_mean = np.mean(P_watts)
    P_wheel = P_mean * eta

    # Use averaged Crr_eff (accounts for lean in turns)
    if phi_rad is not None:
        phi_rad = np.asarray(phi_rad, dtype=float)
        cos_phi = np.cos(phi_rad)
        cos_phi = np.maximum(cos_phi, 1e-6)
        Crr_eff = Crr / cos_phi
        Crr_avg = np.mean(Crr_eff)
    else:
        Crr_avg = Crr

    P_rolling = m_kg * g * Crr_avg * v_mean
    aero_factor = 0.5 * rho * v_mean ** 3

    if aero_factor < 1e-12:
        return {"CdA_point": np.nan}

    cda = (P_wheel - P_rolling) / aero_factor
    return {"CdA_point": round(float(cda), 4)}


def estimate_cda_with_error_bars(
    v_mps,
    P_watts,
    rho=1.21,
    m_kg=82.0,
    Crr=0.004,
    eta=0.98,
    g=9.80665,
    use_lean_rr=False,
    turn_selector=None,
    R_turn_m=23.0,
    N_mc=5000,
    sigma_v_mps=None,
    sigma_P_watts=None,
    sigma_rho=None,
    sigma_m=None,
    sigma_Crr=None,
    sigma_eta=None,
    seed=0,
    ci=0.95,
    return_samples=False,
    # New parameters for dynamic model
    phi_rad=None,
    dt=1.0,
    use_lean_cda=False,
    use_accel=False,
    sigma_R_turn=None,
    include_measurement_uncertainty_in_ci=False,
):
    """Estimate CdA from speed & power with optional Monte Carlo uncertainty propagation.

    New modes (backward-compatible defaults keep old behaviour):
      use_lean_cda : apply cos(phi) projection to aero term
      use_accel    : subtract acceleration power P_accel = m * dv/dt * v
      phi_rad      : lean angle array from signal_processing
      dt           : sample interval for dv/dt
      sigma_R_turn : turn radius uncertainty for MC
    """
    v = np.asarray(v_mps, dtype=float)
    P = np.asarray(P_watts, dtype=float)

    if v.shape != P.shape:
        raise ValueError("v_mps and P_watts must have same shape")

    if turn_selector is not None:
        turn_selector = np.asarray(turn_selector, dtype=bool)
        if turn_selector.shape != v.shape:
            raise ValueError("turn_selector must match v_mps/P_watts shape")
    else:
        turn_selector = np.zeros_like(v, dtype=bool)

    if phi_rad is not None:
        phi_rad = np.asarray(phi_rad, dtype=float)
    else:
        phi_rad = np.zeros_like(v, dtype=float)

    def _sigma_arr(s, like):
        if s is None:
            return None
        s = np.asarray(s, dtype=float)
        if s.ndim == 0:
            return np.full_like(like, s, dtype=float)
        if s.shape != like.shape:
            raise ValueError("Sigma array must match v/P shape or be scalar")
        return s

    sv = _sigma_arr(sigma_v_mps, v)
    sP = _sigma_arr(sigma_P_watts, P)
    srho = float(sigma_rho) if sigma_rho is not None else None
    sm = float(sigma_m) if sigma_m is not None else None
    sCrr = float(sigma_Crr) if sigma_Crr is not None else None
    seta = float(sigma_eta) if sigma_eta is not None else None
    sR = float(sigma_R_turn) if sigma_R_turn is not None else None

    rng = np.random.default_rng(seed)

    def _solve_cda(v_, P_, rho_, m_, Crr_, eta_, phi_=None):
        """Core CdA solver using least squares.

        Updated methodology (from CdA_Calculation_Walkthrough_2.ipynb):
        - Mode 2: NO cos(phi) on aero term, P_accel = 0, keep Crr/cos(phi)
        - Mode 3: Standard calculation, no corrections
        """
        v_ = np.asarray(v_, dtype=float)
        P_ = np.asarray(P_, dtype=float)
        if phi_ is None:
            phi_ = np.zeros_like(v_)

        # Rolling resistance with optional lean correction
        if use_lean_rr:
            cos_phi = np.cos(phi_)
            cos_phi = np.maximum(cos_phi, 1e-6)
            Crr_eff = Crr_ / cos_phi
        else:
            Crr_eff = Crr_

        P_wheel = P_ * eta_
        rr_power = (m_ * g * Crr_eff) * v_

        # Acceleration power - DISABLED per revised methodology
        # P_accel = 0 for all modes (acceleration term adds noise, not signal)
        P_accel = np.zeros_like(v_)

        # Aero factor - NO cos(phi) projection per revised methodology
        # The lean angle correction on CdA was found to be incorrect
        aero_factor = 0.5 * rho_ * (v_ ** 3)

        y = P_wheel - rr_power - P_accel

        mask = np.isfinite(aero_factor) & (np.abs(aero_factor) > 1e-12) & np.isfinite(y)
        if not np.any(mask):
            return np.nan, np.nan, np.zeros(3)

        af = aero_factor[mask]
        yy = y[mask]

        cda = float(np.dot(af, yy) / np.dot(af, af))

        if af.size > 1:
            r = yy - af * cda
            sigma2 = np.dot(r, r) / (len(af) - 1)
            se = np.sqrt(sigma2 / np.dot(af, af))
        else:
            se = np.nan

        # Power breakdown (means over valid points): P_aero, P_rolling, P_drivetrain
        # P_drivetrain = power lost in drivetrain = P * (1 - eta)
        P_aero_arr = af * cda
        P_drivetrain = P_ * (1.0 - eta_)
        breakdown = np.array([
            float(np.mean(P_aero_arr)),
            float(np.mean(rr_power[mask])),
            float(np.mean(P_drivetrain[mask])),
        ])

        return cda, se, breakdown

    cda_point, cda_se, breakdown = _solve_cda(v, P, rho, m_kg, Crr, eta, phi_=phi_rad)

    # 95% CI: z = 1.96 (two-tailed normal)
    z = 1.96

    if not include_measurement_uncertainty_in_ci:
        # CI from regression standard error only
        se_used = cda_se if np.isfinite(cda_se) else np.nan
        lo = cda_point - z * se_used if np.isfinite(se_used) else np.nan
        hi = cda_point + z * se_used if np.isfinite(se_used) else np.nan
        samples = np.array([], dtype=float)
        neg_count = 0
    else:
        # Monte Carlo: combine regression SE and measurement uncertainty in quadrature
        samples = np.empty(N_mc, dtype=float)
        neg_count = 0

        for i in range(N_mc):
            v_i = v.copy()
            P_i = P.copy()

            if sv is not None:
                v_i = v_i + rng.normal(0.0, sv)
            if sP is not None:
                P_i = P_i + rng.normal(0.0, sP)

            rho_i = rho if srho is None else (rho + rng.normal(0.0, srho))
            m_i = m_kg if sm is None else (m_kg + rng.normal(0.0, sm))
            Crr_i = Crr if sCrr is None else (Crr + rng.normal(0.0, sCrr))
            eta_i = eta if seta is None else (eta + rng.normal(0.0, seta))

            # Recompute phi from perturbed v and optionally perturbed R
            if use_lean_cda or use_lean_rr:
                R_i = R_turn_m if sR is None else max(R_turn_m + rng.normal(0.0, sR), 1.0)
                phi_i = np.zeros_like(v_i)
                if np.any(turn_selector) and R_i > 0:
                    phi_i[turn_selector] = np.arctan(
                        (v_i[turn_selector] ** 2) / (g * R_i)
                    )
            else:
                phi_i = np.zeros_like(v_i)

            cda_i = _solve_cda(v_i, P_i, rho_i, m_i, Crr_i, eta_i, phi_=phi_i)[0]

            if np.isfinite(cda_i) and cda_i < 0:
                neg_count += 1
                samples[i] = np.nan
            else:
                samples[i] = cda_i

        samples = samples[np.isfinite(samples)]
        std_mc = np.std(samples) if samples.size else np.nan
        se_used = (
            np.sqrt(cda_se**2 + std_mc**2)
            if np.isfinite(cda_se) and np.isfinite(std_mc)
            else (cda_se if np.isfinite(cda_se) else std_mc)
        )
        lo = cda_point - z * se_used if np.isfinite(se_used) else np.nan
        hi = cda_point + z * se_used if np.isfinite(se_used) else np.nan

    out = {
        "CdA_point": round(float(cda_point), 4),
        "CdA_point_se": round(float(cda_se), 5) if np.isfinite(cda_se) else np.nan,
        f"CdA_ci_{int(ci*100)}": [round(float(lo), 4), round(float(hi), 4)] if np.isfinite(lo) and np.isfinite(hi) else [np.nan, np.nan],
        "N_mc_effective": int(samples.size),
        "power_breakdown": {
            "P_aero_mean": round(float(breakdown[0]), 2),
            "P_rolling_mean": round(float(breakdown[1]), 2),
            "P_drivetrain_mean": round(float(breakdown[2]), 2),
        },
    }

    if neg_count > 0:
        out["mc_negative_cda_warning"] = neg_count

    if return_samples and samples.size:
        out["CdA_samples"] = samples

    return out
