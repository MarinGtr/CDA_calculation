"""CdA estimation with Monte Carlo error propagation"""

import numpy as np

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
    return_samples=False
):
    """Estimate CdA from speed & power with Monte Carlo uncertainty propagation"""
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
    
    rng = np.random.default_rng(seed)
    
    def _solve_cda(v_, P_, rho_, m_, Crr_, eta_):
        """Core CdA solver using least squares"""
        v_ = np.asarray(v_, dtype=float)
        P_ = np.asarray(P_, dtype=float)
        
        if use_lean_rr:
            Crr_eff = np.full_like(v_, Crr_, dtype=float)
            if np.any(turn_selector):
                tan_phi = (v_**2) / (g * R_turn_m)
                cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
                Crr_eff[turn_selector] = Crr_ / cos_phi[turn_selector]
        else:
            Crr_eff = Crr_
        
        P_wheel = P_ * eta_
        aero_factor = 0.5 * rho_ * (v_**3)
        rr_power = (m_ * g * Crr_eff) * v_
        
        y = P_wheel - rr_power
        
        mask = np.isfinite(aero_factor) & (np.abs(aero_factor) > 1e-12) & np.isfinite(y)
        if not np.any(mask):
            return np.nan, np.nan
        
        af = aero_factor[mask]
        yy = y[mask]
        
        cda = float(np.dot(af, yy) / np.dot(af, af))
        
        if af.size > 1:
            r = yy - af * cda
            sigma2 = np.dot(r, r) / (len(af) - 1)
            se = np.sqrt(sigma2 / np.dot(af, af))
            return cda, se
        else:
            return cda, np.nan
    
    cda_point, cda_se = _solve_cda(v, P, rho, m_kg, Crr, eta)
    
    samples = np.empty(N_mc, dtype=float)
    
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
        
        samples[i] = _solve_cda(v_i, P_i, rho_i, m_i, Crr_i, eta_i)[0]
    
    samples = samples[np.isfinite(samples)]
    
    alpha = 1.0 - ci
    lo = np.quantile(samples, alpha / 2.0) if samples.size else np.nan
    hi = np.quantile(samples, 1.0 - alpha / 2.0) if samples.size else np.nan
    
    out = {
        "CdA_point": round(float(cda_point), 4),
        "CdA_point_se": round(float(cda_se), 5) if np.isfinite(cda_se) else np.nan,
        f"CdA_ci_{int(ci*100)}": [round(float(lo), 4), round(float(hi), 4)] if samples.size else [np.nan, np.nan],
        "N_mc_effective": int(samples.size),
    }
    
    if return_samples:
        out["CdA_samples"] = samples
    
    return out
