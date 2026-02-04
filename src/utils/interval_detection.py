"""Interval detection for velodrome testing — RMSE-prioritized algorithm"""

import numpy as np


def detect_intervals(df, n_intervals, interval_distance, target_speeds_kmh,
                     distance_tolerance=50, speed_tolerance=3.0):
    """Detect intervals by prioritizing RMSE (matches notebook methodology).

    Logic:
    - For each target speed, slide a window over distance.
    - Enforce a distance tolerance around the desired interval_distance.
    - Enforce non-overlapping intervals.
    - Among all candidates, pick the one with minimal MSE between speed and target_speed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'distance' and 'enhanced_speed' columns
    n_intervals : int
        Number of intervals to detect
    interval_distance : float
        Target distance for each interval (meters)
    target_speeds_kmh : list of float
        Target speeds for each interval (km/h). Must have length == n_intervals
    distance_tolerance : float
        Tolerance for distance matching (meters)
    speed_tolerance : float
        Speed tolerance for matching (km/h) - used for filtering candidates

    Returns
    -------
    list of dict
        List of interval dictionaries with keys:
        - interval_num, start_idx, end_idx, start_distance, end_distance
        - actual_distance, target_speed_kmh, actual_avg_speed_kmh
        - avg_power, rmse_mps
    """
    if len(target_speeds_kmh) != n_intervals:
        raise ValueError(f"target_speeds_kmh must have {n_intervals} values")

    target_speeds_mps = [s / 3.6 for s in target_speeds_kmh]

    distance = df["distance"].values
    speed = df["enhanced_speed"].values

    intervals = []
    used_indices = np.zeros(len(df), dtype=bool)

    for i, target_speed in enumerate(target_speeds_mps):
        best_start, best_end, best_mse = None, None, None

        for start in range(len(df)):
            if used_indices[start]:
                continue

            # Find end index where distance matches
            max_end = start
            while max_end < len(df) and distance[max_end] - distance[start] < interval_distance:
                max_end += 1
            if max_end == len(df):
                break

            actual_dist = distance[max_end] - distance[start]
            if abs(actual_dist - interval_distance) > distance_tolerance:
                continue

            # Skip windows that overlap existing intervals
            if np.any(used_indices[start:max_end]):
                continue

            # Compute MSE against target speed
            speed_window = speed[start:max_end]
            mse = np.mean((speed_window - target_speed) ** 2)

            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_start = start
                best_end = max_end

        if best_start is not None:
            avg_speed = speed[best_start:best_end].mean()
            actual_dist = distance[best_end] - distance[best_start]
            avg_power = df["power"].iloc[best_start:best_end].mean()

            intervals.append({
                "interval_num": i + 1,
                "start_idx": best_start,
                "end_idx": best_end,
                "start_distance": float(distance[best_start]),
                "end_distance": float(distance[best_end]),
                "actual_distance": float(actual_dist),
                "target_speed_kmh": target_speeds_kmh[i],
                "actual_avg_speed_kmh": float(avg_speed * 3.6),
                "avg_power": float(avg_power),
                "rmse_mps": float(np.sqrt(best_mse)),
                "target_speed": target_speed,  # m/s for compatibility
                "mse": best_mse,
                "actual_avg_speed": avg_speed,  # m/s for compatibility
            })
            used_indices[best_start:best_end] = True
        else:
            intervals.append({
                "interval_num": i + 1,
                "start_idx": None,
                "end_idx": None,
                "start_distance": None,
                "end_distance": None,
                "actual_distance": None,
                "target_speed_kmh": target_speeds_kmh[i],
                "actual_avg_speed_kmh": None,
                "avg_power": None,
                "rmse_mps": None,
                "target_speed": target_speed,
                "mse": None,
                "actual_avg_speed": None,
            })

    return intervals


def detect_interval_indices(df, target_speed, interval_distance=1250, tolerance=15):
    """Find best fitting interval by minimizing MSE to target_speed.

    Legacy function for single interval detection.
    """
    distances = df['distance'].to_numpy()
    best_mse = np.inf
    best_start = None
    best_end = None

    for start in range(len(distances)):
        max_end = start
        while max_end < len(distances) and distances[max_end] - distances[start] < interval_distance:
            max_end += 1
        if max_end == len(distances):
            break

        interval_actual_distance = distances[max_end] - distances[start]
        if abs(interval_actual_distance - interval_distance) > tolerance:
            continue

        speed_window = df['enhanced_speed'].iloc[start:max_end].to_numpy()
        mse = np.mean((speed_window - target_speed) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_start = start
            best_end = max_end

    return best_start, best_end, best_mse


def detect_4_intervals(df, interval_distance=1250, tolerance=15):
    """Find four non-overlapping intervals at target speeds.

    Legacy wrapper that calls detect_intervals with default parameters.
    """
    target_speeds_kmh = [45.0, 45.0, 40.0, 50.0]
    return detect_intervals(
        df,
        n_intervals=4,
        interval_distance=interval_distance,
        target_speeds_kmh=target_speeds_kmh,
        distance_tolerance=tolerance,
    )
