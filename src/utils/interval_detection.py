"""Interval detection for velodrome testing"""

import numpy as np

def detect_interval_indices(df, target_speed, interval_distance=1250, tolerance=15):
    """Find best fitting interval by minimizing MSE to target_speed"""
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
    """Find four non-overlapping intervals at target speeds"""
    used_indices = np.zeros(len(df), dtype=bool)
    results = []
    target_speeds = [45/3.6, 45/3.6, 40/3.6, 50/3.6]
    
    for target_speed in target_speeds:
        best_start, best_end, best_mse = None, None, None
        
        for start in range(len(df)):
            if used_indices[start]:
                continue
            
            distances = df['distance'].to_numpy()
            max_end = start
            
            while max_end < len(df) and distances[max_end] - distances[start] < interval_distance:
                max_end += 1
            
            if max_end == len(df):
                break
            
            if abs(distances[max_end] - distances[start] - interval_distance) > tolerance:
                continue
            
            if np.any(used_indices[start:max_end]):
                continue
            
            speed_window = df['enhanced_speed'].iloc[start:max_end].to_numpy()
            mse = np.mean((speed_window - target_speed) ** 2)
            
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_start = start
                best_end = max_end
        
        if best_start is not None:
            results.append({
                'start_idx': best_start,
                'end_idx': best_end,
                'target_speed': target_speed,
                'mse': best_mse,
                'actual_avg_speed': df['enhanced_speed'].iloc[best_start:best_end].mean(),
                'actual_distance': df['distance'].iloc[best_end] - df['distance'].iloc[best_start]
            })
            used_indices[best_start:best_end] = True
        else:
            results.append({
                'start_idx': None,
                'end_idx': None,
                'target_speed': target_speed,
                'mse': None,
                'actual_avg_speed': None,
                'actual_distance': None
            })
    
    return results
