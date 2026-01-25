"""FIT file parsing utilities"""

import pandas as pd
from fitparse import FitFile
import io

def parse_fit_file(uploaded_file):
    """Parse FIT file and extract cycling data"""
    if isinstance(uploaded_file, str):
        fitfile = FitFile(uploaded_file)
    else:
        fitfile = FitFile(io.BytesIO(uploaded_file.read()))
    
    records = []
    for record in fitfile.get_messages("record"):
        record_data = {}
        for data in record:
            record_data[data.name] = data.value
        records.append(record_data)
    
    df = pd.DataFrame(records)
    
    required_cols = ['enhanced_speed', 'power', 'distance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.dropna(subset=required_cols)
    
    if df['enhanced_speed'].max() > 100:
        df['enhanced_speed'] = df['enhanced_speed'] / 1000
    
    return df

def validate_fit_data(df):
    """Validate FIT data for CdA analysis"""
    if len(df) < 100:
        return False, "Insufficient data points"
    if df['enhanced_speed'].max() < 5:
        return False, "Speed values too low"
    if df['power'].max() < 50:
        return False, "Power values too low"
    if df['distance'].max() < 1000:
        return False, "Total distance too short"
    return True, "Data validation passed"
