"""Visualization utilities"""

import plotly.graph_objects as go

def plot_intervals(df, intervals):
    """Create interactive plot showing detected intervals"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['distance'],
        y=df['enhanced_speed'] * 3.6,
        mode='lines',
        name='All Data',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    colors = ['red', 'green', 'blue', 'magenta']
    
    for i, interval in enumerate(intervals):
        if interval['start_idx'] is not None:
            interval_df = df.iloc[interval['start_idx']:interval['end_idx']]
            
            fig.add_trace(go.Scatter(
                x=interval_df['distance'],
                y=interval_df['enhanced_speed'] * 3.6,
                mode='lines',
                name=f"Interval {i+1} ({interval['target_speed']*3.6:.1f} km/h target)",
                line=dict(color=colors[i], width=2)
            ))
    
    fig.update_layout(
        title="Detected Intervals on Velodrome Test",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        height=500,
        hovermode='x unified'
    )
    
    return fig
