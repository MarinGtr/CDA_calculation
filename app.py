"""
Velodrome CdA Analyzer - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="Velodrome CdA Analyzer",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🚴 Velodrome CdA Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Aerodynamic Testing Analysis Platform</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'intervals' not in st.session_state:
        st.session_state.intervals = None
    if 'cda_results' not in st.session_state:
        st.session_state.cda_results = {}
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis Step:",
        ["📁 Upload Data", "🔍 Interval Detection", "📊 CdA Analysis", "📈 Results Summary"]
    )
    
    if page == "📁 Upload Data":
        show_upload_page()
    elif page == "🔍 Interval Detection":
        show_interval_detection_page()
    elif page == "📊 CdA Analysis":
        show_cda_analysis_page()
    elif page == "📈 Results Summary":
        show_results_summary_page()

def show_upload_page():
    from utils.fit_parser import parse_fit_file, validate_fit_data
    
    st.header("📁 Upload FIT File")
    
    uploaded_file = st.file_uploader(
        "Choose a FIT file from your velodrome test",
        type=['fit', 'FIT'],
        help="Upload a FIT file containing power and speed data"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Parsing FIT file..."):
                df = parse_fit_file(uploaded_file)
                
            is_valid, message = validate_fit_data(df)
            
            if is_valid:
                st.success(f"✅ File loaded! {len(df)} records found.")
                st.session_state.df = df
                
                st.subheader("Data Preview")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", len(df))
                col2.metric("Duration (min)", f"{len(df)/60:.1f}")
                col3.metric("Avg Speed (km/h)", f"{df['enhanced_speed'].mean()*3.6:.1f}")
                col4.metric("Avg Power (W)", f"{df['power'].mean():.0f}")
                
                st.subheader("Data Overview")
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Speed over Distance', 'Power over Distance'),
                    vertical_spacing=0.12
                )
                
                fig.add_trace(
                    go.Scatter(x=df['distance'], y=df['enhanced_speed']*3.6, 
                              mode='lines', name='Speed (km/h)', line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=df['distance'], y=df['power'], 
                              mode='lines', name='Power (W)', line=dict(color='red')),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Distance (m)", row=2, col=1)
                fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
                fig.update_yaxes(title_text="Power (W)", row=2, col=1)
                fig.update_layout(height=600, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Raw Data"):
                    st.dataframe(df.head(100))
                
                st.info("👉 Proceed to 'Interval Detection' in the sidebar.")
            else:
                st.error(f"❌ Data validation failed: {message}")
                
        except Exception as e:
            st.error(f"❌ Error parsing file: {str(e)}")
    else:
        st.info("👆 Upload a FIT file to begin analysis")

def show_interval_detection_page():
    from utils.interval_detection import detect_4_intervals
    from utils.visualization import plot_intervals
    
    st.header("🔍 Interval Detection")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a FIT file first!")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    Detects four 1250m intervals at target speeds:
    - Interval 1: ~45 km/h
    - Interval 2: ~45 km/h (different section)
    - Interval 3: ~40 km/h
    - Interval 4: ~50 km/h
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        interval_distance = st.number_input("Interval Distance (m)", value=1250, min_value=500, max_value=2000)
    with col2:
        tolerance = st.number_input("Distance Tolerance (m)", value=15, min_value=5, max_value=50)
    
    if st.button("🔍 Detect Intervals", type="primary"):
        with st.spinner("Detecting intervals..."):
            intervals = detect_4_intervals(df, interval_distance=interval_distance, tolerance=tolerance)
            st.session_state.intervals = intervals
        st.success("✅ Interval detection complete!")
    
    if st.session_state.intervals:
        intervals = st.session_state.intervals
        
        st.subheader("Detected Intervals")
        
        summary_data = []
        for i, interval in enumerate(intervals, 1):
            if interval['start_idx'] is not None:
                summary_data.append({
                    'Interval': f"Interval {i}",
                    'Target Speed (km/h)': f"{interval['target_speed']*3.6:.1f}",
                    'Actual Avg Speed (km/h)': f"{interval['actual_avg_speed']*3.6:.1f}",
                    'Distance (m)': f"{interval['actual_distance']:.1f}",
                    'MSE': f"{interval['mse']:.4f}",
                    'Start Index': interval['start_idx'],
                    'End Index': interval['end_idx']
                })
            else:
                summary_data.append({
                    'Interval': f"Interval {i}",
                    'Target Speed (km/h)': f"{interval['target_speed']*3.6:.1f}",
                    'Actual Avg Speed (km/h)': 'Not Found',
                    'Distance (m)': '-',
                    'MSE': '-',
                    'Start Index': '-',
                    'End Index': '-'
                })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        st.subheader("Interval Visualization")
        fig = plot_intervals(df, intervals)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("👉 Proceed to 'CdA Analysis'.")

def show_cda_analysis_page():
    from utils.cda_estimation import estimate_cda_with_error_bars
    from utils.signal_processing import find_turn_selector
    
    st.header("📊 CdA Analysis")
    
    if st.session_state.df is None or st.session_state.intervals is None:
        st.warning("⚠️ Please complete previous steps first!")
        return
    
    df = st.session_state.df
    intervals = st.session_state.intervals
    
    st.subheader("Physical Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Rider & Equipment**")
        m_kg = st.number_input("Total Mass (kg)", value=82.0, min_value=50.0, max_value=150.0)
        rho = st.number_input("Air Density (kg/m³)", value=1.21, min_value=1.0, max_value=1.3)
    
    with col2:
        st.markdown("**Rolling & Drivetrain**")
        Crr = st.number_input("Rolling Resistance", value=0.004, min_value=0.002, max_value=0.010, format="%.4f")
        eta = st.number_input("Drivetrain Efficiency", value=0.98, min_value=0.90, max_value=1.0, format="%.3f")
    
    with col3:
        st.markdown("**Velodrome**")
        R_turn = st.number_input("Turn Radius (m)", value=23.0, min_value=15.0, max_value=40.0)
        use_lean = st.checkbox("Apply Lean Correction", value=True)
    
    st.subheader("Measurement Uncertainties (1σ)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sigma_v = st.number_input("Speed (m/s)", value=0.1, min_value=0.0, max_value=1.0, format="%.2f")
    with col2:
        sigma_P = st.number_input("Power (W)", value=3.0, min_value=0.0, max_value=20.0, format="%.1f")
    with col3:
        sigma_rho = st.number_input("Density (kg/m³)", value=0.02, min_value=0.0, max_value=0.1, format="%.3f")
    with col4:
        sigma_m = st.number_input("Mass (kg)", value=0.5, min_value=0.0, max_value=5.0, format="%.1f")
    
    with st.expander("⚙️ Advanced Settings"):
        N_mc = st.number_input("Monte Carlo Samples", value=5000, min_value=1000, max_value=20000, step=1000)
        ci = st.slider("Confidence Interval", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        target_periods = st.number_input("Expected Periods", value=10, min_value=5, max_value=20)
    
    if st.button("🚀 Calculate CdA", type="primary"):
        results = {}
        
        for i, interval in enumerate(intervals, 1):
            if interval['start_idx'] is None:
                continue
            
            with st.spinner(f"Analyzing Interval {i}..."):
                interval_df = df.iloc[interval['start_idx']:interval['end_idx']].copy()
                v_mps = interval_df['enhanced_speed'].values
                P_watts = interval_df['power'].values
                
                turn_selector = None
                if use_lean:
                    try:
                        _, _, turn_selector = find_turn_selector(v_mps, target_periods=target_periods)
                    except Exception as e:
                        st.warning(f"Could not detect turns for interval {i}: {e}")
                
                result = estimate_cda_with_error_bars(
                    v_mps=v_mps,
                    P_watts=P_watts,
                    rho=rho,
                    m_kg=m_kg,
                    Crr=Crr,
                    eta=eta,
                    use_lean_rr=use_lean,
                    turn_selector=turn_selector,
                    R_turn_m=R_turn,
                    N_mc=N_mc,
                    sigma_v_mps=sigma_v if sigma_v > 0 else None,
                    sigma_P_watts=sigma_P if sigma_P > 0 else None,
                    sigma_rho=sigma_rho if sigma_rho > 0 else None,
                    sigma_m=sigma_m if sigma_m > 0 else None,
                    ci=ci,
                    return_samples=True
                )
                
                results[f"Interval {i}"] = result
        
        st.session_state.cda_results = results
        st.success("✅ CdA analysis complete!")
    
    if st.session_state.cda_results:
        st.subheader("CdA Results")
        
        results_data = []
        for interval_name, result in st.session_state.cda_results.items():
            ci_key = f"CdA_ci_{int(ci*100)}"
            results_data.append({
                'Interval': interval_name,
                'CdA (m²)': result['CdA_point'],
                'Std Error': result['CdA_point_se'],
                f'CI {int(ci*100)}% Lower': result[ci_key][0],
                f'CI {int(ci*100)}% Upper': result[ci_key][1],
            })
        
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for interval_name, result in st.session_state.cda_results.items():
            ci_key = f"CdA_ci_{int(ci*100)}"
            fig.add_trace(go.Bar(
                name=interval_name,
                x=[interval_name],
                y=[result['CdA_point']],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[result[ci_key][1] - result['CdA_point']],
                    arrayminus=[result['CdA_point'] - result[ci_key][0]]
                )
            ))
        
        fig.update_layout(
            title=f"CdA Estimates with {int(ci*100)}% Confidence Intervals",
            yaxis_title="CdA (m²)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("👉 View 'Results Summary' for export options.")

def show_results_summary_page():
    st.header("📈 Results Summary")
    
    if not st.session_state.cda_results:
        st.warning("⚠️ Please complete CdA analysis first!")
        return
    
    st.subheader("Overall Statistics")
    
    cda_values = [r['CdA_point'] for r in st.session_state.cda_results.values()]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean CdA", f"{np.mean(cda_values):.4f} m²")
    col2.metric("Std Dev", f"{np.std(cda_values):.4f} m²")
    col3.metric("Min CdA", f"{np.min(cda_values):.4f} m²")
    col4.metric("Max CdA", f"{np.max(cda_values):.4f} m²")
    
    st.subheader("Detailed Results")
    results_df = pd.DataFrame([
        {'Interval': name, **result}
        for name, result in st.session_state.cda_results.items()
    ])
    st.dataframe(results_df, use_container_width=True)
    
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cda_results.csv",
            mime="text/csv"
        )
    
    with col2:
        json_str = results_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name="cda_results.json",
            mime="application/json"
        )
    
    st.subheader("Reference Values")
    st.markdown("""
    - **Elite Track Cyclist**: 0.200 - 0.220 m²
    - **Good Amateur**: 0.230 - 0.260 m²
    - **Recreational**: 0.270 - 0.320 m²
    """)

if __name__ == "__main__":
    main()
