"""
Velodrome CdA Analyzer - Main Application
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    .main-header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    .main-header-logo {
        height: 80px;
        width: auto;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def show_logo_header():
    """Display the logo and app title header."""
    logo_path = Path(__file__).parent / "assets" / "Soudal-quickstep-logo.png"

    if logo_path.exists():
        import base64
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(f'''
            <div class="main-header-container">
                <img src="data:image/png;base64,{logo_data}" class="main-header-logo" alt="Soudal Quick-Step">
                <h1 class="main-header">Velodrome CdA Analyzer</h1>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header-container"><h1 class="main-header">Velodrome CdA Analyzer</h1></div>', unsafe_allow_html=True)


def check_login():
    """Show login form if not authenticated. Returns True if user may access the app."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    # Support both Streamlit Cloud secrets and local .env
    expected_user = st.secrets.get("VELODROME_APP_USERNAME") if "VELODROME_APP_USERNAME" in st.secrets else os.environ.get("VELODROME_APP_USERNAME")
    expected_pass = st.secrets.get("VELODROME_APP_PASSWORD") if "VELODROME_APP_PASSWORD" in st.secrets else os.environ.get("VELODROME_APP_PASSWORD")

    if not expected_user or not expected_pass:
        st.warning(
            "Login is not configured. Set environment variables "
            "**VELODROME_APP_USERNAME** and **VELODROME_APP_PASSWORD** to enable authentication."
        )
        return True  # Allow access when not configured (e.g. local dev)

    # Show logo on login page
    show_logo_header()

    st.markdown('<div class="sub-header">Please sign in to continue</div>', unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if username == expected_user and password == expected_pass:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    return False  # Still on login screen; don't render main app


def main():
    if not check_login():
        return

    # Header with logo
    show_logo_header()
    st.markdown('<div class="sub-header">Professional Aerodynamic Testing Analysis Platform</div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'intervals' not in st.session_state:
        st.session_state.intervals = None
    if 'adjusted_intervals' not in st.session_state:
        st.session_state.adjusted_intervals = None
    if 'cda_results' not in st.session_state:
        st.session_state.cda_results = {}
    if 'final_cda' not in st.session_state:
        st.session_state.final_cda = {}
    if 'excluded_intervals' not in st.session_state:
        st.session_state.excluded_intervals = set()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis Step:",
        ["📁 Upload Data", "🔍 Interval Detection", "📊 CdA Analysis", "📈 Results Summary"]
    )

    # Show sign out if authentication is configured
    has_auth = ("VELODROME_APP_USERNAME" in st.secrets) or os.environ.get("VELODROME_APP_USERNAME")
    if has_auth:
        st.sidebar.divider()
        if st.sidebar.button("Sign out"):
            st.session_state.logged_in = False
            st.rerun()

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
    from utils.interval_detection import detect_intervals
    from utils.visualization import plot_intervals
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.header("🔍 Interval Detection")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a FIT file first!")
        return

    df = st.session_state.df

    # ── Detection Mode Selection ──────────────────────────────────────────
    detection_mode = st.radio(
        "Detection Mode",
        ["🔍 Automatic Detection", "✋ Manual Selection"],
        horizontal=True,
        help="Choose between automatic algorithm-based detection or manual interval selection"
    )

    if detection_mode == "🔍 Automatic Detection":
        # ══════════════════════════════════════════════════════════════════
        # AUTOMATIC DETECTION MODE
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Detection Parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_intervals = st.number_input("Number of Intervals", value=4, min_value=1, max_value=10)
            interval_distance = st.number_input("Interval Distance (m)", value=1250, min_value=500, max_value=2000)

        with col2:
            distance_tolerance = st.number_input("Distance Tolerance (m)", value=50, min_value=5, max_value=100)
            speed_tolerance = st.number_input("Speed Tolerance (km/h)", value=3.0, min_value=0.5, max_value=10.0)

        with col3:
            st.markdown("**Target Speeds (km/h)**")
            target_speeds_str = st.text_input(
                "Comma-separated",
                value="40, 45, 45, 50",
                help="Enter target speeds for each interval, separated by commas"
            )

        # Parse target speeds
        try:
            target_speeds = [float(s.strip()) for s in target_speeds_str.split(",")]
            if len(target_speeds) != n_intervals:
                st.warning(f"⚠️ Please enter exactly {n_intervals} target speeds (currently {len(target_speeds)})")
                target_speeds = target_speeds[:n_intervals] if len(target_speeds) > n_intervals else target_speeds + [45.0] * (n_intervals - len(target_speeds))
        except ValueError:
            st.error("Invalid target speeds format. Using defaults.")
            target_speeds = [45.0] * n_intervals

        if st.button("🔍 Detect Intervals", type="primary"):
            with st.spinner("Detecting intervals..."):
                intervals = detect_intervals(
                    df,
                    n_intervals=n_intervals,
                    interval_distance=interval_distance,
                    target_speeds_kmh=target_speeds,
                    distance_tolerance=distance_tolerance,
                    speed_tolerance=speed_tolerance
                )
                st.session_state.intervals = intervals
                st.session_state.interval_distance = interval_distance  # Store for CdA analysis
                # Initialize adjusted intervals as a copy
                st.session_state.adjusted_intervals = [
                    {
                        'interval_num': intv['interval_num'],
                        'start_idx': intv['start_idx'],
                        'end_idx': intv['end_idx'],
                        'target_speed_kmh': intv['target_speed_kmh'],
                    }
                    for intv in intervals
                ]
            st.success("✅ Interval detection complete!")

    else:
        # ══════════════════════════════════════════════════════════════════
        # MANUAL SELECTION MODE
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Manual Interval Selection")

        st.markdown("""
        Select intervals manually by specifying start and end distances.
        Use the plot below to identify the distance ranges for your intervals.
        """)

        # Show full data plot for reference
        st.markdown("#### Data Overview")
        fig_overview = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Speed over Distance', 'Power over Distance'),
            vertical_spacing=0.12,
            shared_xaxes=True
        )

        fig_overview.add_trace(
            go.Scatter(x=df['distance'], y=df['enhanced_speed']*3.6,
                      mode='lines', name='Speed (km/h)', line=dict(color='blue')),
            row=1, col=1
        )
        fig_overview.add_trace(
            go.Scatter(x=df['distance'], y=df['power'],
                      mode='lines', name='Power (W)', line=dict(color='red')),
            row=2, col=1
        )

        # Add shading for existing manual intervals if any
        if st.session_state.adjusted_intervals:
            colors = ['rgba(0,255,0,0.2)', 'rgba(255,165,0,0.2)', 'rgba(0,0,255,0.2)', 'rgba(255,0,255,0.2)',
                     'rgba(0,255,255,0.2)', 'rgba(255,255,0,0.2)', 'rgba(128,0,128,0.2)', 'rgba(0,128,128,0.2)']
            for i, intv in enumerate(st.session_state.adjusted_intervals):
                if intv['start_idx'] is not None:
                    d_start = df['distance'].iloc[intv['start_idx']]
                    d_end = df['distance'].iloc[intv['end_idx']] if intv['end_idx'] < len(df) else df['distance'].iloc[-1]
                    color = colors[i % len(colors)]
                    for row in [1, 2]:
                        fig_overview.add_vrect(
                            x0=d_start, x1=d_end,
                            fillcolor=color, opacity=0.5, line_width=0,
                            row=row, col=1,
                            annotation_text=f"Int {i+1}" if row == 1 else None,
                            annotation_position="top left" if row == 1 else None,
                        )

        fig_overview.update_xaxes(title_text="Distance (m)", row=2, col=1)
        fig_overview.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig_overview.update_yaxes(title_text="Power (W)", row=2, col=1)
        fig_overview.update_layout(height=500, showlegend=False)

        st.plotly_chart(fig_overview, use_container_width=True)

        # Show distance range info
        st.caption(f"Data range: {df['distance'].min():.0f} m to {df['distance'].max():.0f} m ({len(df)} samples)")

        st.divider()

        # ── Add New Interval ──────────────────────────────────────────────
        st.markdown("#### Add New Interval")

        col1, col2, col3 = st.columns(3)
        with col1:
            new_start_dist = st.number_input(
                "Start Distance (m)",
                min_value=float(df['distance'].min()),
                max_value=float(df['distance'].max()),
                value=float(df['distance'].min()),
                step=10.0,
                key="manual_start_dist"
            )
        with col2:
            new_end_dist = st.number_input(
                "End Distance (m)",
                min_value=float(df['distance'].min()),
                max_value=float(df['distance'].max()),
                value=min(float(df['distance'].min()) + 1250, float(df['distance'].max())),
                step=10.0,
                key="manual_end_dist"
            )
        with col3:
            new_target_speed = st.number_input(
                "Target Speed (km/h)",
                min_value=20.0,
                max_value=70.0,
                value=45.0,
                step=1.0,
                key="manual_target_speed"
            )

        # Find indices for the distances
        start_idx = (df['distance'] - new_start_dist).abs().idxmin()
        end_idx = (df['distance'] - new_end_dist).abs().idxmin()

        # Show preview stats
        if start_idx < end_idx:
            preview_v = df['enhanced_speed'].iloc[start_idx:end_idx]
            preview_p = df['power'].iloc[start_idx:end_idx]
            actual_dist = df['distance'].iloc[end_idx] - df['distance'].iloc[start_idx]

            st.markdown("**Preview:**")
            prev_cols = st.columns(4)
            prev_cols[0].metric("Distance", f"{actual_dist:.0f} m")
            prev_cols[1].metric("Avg Speed", f"{preview_v.mean()*3.6:.1f} km/h")
            prev_cols[2].metric("Avg Power", f"{preview_p.mean():.0f} W")
            prev_cols[3].metric("Samples", f"{len(preview_v)}")

        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("➕ Add Interval", type="primary"):
                if start_idx >= end_idx:
                    st.error("Start distance must be less than end distance!")
                else:
                    # Initialize if needed
                    if st.session_state.adjusted_intervals is None:
                        st.session_state.adjusted_intervals = []

                    # Add new interval
                    new_interval_num = len(st.session_state.adjusted_intervals) + 1
                    st.session_state.adjusted_intervals.append({
                        'interval_num': new_interval_num,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'target_speed_kmh': new_target_speed,
                    })

                    # Store interval distance (use actual distance of first interval)
                    if 'interval_distance' not in st.session_state or st.session_state.interval_distance is None:
                        st.session_state.interval_distance = int(actual_dist)

                    st.success(f"✅ Added Interval {new_interval_num}")
                    st.rerun()

        with col_clear:
            if st.button("🗑️ Clear All Intervals"):
                st.session_state.adjusted_intervals = []
                st.session_state.intervals = None
                st.rerun()

        # ── Remove Individual Intervals ───────────────────────────────────
        if st.session_state.adjusted_intervals:
            st.markdown("#### Remove Interval")
            remove_cols = st.columns(min(len(st.session_state.adjusted_intervals), 4))
            for i, intv in enumerate(st.session_state.adjusted_intervals):
                col_idx = i % len(remove_cols)
                with remove_cols[col_idx]:
                    if st.button(f"❌ Remove Int {i+1}", key=f"remove_int_{i}"):
                        st.session_state.adjusted_intervals.pop(i)
                        # Renumber remaining intervals
                        for j, remaining_intv in enumerate(st.session_state.adjusted_intervals):
                            remaining_intv['interval_num'] = j + 1
                        st.rerun()

    # ── Display and Adjust Intervals ──────────────────────────────────────
    if st.session_state.adjusted_intervals:
        adjusted_intervals = st.session_state.adjusted_intervals

        st.subheader("Selected Intervals")

        # Summary table
        summary_data = []
        for intv in adjusted_intervals:
            if intv['start_idx'] is not None:
                start_idx = intv['start_idx']
                end_idx = intv['end_idx']
                v_slice = df['enhanced_speed'].iloc[start_idx:end_idx]
                p_slice = df['power'].iloc[start_idx:end_idx]
                d_start = df['distance'].iloc[start_idx]
                d_end = df['distance'].iloc[end_idx] if end_idx < len(df) else df['distance'].iloc[-1]

                summary_data.append({
                    'Interval': f"Interval {intv['interval_num']}",
                    'Target Speed (km/h)': f"{intv['target_speed_kmh']:.1f}",
                    'Actual Avg Speed (km/h)': f"{v_slice.mean()*3.6:.1f}",
                    'Distance (m)': f"{d_end - d_start:.1f}",
                    'Avg Power (W)': f"{p_slice.mean():.0f}",
                    'Start Index': start_idx,
                    'End Index': end_idx
                })
            else:
                summary_data.append({
                    'Interval': f"Interval {intv['interval_num']}",
                    'Target Speed (km/h)': f"{intv['target_speed_kmh']:.1f}",
                    'Actual Avg Speed (km/h)': 'Not Found',
                    'Distance (m)': '-',
                    'Avg Power (W)': '-',
                    'Start Index': '-',
                    'End Index': '-'
                })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # ── Interactive Interval Adjustment ───────────────────────────────
        st.subheader("🎚️ Adjust Interval Boundaries")

        st.markdown("""
        Fine-tune interval boundaries by sliding them left or right.
        The buttons shift the **entire interval** (both start and end) by the specified number of samples.
        """)

        # Interval selector
        valid_intervals = [i for i, intv in enumerate(adjusted_intervals) if intv['start_idx'] is not None]
        if valid_intervals:
            selected_idx = st.selectbox(
                "Select Interval to Adjust",
                valid_intervals,
                format_func=lambda x: f"Interval {adjusted_intervals[x]['interval_num']}"
            )

            intv = adjusted_intervals[selected_idx]
            start_idx = intv['start_idx']
            end_idx = intv['end_idx']
            interval_len = end_idx - start_idx

            # Move buttons
            st.markdown("**Shift Interval:**")
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

            def move_interval(step):
                new_start = max(0, start_idx + step)
                new_end = new_start + interval_len
                if new_end > len(df):
                    new_end = len(df)
                    new_start = new_end - interval_len
                if new_start < 0:
                    new_start = 0
                    new_end = interval_len
                intv['start_idx'] = new_start
                intv['end_idx'] = new_end

            with col1:
                if st.button("⏪ -10", key="move_m10"):
                    move_interval(-10)
                    st.rerun()
            with col2:
                if st.button("◀ -1", key="move_m1"):
                    move_interval(-1)
                    st.rerun()
            with col3:
                st.markdown(f"**Current: [{start_idx} - {end_idx}]**")
            with col4:
                if st.button("+1 ▶", key="move_p1"):
                    move_interval(1)
                    st.rerun()
            with col5:
                if st.button("+10 ⏩", key="move_p10"):
                    move_interval(10)
                    st.rerun()

            # Show interval stats
            v_slice = df['enhanced_speed'].iloc[start_idx:end_idx]
            p_slice = df['power'].iloc[start_idx:end_idx]
            d_start = df['distance'].iloc[start_idx]
            d_end = df['distance'].iloc[end_idx] if end_idx < len(df) else df['distance'].iloc[-1]

            stat_cols = st.columns(4)
            stat_cols[0].metric("Distance", f"{d_end - d_start:.1f} m")
            stat_cols[1].metric("Avg Speed", f"{v_slice.mean()*3.6:.2f} km/h")
            stat_cols[2].metric("Avg Power", f"{p_slice.mean():.1f} W")
            stat_cols[3].metric("Samples", f"{len(v_slice)}")

            # Plot selected interval with context
            margin = 200
            plot_start = max(0, start_idx - margin)
            plot_end = min(len(df), end_idx + margin)

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Speed', 'Power'),
                vertical_spacing=0.1,
                shared_xaxes=True
            )

            # Context (gray)
            fig.add_trace(
                go.Scatter(
                    x=df['distance'].iloc[plot_start:plot_end],
                    y=df['enhanced_speed'].iloc[plot_start:plot_end] * 3.6,
                    mode='lines', line=dict(color='gray', width=1),
                    name='Context', showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['distance'].iloc[plot_start:plot_end],
                    y=df['power'].iloc[plot_start:plot_end],
                    mode='lines', line=dict(color='gray', width=1),
                    name='Context', showlegend=False
                ),
                row=2, col=1
            )

            # Selected interval (colored)
            fig.add_trace(
                go.Scatter(
                    x=df['distance'].iloc[start_idx:end_idx],
                    y=df['enhanced_speed'].iloc[start_idx:end_idx] * 3.6,
                    mode='lines', line=dict(color='blue', width=2),
                    name=f'Interval {intv["interval_num"]}'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['distance'].iloc[start_idx:end_idx],
                    y=df['power'].iloc[start_idx:end_idx],
                    mode='lines', line=dict(color='red', width=2),
                    name=f'Interval {intv["interval_num"]}', showlegend=False
                ),
                row=2, col=1
            )

            # Boundary lines
            for row in [1, 2]:
                fig.add_vline(x=d_start, line=dict(color='green', width=2, dash='dash'), row=row, col=1)
                fig.add_vline(x=d_end, line=dict(color='red', width=2, dash='dash'), row=row, col=1)

            fig.update_xaxes(title_text="Distance (m)", row=2, col=1)
            fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
            fig.update_yaxes(title_text="Power (W)", row=2, col=1)
            fig.update_layout(height=500, showlegend=True)

            st.plotly_chart(fig, use_container_width=True)

        # ── Full Visualization ────────────────────────────────────────────
        st.subheader("All Intervals Overview")
        fig = plot_intervals(df, [
            {
                'start_idx': intv['start_idx'],
                'end_idx': intv['end_idx'],
                'target_speed': intv['target_speed_kmh'] / 3.6,
                'actual_avg_speed': df['enhanced_speed'].iloc[intv['start_idx']:intv['end_idx']].mean() if intv['start_idx'] is not None else None,
                'actual_distance': (df['distance'].iloc[intv['end_idx']] - df['distance'].iloc[intv['start_idx']]) if intv['start_idx'] is not None else None,
                'mse': 0,
            }
            for intv in adjusted_intervals
        ])
        st.plotly_chart(fig, use_container_width=True)

        st.info("👉 Proceed to 'CdA Analysis' when intervals are correctly positioned.")

def show_cda_analysis_page():
    from utils.cda_estimation import (
        estimate_cda_with_error_bars,
        estimate_cda_simple_average,
        compute_interval_cda,
        compute_final_cda,
    )
    from utils.signal_processing import find_turn_selector

    st.header("📊 CdA Analysis — 3-Mode Comparison")

    if st.session_state.df is None or st.session_state.adjusted_intervals is None:
        st.warning("⚠️ Please complete previous steps first!")
        return

    df = st.session_state.df
    intervals = st.session_state.adjusted_intervals

    # ── Physical Parameters ──────────────────────────────────────────────
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

    with st.expander("⚙️ Advanced Settings"):
        st.markdown("**Measurement Uncertainties (1σ)**")
        col1, col2 = st.columns(2)
        with col1:
            sigma_v = st.number_input("Speed (m/s)", value=0.1, min_value=0.0, max_value=1.0, format="%.2f")
            sigma_P = st.number_input("Power (W)", value=3.0, min_value=0.0, max_value=20.0, format="%.1f")
        with col2:
            sigma_rho = st.number_input("Air Density (kg/m³)", value=0.05, min_value=0.0, max_value=0.1, format="%.3f")
            sigma_m = st.number_input("Mass (kg)", value=1.0, min_value=0.0, max_value=5.0, format="%.1f")

        st.markdown("**Confidence interval**")
        include_meas_uncertainty_in_ci = st.checkbox(
            "Measurement errors taken into account",
            value=False,
            help="If checked, 95% CI combines regression standard error and measurement uncertainty (from Monte Carlo) in quadrature. If unchecked, CI uses only the regression standard error.",
        )

    # ── Run 3-mode analysis ──────────────────────────────────────────────
    if st.button("🚀 Calculate CdA (3 Modes)", type="primary"):
        results = {}
        N_mc = 5000
        ci = 0.95

        common_mc_kwargs = dict(
            rho=rho, m_kg=m_kg, Crr=Crr, eta=eta,
            N_mc=N_mc, ci=ci,
            sigma_v_mps=sigma_v if sigma_v > 0 else None,
            sigma_P_watts=sigma_P if sigma_P > 0 else None,
            sigma_rho=sigma_rho if sigma_rho > 0 else None,
            sigma_m=sigma_m if sigma_m > 0 else None,
            return_samples=True,
            include_measurement_uncertainty_in_ci=include_meas_uncertainty_in_ci,
        )

        for i, interval in enumerate(intervals, 1):
            if interval['start_idx'] is None:
                continue

            with st.spinner(f"Analyzing Interval {i} (3 modes)..."):
                interval_df = df.iloc[interval['start_idx']:interval['end_idx']].copy()
                v_mps = interval_df['enhanced_speed'].values
                P_watts = interval_df['power'].values

                # ── Calculate per-interval target_periods ─────────────
                d_start = df['distance'].iloc[interval['start_idx']]
                d_end = df['distance'].iloc[interval['end_idx']] if interval['end_idx'] < len(df) else df['distance'].iloc[-1]
                actual_distance = d_end - d_start
                target_periods = max(2, round(actual_distance / 125))

                # ── Turn detection (shared by all modes) ─────────────
                turn_selector = None
                phi_rad = None
                r_squared = 0.0
                fit_curve = None
                x_fit = None
                try:
                    x_fit, fit_curve, turn_selector, phi_rad, r_squared = \
                        find_turn_selector(v_mps, target_periods=target_periods, R_turn_m=R_turn)
                except Exception as e:
                    st.warning(f"Could not detect turns for interval {i}: {e}")
                    turn_selector = np.zeros(len(v_mps), dtype=bool)
                    phi_rad = np.zeros(len(v_mps), dtype=float)

                # ── Mode 1: Simple Average (with averaged Crr_eff) ───
                mode1 = estimate_cda_simple_average(
                    v_mps, P_watts, rho, m_kg, Crr, eta,
                    phi_rad=phi_rad  # NEW: pass phi_rad for averaged Crr_eff
                )
                mode1["n_points"] = 1  # single-point from averages

                # ── Mode 2: Dynamic with Turns ───────────────────────
                # Updated methodology:
                # - use_lean_rr=True: Keep Crr/cos(phi) correction
                # - use_lean_cda=False: NO cos(phi) on aero term (internally disabled)
                # - use_accel=False: P_accel = 0 (internally disabled)
                mode2 = estimate_cda_with_error_bars(
                    v_mps=v_mps,
                    P_watts=P_watts,
                    use_lean_rr=True,    # Keep Crr/cos(phi) correction
                    use_lean_cda=False,  # No cos(phi) on aero (ignored internally)
                    use_accel=False,     # P_accel = 0 (ignored internally)
                    turn_selector=turn_selector,
                    phi_rad=phi_rad,
                    R_turn_m=R_turn,
                    **common_mc_kwargs,
                )
                mode2["n_points"] = int(len(v_mps))

                # ── Mode 3: Straights Only ───────────────────────────
                straight_mask = ~turn_selector
                v_straight = v_mps[straight_mask]
                P_straight = P_watts[straight_mask]

                if len(v_straight) > 1:
                    mode3 = estimate_cda_with_error_bars(
                        v_mps=v_straight,
                        P_watts=P_straight,
                        use_lean_rr=False,
                        use_lean_cda=False,
                        use_accel=False,
                        turn_selector=None,
                        R_turn_m=R_turn,
                        **common_mc_kwargs,
                    )
                else:
                    mode3 = {
                        "CdA_point": np.nan,
                        "CdA_point_se": np.nan,
                        "N_mc_effective": 0,
                        "power_breakdown": {"P_aero_mean": 0, "P_rolling_mean": 0, "P_drivetrain_mean": 0},
                    }
                mode3["n_points"] = int(np.sum(straight_mask))

                # ── Compute combined interval CdA ─────────────────────
                mode1_cda = mode1['CdA_point']
                mode2_cda = mode2['CdA_point']
                mode2_se = mode2.get('CdA_point_se', 0.01)
                mode3_cda = mode3['CdA_point']
                mode3_se = mode3.get('CdA_point_se', 0.01)

                # Handle NaN values
                if not np.isfinite(mode1_cda):
                    mode1_cda = mode2_cda  # fallback to mode2
                if not np.isfinite(mode3_cda):
                    mode3_cda = mode2_cda  # fallback to mode2
                    mode3_se = mode2_se * 2  # increase SE for fallback

                combined = compute_interval_cda(
                    mode1_cda=mode1_cda,
                    mode2_cda=mode2_cda,
                    mode2_se=mode2_se,
                    mode3_cda=mode3_cda,
                    mode3_se=mode3_se,
                    r2=r_squared,
                )

                # ── Store per-interval ───────────────────────────────
                results[f"Interval {i}"] = {
                    "mode1": mode1,
                    "mode2": mode2,
                    "mode3": mode3,
                    "combined": combined,
                    "turn_info": {
                        "r_squared": round(r_squared, 3),
                        "n_turn": int(np.sum(turn_selector)),
                        "n_straight": int(np.sum(~turn_selector)),
                        "mean_lean_deg": round(float(np.degrees(np.mean(phi_rad[turn_selector]))) if np.any(turn_selector) else 0, 2),
                        "max_lean_deg": round(float(np.degrees(np.max(phi_rad[turn_selector]))) if np.any(turn_selector) else 0, 2),
                    },
                    "fit_curve": fit_curve,
                    "x_fit": x_fit,
                    "v_mps": v_mps,
                    "turn_selector": turn_selector,
                }

        # ── Compute Final CdA from all intervals ──────────────────
        interval_combined_results = [
            res["combined"] for res in results.values()
            if np.isfinite(res["combined"]["CdA"])
        ]
        final_cda_result = compute_final_cda(interval_combined_results)
        st.session_state.final_cda = final_cda_result

        st.session_state.cda_results = results
        st.success("✅ 3-mode CdA analysis complete!")

    # ── Display results ──────────────────────────────────────────────────
    if st.session_state.cda_results:
        results = st.session_state.cda_results
        adjusted_intervals = st.session_state.get('adjusted_intervals', [])

        # ── Interval Exclusion Checkboxes ─────────────────────────────
        st.subheader("Exclude Intervals")
        st.markdown("Check intervals to **exclude** from final CdA calculation (set weight to 0):")

        # Create columns for checkboxes
        n_intervals = len(results)
        cols = st.columns(min(n_intervals, 4))

        excluded = set()
        for i, (name, res) in enumerate(results.items()):
            # Get target speed for this interval
            speed_kmh = adjusted_intervals[i].get('target_speed_kmh', 0) if i < len(adjusted_intervals) else 0

            col_idx = i % len(cols)
            with cols[col_idx]:
                # Check if this interval was previously excluded
                default_excluded = i in st.session_state.excluded_intervals
                is_excluded = st.checkbox(
                    f"Interval {i+1} ({speed_kmh:.0f} km/h)",
                    value=default_excluded,
                    key=f"exclude_interval_{i}"
                )
                if is_excluded:
                    excluded.add(i)

        # Update session state
        st.session_state.excluded_intervals = excluded

        # Recompute final CdA with exclusions
        interval_combined_results = []
        for i, (name, res) in enumerate(results.items()):
            if i not in excluded and np.isfinite(res["combined"]["CdA"]):
                interval_combined_results.append(res["combined"])

        final_cda = compute_final_cda(interval_combined_results)
        st.session_state.final_cda = final_cda

        if excluded:
            st.info(f"Excluding {len(excluded)} interval(s) from final calculation.")

        st.divider()

        # ── Final CdA Summary (top of results) ───────────────────────
        if final_cda and np.isfinite(final_cda.get('CdA_final', np.nan)):
            st.subheader("Final CdA Estimate")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Final CdA",
                f"{final_cda['CdA_final']:.4f} m²",
                help="Inverse-variance weighted combination of all intervals"
            )
            col2.metric(
                "Standard Error",
                f"{final_cda['SE_final']:.5f}",
            )
            col3.metric(
                "95% CI",
                f"[{final_cda['CI_95'][0]:.3f}, {final_cda['CI_95'][1]:.3f}]",
            )
            col4.metric(
                "Intervals Used",
                final_cda['n_intervals'],
            )

            # Show interval weights
            with st.expander("Interval Weight Breakdown"):
                weight_data = []
                # Map weights to included intervals only
                included_idx = 0
                for i, (name, res) in enumerate(results.items()):
                    comb = res.get("combined", {})
                    speed_kmh = adjusted_intervals[i].get('target_speed_kmh', 0) if i < len(adjusted_intervals) else 0

                    if i in excluded:
                        interval_weight = 0.0
                        status = "EXCLUDED"
                    else:
                        interval_weight = final_cda['interval_weights'][included_idx] if included_idx < len(final_cda['interval_weights']) else 0
                        included_idx += 1
                        status = "Included"

                    weight_data.append({
                        "Interval": name,
                        "Speed (km/h)": f"{speed_kmh:.0f}",
                        "Status": status,
                        "Combined CdA": comb.get('CdA', np.nan),
                        "Combined SE": comb.get('SE', np.nan),
                        "Interval Weight": f"{interval_weight:.1%}",
                        "Mode 1 Weight": f"{comb.get('weight_mode1', 0):.1%}",
                        "Mode 2 Weight": f"{comb.get('weight_mode2', 0):.1%}",
                        "Mode 3 Weight": f"{comb.get('weight_mode3', 0):.1%}",
                    })
                st.dataframe(pd.DataFrame(weight_data), use_container_width=True)

                st.markdown("""
                **Weight Explanation:**
                - **Interval Weight:** Contribution to final CdA (based on inverse-variance, 0% if excluded)
                - **Mode Weights:** Within-interval contribution from each mode
                """)

            st.divider()

        # ── 3.3  Per-interval 3-column comparison table ──────────────
        st.subheader("CdA Results — 3-Mode Comparison")

        st.markdown("""
        **Methodology (from CdA_Calculation_Walkthrough_2.ipynb):**
        - **Mode 1 (Simple Average):** Uses averaged values of Power & Speed
        - **Mode 2 (Dynamic + Turns):** All points, with the Turn correction
        - **Mode 3 (Straights Only):** Standard calculation on straight sections only
        """)

        for i, (interval_name, res) in enumerate(results.items()):
            speed_kmh = adjusted_intervals[i].get('target_speed_kmh', 0) if i < len(adjusted_intervals) else 0
            is_excluded = i in excluded

            # Header with exclusion indicator
            if is_excluded:
                st.markdown(f"#### ~~{interval_name}~~ ({speed_kmh:.0f} km/h) — EXCLUDED")
            else:
                st.markdown(f"#### {interval_name} ({speed_kmh:.0f} km/h)")

            # Show fit quality warning if R² is low
            r2 = res["turn_info"]["r_squared"]
            if r2 < 0.2:
                st.warning(f"⚠️ Poor sine fit (R² = {r2:.3f}) — turn detection may be unreliable")
            elif r2 < 0.4:
                st.info(f"ℹ️ Moderate sine fit (R² = {r2:.3f})")
            else:
                st.info(f"🟢 Good sine fit (R² = {r2:.3f})")

            m1, m2, m3 = res["mode1"], res["mode2"], res["mode3"]
            comb = res.get("combined", {})

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.markdown("**Simple Average**")
                st.metric("CdA (m²)", f"{m1['CdA_point']:.4f}" if np.isfinite(m1['CdA_point']) else "—")
                st.caption("Std Error: —")
                st.caption(f"Weight: {comb.get('weight_mode1', 0):.1%}")

            with col_b:
                st.markdown("**Dynamic + Turns**")
                st.metric("CdA (m²)", f"{m2['CdA_point']:.4f}" if np.isfinite(m2['CdA_point']) else "—")
                se2 = m2.get('CdA_point_se', np.nan)
                st.caption(f"Std Error: {se2:.5f}" if np.isfinite(se2) else "Std Error: —")
                st.caption(f"Weight: {comb.get('weight_mode2', 0):.1%}")

            with col_c:
                st.markdown("**Straights Only**")
                st.metric("CdA (m²)", f"{m3['CdA_point']:.4f}" if np.isfinite(m3['CdA_point']) else "—")
                se3 = m3.get('CdA_point_se', np.nan)
                st.caption(f"Std Error: {se3:.5f}" if np.isfinite(se3) else "Std Error: —")
                st.caption(f"Weight: {comb.get('weight_mode3', 0):.1%}")

            with col_d:
                st.markdown("**Combined**")
                comb_cda = comb.get('CdA', np.nan)
                st.metric("CdA (m²)", f"{comb_cda:.4f}" if np.isfinite(comb_cda) else "—")
                comb_se = comb.get('SE', np.nan)
                st.caption(f"Std Error: {comb_se:.5f}" if np.isfinite(comb_se) else "Std Error: —")
                st.caption("EXCLUDED" if is_excluded else "Merging of 3 modes")

            st.divider()

        # ── 3.4  Comparison bar chart ────────────────────────────────
        st.subheader("CdA Comparison Chart")
        import plotly.graph_objects as go

        interval_names = list(results.keys())

        cda_m1 = [results[n]["mode1"]["CdA_point"] for n in interval_names]
        cda_m2 = [results[n]["mode2"]["CdA_point"] for n in interval_names]
        cda_m3 = [results[n]["mode3"]["CdA_point"] for n in interval_names]

        # Error bars from 95% CI (fixed; not shown in UI)
        ci_key = "CdA_ci_95"
        def _err_bars(mode_key):
            upper, lower = [], []
            for n in interval_names:
                m = results[n][mode_key]
                ci_vals = m.get(ci_key, [np.nan, np.nan])
                pt = m["CdA_point"]
                upper.append(ci_vals[1] - pt if np.isfinite(ci_vals[1]) else 0)
                lower.append(pt - ci_vals[0] if np.isfinite(ci_vals[0]) else 0)
            return upper, lower

        err2_up, err2_lo = _err_bars("mode2")
        err3_up, err3_lo = _err_bars("mode3")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Simple Average", x=interval_names, y=cda_m1,
            marker_color="#636EFA",
        ))
        fig.add_trace(go.Bar(
            name="Dynamic + Turns", x=interval_names, y=cda_m2,
            marker_color="#EF553B",
            error_y=dict(type='data', symmetric=False, array=err2_up, arrayminus=err2_lo),
        ))
        fig.add_trace(go.Bar(
            name="Straights Only", x=interval_names, y=cda_m3,
            marker_color="#00CC96",
            error_y=dict(type='data', symmetric=False, array=err3_up, arrayminus=err3_lo),
        ))
        fig.update_layout(
            barmode='group',
            title="CdA — 3 Modes (95% CI)",
            yaxis_title="CdA (m²)",
            height=500,
        )


        st.plotly_chart(fig, use_container_width=True)

        # ── 3.5  Turn visualization (expandable) ────────────────────
        for interval_name, res in results.items():
            with st.expander(f"🔄 Turn Detection — {interval_name}"):
                ti = res["turn_info"]
                v_mps = res["v_mps"]
                ts = res["turn_selector"]
                x_fit = res["x_fit"]
                fit_curve = res["fit_curve"]

                # Metrics row
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("R²", f"{ti['r_squared']:.3f}")
                mc2.metric("Turn samples", ti['n_turn'])
                mc3.metric("Straight samples", ti['n_straight'])
                mc4.metric("Mean lean (°)", f"{ti['mean_lean_deg']:.1f}")
                mc5.metric("Max lean (°)", f"{ti['max_lean_deg']:.1f}")

                # Fit quality warnings
                if ti['r_squared'] < 0.2:
                    st.warning(f"⚠️ **Poor fit quality (R² = {ti['r_squared']:.3f} < 0.2)** — The rider may not have been really smooth in this interval.")
                elif ti['r_squared'] < 0.4:
                    st.info(f"ℹ️ **Moderate fit quality (R² = {ti['r_squared']:.3f})** — Turn detection results are okay and respresent an OK smoothness for the rider")
                else:
                    st.info(f"🟢 **Good sine fit (R² = {r2:.3f})")

                if x_fit is not None and fit_curve is not None:
                    fig_turn = go.Figure()
                    fig_turn.add_trace(go.Scatter(
                        x=x_fit, y=v_mps, mode='lines', name='Speed',
                        line=dict(color='blue', width=1),
                    ))
                    fig_turn.add_trace(go.Scatter(
                        x=x_fit, y=fit_curve, mode='lines', name='Sine fit',
                        line=dict(color='red', dash='dash', width=2),
                    ))
                    # Shade turn regions
                    turn_starts = np.where(np.diff(ts.astype(int)) == 1)[0]
                    turn_ends = np.where(np.diff(ts.astype(int)) == -1)[0]
                    if ts[0]:
                        turn_starts = np.concatenate([[0], turn_starts])
                    if ts[-1]:
                        turn_ends = np.concatenate([turn_ends, [len(ts) - 1]])
                    for s, e in zip(turn_starts, turn_ends):
                        fig_turn.add_vrect(
                            x0=x_fit[s], x1=x_fit[min(e, len(x_fit) - 1)],
                            fillcolor="orange", opacity=0.15, line_width=0,
                        )
                    fig_turn.update_layout(
                        title="Speed Signal with Turn Detection",
                        xaxis_title="Sample", yaxis_title="Speed (m/s)",
                        height=350, showlegend=True,
                    )
                    st.plotly_chart(fig_turn, use_container_width=True)

        # ── 3.6  Power breakdown (expandable) ───────────────────────
        for interval_name, res in results.items():
            with st.expander(f"⚡ Power Breakdown — {interval_name}"):
                pb = res["mode2"].get("power_breakdown", {})
                if pb:
                    fig_pow = go.Figure(go.Bar(
                        x=["P_aero", "P_rolling", "P_drivetrain"],
                        y=[pb.get("P_aero_mean", 0), pb.get("P_rolling_mean", 0),
                           pb.get("P_drivetrain_mean", 0)],
                        marker_color=["#636EFA", "#EF553B", "#AB63FA"],
                    ))
                    fig_pow.update_layout(
                        title="Mean Power Components (Dynamic + Turns mode)",
                        yaxis_title="Power (W)", height=350,
                    )
                    st.plotly_chart(fig_pow, use_container_width=True)
                else:
                    st.info("No power breakdown available.")

        st.info("👉 View 'Results Summary' for export options.")

def show_results_summary_page():
    from utils.cda_estimation import compute_interval_cda, compute_final_cda

    st.header("📈 Results Summary")

    if not st.session_state.cda_results:
        st.warning("⚠️ Please complete CdA analysis first!")
        return

    results = st.session_state.cda_results
    final_cda = st.session_state.get('final_cda', {})

    # ── Final CdA (primary result) ────────────────────────────────────
    st.subheader("Final CdA Estimate")

    if final_cda and np.isfinite(final_cda.get('CdA_final', np.nan)):
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Final CdA",
                f"{final_cda['CdA_final']:.4f} m²",
                help="Inverse-variance weighted combination of all intervals"
            )

        with col2:
            st.metric(
                "95% Confidence Interval",
                f"[{final_cda['CI_95'][0]:.3f}, {final_cda['CI_95'][1]:.3f}]",
            )

        st.info(f"Based on {final_cda['n_intervals']} intervals")
    else:
        st.warning("Final CdA not available. Please run CdA analysis.")

    st.divider()

    # ── CdA vs Speed Graph ────────────────────────────────────────────
    st.subheader("CdA vs Speed")

    # Get target speeds from adjusted_intervals and exclusions
    adjusted_intervals = st.session_state.get('adjusted_intervals', [])
    excluded = st.session_state.get('excluded_intervals', set())

    # Build data: speed -> list of (CdA, SE) tuples (excluding excluded intervals)
    speed_to_cda = {}
    for i, (name, res) in enumerate(results.items()):
        # Skip excluded intervals
        if i in excluded:
            continue

        comb = res.get("combined", {})
        cda_val = comb.get("CdA", np.nan)
        se_val = comb.get("SE", np.nan)

        # Get target speed for this interval
        if i < len(adjusted_intervals):
            speed_kmh = adjusted_intervals[i].get('target_speed_kmh', np.nan)
        else:
            speed_kmh = np.nan

        if np.isfinite(cda_val) and np.isfinite(speed_kmh) and np.isfinite(se_val):
            if speed_kmh not in speed_to_cda:
                speed_to_cda[speed_kmh] = []
            speed_to_cda[speed_kmh].append((cda_val, se_val))

    # Compute weighted average for each speed
    plot_speeds = []
    plot_cdas = []
    plot_ses = []
    plot_counts = []

    for speed in sorted(speed_to_cda.keys()):
        entries = speed_to_cda[speed]
        if len(entries) == 1:
            # Single interval at this speed
            plot_speeds.append(speed)
            plot_cdas.append(entries[0][0])
            plot_ses.append(entries[0][1])
            plot_counts.append(1)
        else:
            # Multiple intervals: inverse-variance weighted average
            cdas = np.array([e[0] for e in entries])
            ses = np.array([e[1] for e in entries])
            weights = 1.0 / (ses ** 2)
            weights_norm = weights / np.sum(weights)
            weighted_cda = np.sum(weights_norm * cdas)
            combined_se = np.sqrt(1.0 / np.sum(weights))

            plot_speeds.append(speed)
            plot_cdas.append(weighted_cda)
            plot_ses.append(combined_se)
            plot_counts.append(len(entries))

    if plot_speeds:
        import plotly.graph_objects as go

        # Calculate error bars (95% CI)
        error_upper = [1.96 * se for se in plot_ses]
        error_lower = [1.96 * se for se in plot_ses]

        # Create hover text
        hover_texts = [
            f"Speed: {s:.0f} km/h<br>CdA: {c:.4f} m²<br>SE: {se:.5f}<br>95% CI: [{c-1.96*se:.4f}, {c+1.96*se:.4f}]<br>Intervals: {n}"
            for s, c, se, n in zip(plot_speeds, plot_cdas, plot_ses, plot_counts)
        ]

        fig = go.Figure()

        # Add scatter with error bars
        fig.add_trace(go.Scatter(
            x=plot_speeds,
            y=plot_cdas,
            mode='markers+lines',
            marker=dict(size=12, color='#1f77b4'),
            line=dict(width=2, color='#1f77b4'),
            error_y=dict(
                type='data',
                symmetric=False,
                array=error_upper,
                arrayminus=error_lower,
                color='#1f77b4',
                thickness=1.5,
                width=6,
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            name='CdA',
        ))

        # Add horizontal line for final CdA
        if final_cda and np.isfinite(final_cda.get('CdA_final', np.nan)):
            fig.add_hline(
                y=final_cda['CdA_final'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Final CdA: {final_cda['CdA_final']:.4f}",
                annotation_position="right",
            )

        fig.update_layout(
            title="CdA vs Speed (Weighted Average)",
            xaxis_title="Speed (km/h)",
            yaxis_title="CdA (m²)",
            height=450,
            showlegend=False,
            xaxis=dict(dtick=5),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table below
        with st.expander("View Data Table"):
            table_data = []
            for s, c, se, n in zip(plot_speeds, plot_cdas, plot_ses, plot_counts):
                table_data.append({
                    "Speed (km/h)": f"{s:.0f}",
                    "CdA (m²)": f"{c:.4f}",
                    "SE": f"{se:.5f}",
                    "95% CI": f"[{c-1.96*se:.4f}, {c+1.96*se:.4f}]",
                    "N Intervals": n,
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.warning("No valid CdA data to plot.")

    # Show excluded intervals count if any
    if excluded:
        st.info(f"{len(excluded)} interval(s) excluded from analysis. Modify exclusions in the CdA Analysis page.")

    # Prepare comprehensive export data
    export_data = {
        "final_cda": final_cda,
        "cda_vs_speed": [
            {"speed_kmh": s, "cda": c, "se": se, "n_intervals": n}
            for s, c, se, n in zip(plot_speeds, plot_cdas, plot_ses, plot_counts)
        ] if plot_speeds else [],
        "intervals": {}
    }
    for i, (name, res) in enumerate(results.items()):
        speed_kmh = adjusted_intervals[i].get('target_speed_kmh') if i < len(adjusted_intervals) else None
        export_data["intervals"][name] = {
            "speed_kmh": speed_kmh,
            "excluded": i in excluded,
            "combined": res.get("combined", {}),
            "mode1_cda": res["mode1"].get("CdA_point"),
            "mode2_cda": res["mode2"].get("CdA_point"),
            "mode2_se": res["mode2"].get("CdA_point_se"),
            "mode3_cda": res["mode3"].get("CdA_point"),
            "mode3_se": res["mode3"].get("CdA_point_se"),
            "r_squared": res["turn_info"]["r_squared"],
        }

    # Summary CSV with combined results
    summary_rows = []
    included_idx = 0
    for i, (name, res) in enumerate(results.items()):
        comb = res.get("combined", {})
        speed_kmh = adjusted_intervals[i].get('target_speed_kmh') if i < len(adjusted_intervals) else None
        is_excluded = i in excluded

        if is_excluded:
            interval_weight = 0.0
        else:
            interval_weight = final_cda.get('interval_weights', [])[included_idx] if included_idx < len(final_cda.get('interval_weights', [])) else 0
            included_idx += 1

        summary_rows.append({
            "Interval": name,
            "Speed_kmh": speed_kmh,
            "Excluded": is_excluded,
            "Combined_CdA": comb.get("CdA"),
            "Combined_SE": comb.get("SE"),
            "Interval_Weight": interval_weight,
            "Mode1_CdA": res["mode1"].get("CdA_point"),
            "Mode1_Weight": comb.get("weight_mode1"),
            "Mode2_CdA": res["mode2"].get("CdA_point"),
            "Mode2_SE": res["mode2"].get("CdA_point_se"),
            "Mode2_Weight": comb.get("weight_mode2"),
            "Mode3_CdA": res["mode3"].get("CdA_point"),
            "Mode3_SE": res["mode3"].get("CdA_point_se"),
            "Mode3_Weight": comb.get("weight_mode3"),
        })

    # Add final row
    summary_rows.append({
        "Interval": "FINAL",
        "Speed_kmh": None,
        "Excluded": False,
        "Combined_CdA": final_cda.get("CdA_final"),
        "Combined_SE": final_cda.get("SE_final"),
        "Interval_Weight": 1.0,
        "Mode1_CdA": None,
        "Mode1_Weight": None,
        "Mode2_CdA": None,
        "Mode2_SE": None,
        "Mode2_Weight": None,
        "Mode3_CdA": None,
        "Mode3_SE": None,
        "Mode3_Weight": None,
    })

    export_df = pd.DataFrame(summary_rows)

    col1, col2 = st.columns(2)

    with col1:
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="cda_results.csv",
            mime="text/csv"
        )

    with col2:
        import json
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name="cda_results.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
