import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Patient Admission Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COLOR PALETTE
# ============================================================

NAVY = "#0B1F33"
NAVY_LIGHT = "#123A52"
TEAL = "#0F766E"
CYAN = "#14919B"
BACKGROUND = "#F4F8FB"
BORDER = "#DCE7ED"
TEXT = "#0B1F33"
MUTED = "#647789"


# ============================================================
# GLOBAL UI STYLING
# ============================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND};
        }}

        .main .block-container {{
            max-width: 1450px;
            padding: 2rem 2.5rem 3rem 2.5rem;
        }}

        /* Sidebar */

        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
        }}

        section[data-testid="stSidebar"] * {{
            color: #EAF4F8;
        }}

        section[data-testid="stSidebar"] .stRadio label {{
            color: #EAF4F8 !important;
        }}

        /* Hide Streamlit branding */

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        /* Hero */

        .hero-box {{
            background: linear-gradient(
                135deg,
                {NAVY} 0%,
                {NAVY_LIGHT} 55%,
                {TEAL} 100%
            );

            border-radius: 22px;
            padding: 32px 36px;
            margin-bottom: 28px;

            box-shadow:
                0 12px 30px rgba(11, 31, 51, 0.14);
        }}

        .hero-title {{
            color: white;
            font-size: 32px;
            font-weight: 750;
            letter-spacing: -0.5px;
        }}

        .hero-subtitle {{
            color: #D9EEF0;
            font-size: 15px;
            margin-top: 8px;
        }}

        .hero-badge {{
            display: inline-block;
            margin-top: 18px;
            padding: 7px 13px;

            border-radius: 999px;

            color: #EAF8F8;
            background: rgba(255,255,255,0.12);

            border: 1px solid rgba(255,255,255,0.18);

            font-size: 11px;
            font-weight: 700;
        }}

        /* Sidebar brand */

        .sidebar-brand {{
            padding: 12px 4px 20px 4px;
        }}

        .sidebar-icon {{
            font-size: 35px;
            margin-bottom: 8px;
        }}

        .sidebar-title {{
            color: white;
            font-size: 21px;
            font-weight: 750;
        }}

        .sidebar-subtitle {{
            color: #9FB6C5;
            font-size: 11px;
            margin-top: 5px;
        }}

        /* Section titles */

        .section-title {{
            color: {TEXT};
            font-size: 22px;
            font-weight: 750;
            margin-top: 25px;
            margin-bottom: 5px;
        }}

        .section-description {{
            color: {MUTED};
            font-size: 13px;
            margin-bottom: 18px;
        }}

        /* Cards */

        .metric-card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 5px 18px rgba(11,31,51,0.045);
        }}

        .card-title {{
            color: {TEXT};
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .card-text {{
            color: {MUTED};
            font-size: 13px;
            line-height: 1.65;
        }}

        /* Notice */

        .notice-box {{
            background: #FFF8E7;
            border-left: 4px solid #D69E2E;
            border-radius: 10px;
            padding: 14px 17px;
            color: #624D16;
            font-size: 13px;
            line-height: 1.6;
            margin: 15px 0;
        }}

        /* Status */

        .status-box {{
            background: #0C3F43;
            border: 1px solid #15595C;
            border-radius: 12px;
            padding: 14px;
            color: #EAF8F8;
            font-size: 13px;
            font-weight: 600;
        }}

        /* Footer */

        .footer-line {{
            border-top: 1px solid {BORDER};
            margin-top: 45px;
            padding-top: 18px;
        }}

        /* Buttons */

        .stButton > button {{
            border-radius: 9px;
        }}

        /* Tables */

        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}


        /* ========================================================
           ACCESSIBILITY / CONTRAST FIX
           ======================================================== */

        .stApp {{
            color-scheme: light;
            color: #0B1F33 !important;
        }}

        .main .block-container,
        .main .block-container p,
        .main .block-container span,
        .main .block-container label,
        .main .block-container li,
        .main .block-container td,
        .main .block-container th {{
            color: #0B1F33;
        }}

        .main .block-container h1,
        .main .block-container h2,
        .main .block-container h3,
        .main .block-container h4,
        .main .block-container h5,
        .main .block-container h6 {{
            color: #0B1F33 !important;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: #0B1F33 !important;
        }}

        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: #647789 !important;
        }}

        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricLabel"] {{
            color: #647789 !important;
        }}

        [data-testid="stMetricValue"] *,
        [data-testid="stMetricValue"] {{
            color: #0B1F33 !important;
        }}

        [data-testid="stMetricDelta"] *,
        [data-testid="stMetricDelta"] {{
            color: #0F766E !important;
        }}

        [data-testid="stWidgetLabel"] *,
        [data-testid="stWidgetLabel"] {{
            color: #0B1F33 !important;
        }}

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTimeInput input,
        .stSelectbox input,
        .stMultiSelect input {{
            color: #0B1F33 !important;
            background-color: #FFFFFF !important;
            caret-color: #0B1F33 !important;
        }}

        .stTextInput input::placeholder,
        .stNumberInput input::placeholder,
        .stDateInput input::placeholder,
        .stTimeInput input::placeholder,
        .stSelectbox input::placeholder,
        .stMultiSelect input::placeholder {{
            color: #647789 !important;
            opacity: 1 !important;
        }}

        [data-baseweb="select"] *,
        [data-baseweb="input"] * {{
            color: #0B1F33 !important;
        }}

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {{
            background-color: #FFFFFF !important;
        }}

        [data-baseweb="popover"] *,
        [role="listbox"] *,
        [role="option"] {{
            color: #0B1F33 !important;
            background-color: #FFFFFF !important;
        }}

        [data-testid="stRadio"] label *,
        [data-testid="stCheckbox"] label *,
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {{
            color: #0B1F33 !important;
        }}

        button[data-baseweb="tab"] *,
        button[data-baseweb="tab"] {{
            color: #0B1F33 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] * {{
            color: #0F766E !important;
        }}

        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary {{
            color: #0B1F33 !important;
        }}

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] div {{
            color: #0B1F33 !important;
        }}

        /* Preserve dark hero */
        .hero-box,
        .hero-box * {{
            color: white;
        }}

        .hero-subtitle {{
            color: #D9EEF0 !important;
        }}

        .hero-badge {{
            color: #EAF8F8 !important;
        }}

        /* Preserve dark sidebar */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] * {{
            color: #EAF4F8 !important;
        }}

        section[data-testid="stSidebar"] .sidebar-title {{
            color: white !important;
        }}

        section[data-testid="stSidebar"] .sidebar-subtitle {{
            color: #9FB6C5 !important;
        }}

        section[data-testid="stSidebar"] .status-box,
        section[data-testid="stSidebar"] .status-box * {{
            color: #EAF8F8 !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: #BFD2DC !important;
        }}

        .metric-card,
        .metric-card * {{
            color: #0B1F33;
        }}

        .card-text {{
            color: #647789 !important;
        }}

        .notice-box,
        .notice-box * {{
            color: #624D16 !important;
        }}

        .stButton > button {{
            color: #0B1F33 !important;
            background-color: #FFFFFF !important;
            border: 1px solid #DCE7ED !important;
        }}

        .stButton > button:hover {{
            color: #FFFFFF !important;
            background-color: #0F766E !important;
            border-color: #0F766E !important;
        }}

        div[data-testid="stDataFrame"] {{
            background-color: #FFFFFF !important;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            color: #0B1F33;
        }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DAILY_FILE = BASE_DIR / "data" / "daily_admissions.csv"
FORECAST_FILE = BASE_DIR / "outputs" / "7_day_forecast.csv"
COMPARISON_FILE = BASE_DIR / "outputs" / "model_comparison.csv"
TEST_FILE = BASE_DIR / "outputs" / "test_predictions.csv"
IMPORTANCE_FILE = BASE_DIR / "outputs" / "feature_importance.csv"
MODEL_FILE = BASE_DIR / "models" / "best_model.pkl"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_daily_data():

    df = pd.read_csv(
        DAILY_FILE,
        parse_dates=["date"]
    )

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_forecast_data():

    df = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["date"]
    )

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_model_comparison():

    return pd.read_csv(COMPARISON_FILE)


@st.cache_data
def load_test_predictions():

    df = pd.read_csv(
        TEST_FILE,
        parse_dates=["date"]
    )

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_feature_importance():

    if IMPORTANCE_FILE.exists():
        return pd.read_csv(IMPORTANCE_FILE)

    return None


@st.cache_resource
def load_trained_model():

    return joblib.load(MODEL_FILE)


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    daily_df = load_daily_data()
    forecast_df = load_forecast_data()
    comparison_df = load_model_comparison()
    test_df = load_test_predictions()
    importance_df = load_feature_importance()
    model_package = load_trained_model()

except Exception as error:

    st.error("Unable to load the project files.")

    st.code(str(error))

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

if isinstance(model_package, dict):

    model_name = model_package.get(
        "model_name",
        "Random Forest"
    )

else:

    model_name = "Random Forest"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-icon">🏥</div>
            <div class="sidebar-title">Hospital ML</div>
            <div class="sidebar-subtitle">
                Patient Admission Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("DASHBOARD")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📈 Forecast",
            "📊 Analytics",
            "🤖 Model Performance",
            "🛏️ Capacity Planning",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("MODEL STATUS")

    st.markdown(
        f"""
        <div class="status-box">
            ● {model_name} ready
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.caption("Python • Pandas • Scikit-learn")
    st.caption("Plotly • Streamlit • Joblib")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">
            🏥 Patient Admission Forecasting
        </div>
        <div class="hero-subtitle">
            Machine Learning dashboard for historical admission
            analysis and short-term demand forecasting
        </div>
        <div class="hero-badge">
            ● MACHINE LEARNING PROTOTYPE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Hospital Demand Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'A high-level view of recorded admissions and model-generated '
        'short-term forecasts.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    total_admissions = int(
        daily_df["admissions"].sum()
    )

    average_admissions = float(
        daily_df["admissions"].mean()
    )

    peak_admissions = int(
        daily_df["admissions"].max()
    )

    peak_date = daily_df.loc[
        daily_df["admissions"].idxmax(),
        "date"
    ]

    forecast_total = float(
        forecast_df["predicted_admissions"].sum()
    )

    forecast_average = float(
        forecast_df["predicted_admissions"].mean()
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            label="Recorded Admissions",
            value=f"{total_admissions:,}"
        )

        st.caption(
            "Total admissions in available records"
        )

    with c2:

        st.metric(
            label="Average / Observed Day",
            value=f"{average_admissions:.1f}"
        )

        st.caption(
            "Average across recorded admission dates"
        )

    with c3:

        st.metric(
            label="Peak Recorded",
            value=f"{peak_admissions}"
        )

        st.caption(
            peak_date.strftime("%d %b %Y")
        )

    with c4:

        st.metric(
            label="7-Day Forecast",
            value=f"{forecast_total:.1f}"
        )

        st.caption(
            f"Average {forecast_average:.1f} / day"
        )

    # --------------------------------------------------------
    # HISTORICAL CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Admission Demand Timeline</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Recorded admission demand across the available historical data.'
        '</div>',
        unsafe_allow_html=True
    )

    historical_fig = go.Figure()

    historical_fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["admissions"],
            mode="lines",
            name="Recorded Admissions",
            line=dict(
                color=TEAL,
                width=2.5
            ),
            fill="tozeroy",
            fillcolor="rgba(15,118,110,0.08)"
        )
    )

    historical_fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            title="Admissions",
            gridcolor="#E7EEF2"
        )
    )

    st.plotly_chart(
        historical_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # LOWER SECTION
    # --------------------------------------------------------

    left, right = st.columns(
        [1.4, 0.8]
    )

    with left:

        st.markdown(
            '<div class="section-title">7-Day Forecast</div>',
            unsafe_allow_html=True
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Bar(
                x=forecast_df["date"],
                y=forecast_df["predicted_admissions"],
                marker_color=TEAL,
                name="Predicted Admissions"
            )
        )

        forecast_fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            xaxis=dict(
                showgrid=False
            ),
            yaxis=dict(
                title="Predicted Admissions",
                gridcolor="#E7EEF2"
            )
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">System Snapshot</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            st.subheader("Model Status")

            st.success(
                f"{model_name} ready"
            )

            st.write(
                f"**Model:** {model_name}"
            )

            st.write(
                f"**Observed dates:** {len(daily_df):,}"
            )

            st.write(
                "**Forecast horizon:** 7 days"
            )

            st.write(
                "**Latest historical record:** "
                f"{daily_df['date'].max().strftime('%d %b %Y')}"
            )


# ============================================================
# FORECAST
# ============================================================

elif page == "📈 Forecast":

    st.markdown(
        '<div class="section-title">7-Day Admission Forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="section-description">'
        f'Forecast generated using the trained {model_name} model '
        f'from the historical admission records.'
        f'</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FORECAST KPIs
    # --------------------------------------------------------

    expected_total = float(
        forecast_df["predicted_admissions"].sum()
    )

    expected_average = float(
        forecast_df["predicted_admissions"].mean()
    )

    peak_index = forecast_df[
        "predicted_admissions"
    ].idxmax()

    forecast_peak = float(
        forecast_df.loc[
            peak_index,
            "predicted_admissions"
        ]
    )

    forecast_peak_date = forecast_df.loc[
        peak_index,
        "date"
    ]

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Expected Admissions",
            f"{expected_total:.1f}"
        )

    with b:

        st.metric(
            "Average / Day",
            f"{expected_average:.2f}"
        )

    with c:

        st.metric(
            "Peak Forecast",
            f"{forecast_peak:.2f}"
        )

        st.caption(
            forecast_peak_date.strftime("%d %b %Y")
        )

    # --------------------------------------------------------
    # HISTORICAL + FORECAST CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Historical vs Forecast Demand'
        '</div>',
        unsafe_allow_html=True
    )

    combined_fig = go.Figure()

    combined_fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["admissions"],
            mode="lines",
            name="Historical",
            line=dict(
                color="#8497A5",
                width=2
            )
        )
    )

    combined_fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_admissions"],
            mode="lines+markers",
            name="7-Day Forecast",
            line=dict(
                color=TEAL,
                width=4
            ),
            marker=dict(
                size=9
            )
        )
    )

    combined_fig.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            title="Admissions",
            gridcolor="#E7EEF2"
        )
    )

    st.plotly_chart(
        combined_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Forecast Details</div>',
        unsafe_allow_html=True
    )

    display_forecast = forecast_df.copy()

    display_forecast["date"] = (
        display_forecast["date"]
        .dt.strftime("%d %b %Y")
    )

    display_forecast[
        "predicted_admissions"
    ] = display_forecast[
        "predicted_admissions"
    ].round(2)

    display_forecast = display_forecast.rename(
        columns={
            "date": "Forecast Date",
            "predicted_admissions":
                "Predicted Admissions"
        }
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "These are model-generated estimates based on the "
        "historical dataset. They are not clinical predictions "
        "or real-time hospital operating instructions."
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">Admission Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Explore temporal patterns in the available admission records.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MONTHLY ADMISSIONS
    # --------------------------------------------------------

    monthly = (
        daily_df
        .set_index("date")
        .resample("ME")["admissions"]
        .sum()
        .reset_index()
    )

    monthly_fig = px.bar(
        monthly,
        x="date",
        y="admissions",
        title="Monthly Recorded Admissions"
    )

    monthly_fig.update_traces(
        marker_color=TEAL
    )

    monthly_fig.update_layout(
        template="plotly_white",
        height=430,
        xaxis_title="Month",
        yaxis_title="Admissions"
    )

    st.plotly_chart(
        monthly_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DAY OF WEEK
    # --------------------------------------------------------

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    day_of_week = (
        daily_df
        .groupby("day_of_week")["admissions"]
        .mean()
        .reset_index()
    )

    day_of_week["Day"] = (
        day_of_week["day_of_week"]
        .map(day_names)
    )

    dow_fig = px.bar(
        day_of_week,
        x="Day",
        y="admissions",
        title="Average Admissions by Day of Week"
    )

    dow_fig.update_traces(
        marker_color=CYAN
    )

    dow_fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Day",
        yaxis_title="Average Admissions"
    )

    st.plotly_chart(
        dow_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # YEARLY SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Yearly Summary</div>',
        unsafe_allow_html=True
    )

    yearly = (
        daily_df
        .groupby("year")["admissions"]
        .agg(
            Total="sum",
            Average="mean",
            Peak="max"
        )
        .reset_index()
    )

    st.dataframe(
        yearly.round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Comparison of four regression models using the time-based '
        'test set.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader("Selected Model")

        st.success(
            model_name
        )

        st.write(
            "The saved model was selected using the lowest RMSE "
            "among the evaluated models. MAE, RMSE and R² are "
            "shown together for transparent comparison."
        )

    # --------------------------------------------------------
    # MODEL TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Evaluation Metrics</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MAE + RMSE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        mae_fig = px.bar(
            comparison_df,
            x="Model",
            y="MAE",
            title="MAE — Lower is Better"
        )

        mae_fig.update_traces(
            marker_color=CYAN
        )

        mae_fig.update_layout(
            template="plotly_white",
            height=390,
            xaxis_title="",
            yaxis_title="MAE"
        )

        st.plotly_chart(
            mae_fig,
            use_container_width=True
        )

    with col2:

        rmse_fig = px.bar(
            comparison_df,
            x="Model",
            y="RMSE",
            title="RMSE — Lower is Better"
        )

        rmse_fig.update_traces(
            marker_color=TEAL
        )

        rmse_fig.update_layout(
            template="plotly_white",
            height=390,
            xaxis_title="",
            yaxis_title="RMSE"
        )

        st.plotly_chart(
            rmse_fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    r2_fig = px.bar(
        comparison_df,
        x="Model",
        y="R2",
        title="R² — Higher Indicates More Explained Variance"
    )

    r2_fig.update_traces(
        marker_color=NAVY
    )

    r2_fig.update_layout(
        template="plotly_white",
        height=390,
        xaxis_title="",
        yaxis_title="R²"
    )

    st.plotly_chart(
        r2_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if importance_df is not None:

        st.markdown(
            '<div class="section-title">Feature Importance</div>',
            unsafe_allow_html=True
        )

        importance_sorted = (
            importance_df
            .sort_values(
                "Importance",
                ascending=True
            )
        )

        importance_fig = px.bar(
            importance_sorted,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        importance_fig.update_traces(
            marker_color=TEAL
        )

        importance_fig.update_layout(
            template="plotly_white",
            height=500
        )

        st.plotly_chart(
            importance_fig,
            use_container_width=True
        )


# ============================================================
# CAPACITY PLANNING
# ============================================================

elif page == "🛏️ Capacity Planning":

    st.markdown(
        '<div class="section-title">Hospital Capacity Planning</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Illustrative capacity analysis using the model forecast.'
        '</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "Prototype only: this calculation is illustrative. "
        "It does not model discharge timing, length of stay, "
        "transfers, or real hospital occupancy dynamics."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    input1, input2 = st.columns(2)

    with input1:

        total_beds = st.number_input(
            "Total Hospital Beds",
            min_value=1,
            max_value=5000,
            value=100,
            step=1
        )

    with input2:

        occupied_beds = st.number_input(
            "Currently Occupied Beds",
            min_value=0,
            max_value=5000,
            value=70,
            step=1
        )

    # --------------------------------------------------------
    # CALCULATIONS
    # --------------------------------------------------------

    average_forecast = float(
        forecast_df["predicted_admissions"].mean()
    )

    projected_occupancy = (
        occupied_beds +
        average_forecast
    )

    projected_utilization = (
        projected_occupancy /
        total_beds
    ) * 100

    remaining_capacity = (
        total_beds -
        projected_occupancy
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        st.metric(
            "Forecast / Day",
            f"{average_forecast:.1f}"
        )

    with p2:

        st.metric(
            "Current Occupancy",
            f"{occupied_beds}"
        )

    with p3:

        st.metric(
            "Projected Occupancy",
            f"{projected_occupancy:.1f}"
        )

    with p4:

        st.metric(
            "Illustrative Utilization",
            f"{projected_utilization:.1f}%"
        )

    # --------------------------------------------------------
    # CAPACITY CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Capacity View</div>',
        unsafe_allow_html=True
    )

    used_capacity = min(
        projected_occupancy,
        total_beds
    )

    free_capacity = max(
        total_beds -
        projected_occupancy,
        0
    )

    capacity_fig = go.Figure()

    capacity_fig.add_trace(
        go.Bar(
            x=["Hospital Capacity"],
            y=[used_capacity],
            name="Projected Occupancy",
            marker_color=TEAL
        )
    )

    capacity_fig.add_trace(
        go.Bar(
            x=["Hospital Capacity"],
            y=[free_capacity],
            name="Remaining Capacity",
            marker_color="#DCE8ED"
        )
    )

    capacity_fig.update_layout(
        barmode="stack",
        template="plotly_white",
        height=420,
        yaxis_title="Beds"
    )

    st.plotly_chart(
        capacity_fig,
        use_container_width=True
    )

    if remaining_capacity >= 0:

        st.success(
            f"Illustrative remaining capacity: "
            f"{remaining_capacity:.1f} beds"
        )

    else:

        st.error(
            f"Illustrative projected demand exceeds the "
            f"entered capacity by {abs(remaining_capacity):.1f} beds."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Patient Admission Forecasting & Hospital Capacity Planning'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader(
            "Patient Admission Forecasting & Hospital Capacity Planning"
        )

        st.write(
            "This project demonstrates an end-to-end Machine "
            "Learning workflow for analysing historical hospital "
            "admission records and generating a short-term "
            "admission forecast."
        )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader("Machine Learning Pipeline")

    pipeline_steps = [
        "Historical Data",
        "Data Processing",
        "Feature Engineering",
        "Model Training",
        "Model Evaluation",
        "7-Day Forecast",
        "Streamlit Dashboard"
    ]

    pipeline_text = " → ".join(
        pipeline_steps
    )

    st.info(
        pipeline_text
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    st.subheader("Machine Learning Models")

    model_columns = st.columns(4)

    model_list = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ]

    for column, model in zip(
        model_columns,
        model_list
    ):

        with column:

            st.info(model)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("Evaluation Metrics")

    metric_columns = st.columns(3)

    with metric_columns[0]:

        st.metric(
            "MAE",
            "Mean Absolute Error"
        )

    with metric_columns[1]:

        st.metric(
            "RMSE",
            "Root Mean Squared Error"
        )

    with metric_columns[2]:

        st.metric(
            "R²",
            "Coefficient of Determination"
        )

    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    st.subheader("Technology Stack")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Plotly",
        "Streamlit",
        "Joblib"
    ]

    st.write(
        " • ".join(technologies)
    )

    # --------------------------------------------------------
    # DATASET INTERPRETATION
    # --------------------------------------------------------

    st.subheader("Dataset Interpretation")

    st.write(
        "The forecasting pipeline uses the admission dates "
        "actually present in the supplied dataset. Dates without "
        "recorded admissions were not automatically interpreted "
        "as zero admissions."
    )

    st.write(
        "Therefore, this application should be interpreted as "
        "a Machine Learning prototype based on the available "
        "historical records."
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.subheader("Important Limitations")

    st.warning(
        "This project has not been clinically validated. "
        "It should not be used as a substitute for clinical "
        "judgment, hospital operational planning, or medical "
        "decision-making."
    )

    st.write(
        "The capacity-planning page is an illustrative "
        "demonstration rather than a real hospital management "
        "system."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 Patient Admission Forecasting · Machine Learning Prototype"
)

st.caption(
    "Python · Pandas · Scikit-learn · Plotly · Streamlit"
)

st.caption(
    "Historical admission analysis and short-term forecasting"
)
