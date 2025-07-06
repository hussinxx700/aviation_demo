import streamlit as st
import pandas as pd
from processor import predict_and_explain

# -------------------------------------------------
# Page configuration – ONLY supported parameters
# -------------------------------------------------
st.set_page_config(
    page_title="Aviation Risk Assessment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Inject custom CSS – force white background in ANY theme
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary-color: #58a6ff;         /* soft aviation blue */
    --bg-color: #1e1e2f;              /* deep navy background */
    --sidebar-bg: #2c3e50;            /* dark slate for sidebar */
    --card-bg: #2e2e3d;               /* slightly lighter panel background */
    --border-color: #3c3c4c;          /* subtle gray for borders */
    --text-color: #ffffff;            /* bright white text */
    --subtle-text: #bbbbbb;           /* muted gray for less important text */
    --head-color: #f0f0f0;            /* light gray for headers */
    --radius: 10px;
    --shadow: 0 4px 12px rgba(0,0,0,0.3); /* deeper shadows in dark mode */
    --success: #4caf50;               /* rich green */
    --danger:  #ef5350;               /* muted red */
}}

/* ----- FORCE BACKGROUND TO WHITE EVEN IN DARK MODE ----- */
html, body, .stApp, .viewerArea, .block-container, .main {
    background-color: var(--bg-color) !important;
    color: var(--text-color);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color);
    padding: 1.5rem;
}
.sidebar-header {font-size:2.5rem;font-weight:650;color:var(--head-color);margin-bottom:1rem;}

/* Headers */
.main-header {text-align:center;font-size:2.6rem;font-weight:700;color:var(--head-color);margin-bottom:.4rem;}
.subtitle {text-align:center;font-size:1.2rem;color:var(--subtle-text);margin-bottom:2rem;}
.section-header {font-size:1.5rem;font-weight:600;color:var(--head-color);border-bottom:2px solid var(--border-color);margin:2rem 0 1rem;}

/* Cards */
.card {background:var(--card-bg);border:1px solid var(--border-color);border-radius:var(--radius);box-shadow:var(--shadow);padding:1.5rem;transition:box-shadow .2s;}
.card:hover {box-shadow:0 6px 12px rgba(0,0,0,0.12);}

/* Predictions */
.prediction-box {text-align:center;padding:1.8rem 1rem;}
.prediction-title {font-size:1rem;color:var(--subtle-text);margin-bottom:.4rem;}
.risk-percentage {font-size:3rem;font-weight:700;color:var(--primary-color);}  
.risk-class {font-size:1.6rem;font-weight:600;padding:.3rem 1rem;border-radius:6px;display:inline-block;margin-top:.3rem;}

/* Features */
.feature-item {display:flex;justify-content:space-between;align-items:center;padding:.8rem 1rem;margin-bottom:.8rem;}
.feature-name {font-weight:500;}
.shap-value  {font-family:'SF Mono',monospace;font-weight:600;}
.impact-increases {color:var(--danger);} .impact-decreases {color:var(--success);}

/* Info banner & footer */
.info-banner {background:var(--sidebar-bg);border-left:4px solid var(--primary-color);padding:1rem;border-radius:6px;margin-top:1rem;}
.footer-box  {text-align:center;color:var(--subtle-text);padding:1.5rem;margin-top:2rem;border-top:1px solid var(--border-color);}   
hr {border:none;height:1px;background:var(--border-color);margin:2.5rem 0;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar – sample selector
# -------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Input</div>", unsafe_allow_html=True)

    sample_options = {
        "Test Case 1": "sample_input_0.csv",
        "Test Case 2": "sample_input_1.csv",
        "Test Case 3": "sample_input_2.csv",
    }

    selected_label = st.radio(
        "Choose a test case for analysis:",
        list(sample_options.keys()),
        help="Select a sample flight scenario to analyse its risk.",
    )
    selected_file = sample_options[selected_label]

# -------------------------------------------------
# Main header
# -------------------------------------------------
st.markdown("<h1 class='main-header'>Aviation Incident Risk Assessment System (DEMO)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Each test case represents a unique flight scenario. The model analyses its data to predict incident risk.</p>", unsafe_allow_html=True)

# -------------------------------------------------
# Show analysis input
# -------------------------------------------------
st.markdown("<div class='section-header'>Input Sample</div>", unsafe_allow_html=True)
input_df = pd.read_csv(selected_file)
with st.container():
    df_t = input_df.T
    df_t.columns = ["Value"]
    df_t.index.name = "Feature"
    st.dataframe(df_t, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.markdown("<div class='section-header'>Risk Assessment Results</div>", unsafe_allow_html=True)
with st.spinner("Performing comprehensive risk analysis…"):
    result = predict_and_explain(selected_file)

col1, col2 = st.columns(2, gap="large")

with col1:
    pct = result["prediction_probability"] * 100
    st.markdown(f"""
    <div class='card prediction-box'>
        <div class='prediction-title'>Risk Probability</div>
        <div class='risk-percentage'>{pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    cls = result["prediction_label"]
    color = (
        "var(--success)" if cls == "No Incident" else
        "var(--danger)"
    )
    st.markdown(f"""
    <div class='card prediction-box'>
        <div class='prediction-title'>Risk Classification</div>
        <div class='risk-class' style='background-color:{color}20;color:{color};'>{cls}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# Key contributing factors
# -------------------------------------------------
st.markdown("<div class='section-header'>Key Contributing Factors</div>", unsafe_allow_html=True)
st.markdown("<div style='color:var(--subtle-text);margin-bottom:1rem;'>Features ranked by SHAP value:</div>", unsafe_allow_html=True)

for i, (_, shap_val, friendly) in enumerate(result["top_features"]):
    direction = "impact-increases" if shap_val > 0 else "impact-decreases"
    verb = "Increases" if shap_val > 0 else "Decreases"
    st.markdown(f"""
    <div class='card feature-item'>
        <div class='feature-name'>{i+1}. {friendly}</div>
        <div class='shap-value {direction}'>{verb} Risk (SHAP {shap_val:+.4f})</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer-box'>
    <strong>Note:</strong> This demo is developed as a proof-of-concept for academic or prototype use. It uses synthetic data and is intended for demonstration purposes only.
</div>
""", unsafe_allow_html=True)
