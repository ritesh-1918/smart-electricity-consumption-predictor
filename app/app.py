"""
Volterra SaaS Platform
----------------------
Production energy forecast, carbon tracking, and feature attribution dashboard.
Provides clean UI layout metrics and analytics insights.
"""

# =====================================
# SYSTEM DEPENDENCIES & LIBRARIES
# =====================================
# Streamlit: A web-framework for Python to build interactive dashboards quickly.
# Pandas & NumPy: Structured data processing and numeric helpers.
# Pickle & OS: Save/load operations and system filepath access.
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# DASHBOARD PAGE CONFIGURATION
# =====================================
# st.set_page_config sets the browser window title, sidebar state, and uses a wide layout grid.
st.set_page_config(
    page_title="Volterra | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to remove leading spaces from multi-line HTML strings.
# This prevents Streamlit's markdown parser from rendering HTML as a raw text code block.
def clean_html(html_str):
    return "\n".join([line.strip() for line in html_str.split("\n")])

# =====================================
# PREMIUM DARK SAAS DESIGN THEME (CSS)
# =====================================
# Custom CSS stylesheets injected into the page to override default Streamlit themes.
# Styles the background to deep blue-black (#0B0E14), makes borders slate (#1E293B),
# and establishes the premium SaaS visual hierarchy for cards and dividers.
st.markdown("""
<style>
    /* Main container background overrides */
    .stApp {
        background-color: #0B0E14;
    }
    
    /* Premium KPI Cards */
    .kpi-container {
        background-color: #121620;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 16px 20px;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .kpi-trend {
        font-size: 12px;
        color: #10B981;
        font-weight: 500;
    }
    
    /* Status indicator banners */
    .status-banner {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 25px;
    }
    .status-text {
        font-size: 13px;
        color: #94A3B8;
        font-weight: 500;
    }
    
    /* Section dividers */
    .section-header {
        font-size: 16px;
        font-weight: 700;
        color: #F8FAFC;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOADING THE TRAINED MODEL (INFRASTRUCTURE)
# =====================================
# Path to the serialized model file.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")

# @st.cache_resource tells Streamlit to load the model file once and keep it in cache memory.
# This prevents reloading the model from disk on every page refresh or slider adjustment.
@st.cache_resource
def get_prediction_engine(path):
    # Instead of training the model on the fly (which takes time and computational resources),
    # we load our pre-compiled 'pickle' model. This contains the pre-learned mathematical formulas,
    # allowing us to generate predictions in milliseconds!
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None

# Instantiating our predictor engine.
engine = get_prediction_engine(MODEL_PATH)

# =====================================
# SIDEBAR CONTROL PANEL
# =====================================
st.sidebar.markdown("### ⚙️ Simulation Settings")
# Selectbox allows users to test preconfigured energy demand scenarios.
preset = st.sidebar.selectbox(
    "Select Load Profile Preset",
    options=["Manual Configuration", "Peak Demand Profile", "Eco Conservation Profile", "Baseline Utility Profile"]
)

# Preset configs mapping sets default values for inputs based on user's selection.
if preset == "Peak Demand Profile":
    val_temp = 42.0
    val_humidity = 70.0
    val_occupants = 6
    val_ac = 14.0
    val_appliance = 12.0
    val_day = "Weekend"
elif preset == "Eco Conservation Profile":
    val_temp = 24.0
    val_humidity = 50.0
    val_occupants = 3
    val_ac = 2.0
    val_appliance = 4.0
    val_day = "Weekday"
elif preset == "Baseline Utility Profile":
    val_temp = 20.0
    val_humidity = 40.0
    val_occupants = 2
    val_ac = 0.0
    val_appliance = 3.0
    val_day = "Weekday"
else:
    # Default settings
    val_temp = 26.0
    val_humidity = 55.0
    val_occupants = 4
    val_ac = 4.0
    val_appliance = 6.0
    val_day = "Weekday"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Pipeline Stats")
st.sidebar.markdown("""
*   **API Service**: `Online`
*   **Predictive Model**: `Linear Regression v1`
*   **Accuracy (R²)**: `94.95%`
*   **MAE**: `6.38 kWh`
""")

# Main Content Header (Hero Dashboard Section)
st.markdown("### ⚡ VOLTERRA | Energy Intelligence Engine")

# Real-time Status Banner
st.markdown("""
<div class="status-banner">
    <span class="status-text">🟢 Forecast Engine: Online &bull; Latency: 12ms &bull; Active profile: Simulated telemetry input</span>
</div>
""", unsafe_allow_html=True)

if engine is None:
    st.error("Error: Core prediction binary linear_regression_model.pkl is missing. Please run src/train.py to compile.")
else:
    # Split Simulator inputs (left column) and live predictions (right column)
    col_input, col_display = st.columns([1, 1])
    
    with col_input:
        st.markdown('<div class="section-header">Simulated Input Parameters</div>', unsafe_allow_html=True)
        
        # Grid of sliders to collect input features from the user.
        # Streamlit sliders capture real-time values for Temperature, Humidity, Occupancy, AC, and Appliance hours.
        inp_col1, inp_col2 = st.columns(2)
        with inp_col1:
            temperature = st.slider("Outdoor Temperature (°C)", 10.0, 50.0, val_temp, 0.5)
            humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, val_humidity, 1.0)
            occupants = st.number_input("Occupant Count", 1, 10, val_occupants, 1)
        with inp_col2:
            ac_hours = st.slider("AC Operating Hours (daily)", 0.0, 24.0, val_ac, 0.5)
            appliance_hours = st.slider("Appliance Operating Hours (daily)", 0.0, 24.0, val_appliance, 0.5)
            day_type = st.selectbox("Day Classification", ["Weekday", "Weekend"], index=0 if val_day == "Weekday" else 1)
            
    # Map day classification (Categorical Variable Encoding)
    # Weekday -> 0, Weekend -> 1 (identical mapping logic used during model training).
    day_encoded = 0 if day_type == "Weekday" else 1
    
    # =====================================
    # STREAMLIT PREDICTION PIPELINE
    # =====================================
    # Preparing Features for the Model:
    # Machine Learning models expect inputs in a specific structure, matching the exact format
    # they were trained on (columns, scale, order). Here, we take the user's slider/input choices
    # and organize them into a 1-row Pandas DataFrame to feed into the prediction engine.
    input_df = pd.DataFrame([{
        'Temperature': temperature,
        'Humidity': humidity,
        'Occupancy': occupants,
        'AC_Hours': ac_hours,
        'Appliance_Hours': appliance_hours,
        'Day_Type': day_encoded
    }])
    
    # Running Model Inference (Prediction):
    # We call engine.predict() which executes the regression equation using the loaded weights:
    # predicted_load = Intercept + w1*Temp + w2*Humidity + w3*Occupancy + w4*AC_Hours + w5*Appliance_Hours + w6*Day_Type.
    # The output is the forecasted continuous target value (electricity consumption in kWh).
    predicted_load = engine.predict(input_df)[0]
    
    # Estimates derived from forecasted consumption (kWh).
    estimated_cost = predicted_load * 0.15      # Cost estimate ($0.15 per kWh)
    estimated_emissions = predicted_load * 0.4  # Carbon footprint (0.4 kg CO2 per kWh)
    
    with col_display:
        st.markdown('<div class="section-header">Energy Forecast Metrics</div>', unsafe_allow_html=True)
        
        # Grid of premium KPI cards
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.markdown(clean_html(f"""
            <div class="kpi-container">
                <div class="kpi-title">Forecasted Daily Load</div>
                <div class="kpi-value">{predicted_load:.2f} kWh</div>
                <div class="kpi-trend">Estimated Usage Rate</div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown(clean_html(f"""
            <div class="kpi-container">
                <div class="kpi-title">Projected Operating Cost</div>
                <div class="kpi-value" style="color: #F59E0B;">${estimated_cost:.2f}</div>
                <div class="kpi-trend" style="color: #64748B;">Tariff: $0.15 / kWh</div>
            </div>
            """), unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(clean_html(f"""
            <div class="kpi-container">
                <div class="kpi-title">Carbon Footprint Impact</div>
                <div class="kpi-value" style="color: #3B82F6;">{estimated_emissions:.2f} kg</div>
                <div class="kpi-trend" style="color: #64748B;">CO2 Equivalency Rate</div>
            </div>
            """), unsafe_allow_html=True)
            
            # Consumption Threshold Limit Check
            is_critical = predicted_load > 180.0
            status_color = "#EF4444" if is_critical else "#10B981"
            status_text = "CRITICAL LIMIT EXCEEDED" if is_critical else "NOMINAL CAPACITY STATUS"
            
            st.markdown(clean_html(f"""
            <div class="kpi-container">
                <div class="kpi-title">Consumption Status</div>
                <div class="kpi-value" style="color: {status_color}; font-size: 24px; padding-top: 8px;">{status_text}</div>
                <div class="kpi-trend" style="color: #64748B;">Threshold Limit: 180 kWh</div>
            </div>
            """), unsafe_allow_html=True)

    # Key Drivers & Interpretability
    st.markdown('<div class="section-header">Key Drivers & Attribution Analysis</div>', unsafe_allow_html=True)
    
    col_driver1, col_driver2 = st.columns([1, 1])
    
    with col_driver1:
        st.write("##### Real-Time Feature Attribution")
        
        # =====================================
        # FEATURE ATTRIBUTION (XAI)
        # =====================================
        # Calculating Feature Attribution (Key Drivers):
        # We multiply each input value by its learned coefficient (weight) from the trained model.
        # This shows us exactly how many kWh each feature contributed to the final forecast.
        # It explains *why* the model predicted a particular load, making the AI explainable (XAI).
        impacts = [
            ('Outdoor Temperature', engine.coef_[0] * temperature),
            ('Relative Humidity', engine.coef_[1] * humidity),
            ('Occupant Count', engine.coef_[2] * occupants),
            ('AC Operating Hours', engine.coef_[3] * ac_hours),
            ('Appliance Operating Hours', engine.coef_[4] * appliance_hours),
            ('Day Type Classification', engine.coef_[5] * day_encoded)
        ]
        # Sort features so the highest contributing driver shows up first.
        impacts = sorted(impacts, key=lambda x: x[1], reverse=True)
        
        bar_html = ""
        max_val = 110.0  # Scale against maximum potential single feature impact
        
        # Build custom CSS progress bars to visualize feature attribution
        for label, val in impacts:
            pct = min(100.0, max(0.0, (val / max_val) * 100))
            bar_html += f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 12px;">
                    <span style="color: #94A3B8; font-weight: 500;">{label}</span>
                    <span style="color: #F8FAFC; font-weight: 600;">{val:.2f} kWh</span>
                </div>
                <div style="background-color: #1E293B; height: 6px; border-radius: 3px; overflow: hidden; width: 100%;">
                    <div style="background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%); width: {pct:.1f}%; height: 100%; border-radius: 3px;"></div>
                </div>
            </div>
            """
            
        st.markdown(clean_html(f"""
        <div style="background-color: #121620; border: 1px solid #1E293B; border-radius: 6px; padding: 20px; min-height: 275px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 15px;">
                    Active Attribution Weightings
                </div>
                {bar_html}
            </div>
        </div>
        """), unsafe_allow_html=True)
        
    with col_driver2:
        st.write("##### Optimization Recommendations")
        
        # Generate operational recommendations based on slider thresholds.
        rec_list = []
        if ac_hours > 6.0:
            rec_list.append(f"Decrease active AC cooling by 1 hour to reduce demand by <b>{engine.coef_[3]:.2f} kWh</b>.")
        if occupants > 4:
            rec_list.append("Coordinate large appliance schedules to optimize load distribution.")
        if appliance_hours > 8.0:
            rec_list.append("Power down unneeded standby appliances to lower idle load profiles.")
        if temperature > 32.0:
            rec_list.append("Employ shading or passive cooling strategies to limit outdoor thermal gain impact.")
            
        rec_html = ""
        if rec_list:
            for item in rec_list:
                rec_html += f"<li style='margin-bottom: 8px; color: #E2E8F0; font-size: 13px; line-height: 1.4;'>{item}</li>"
        else:
            rec_html = "<li style='margin-bottom: 8px; color: #E2E8F0; font-size: 13px;'>All operational parameters are optimally configured.</li>"
            
        alert_bg = "#7F1D1D" if is_critical else "#064E3B"
        alert_border = "#F87171" if is_critical else "#34D399"
        alert_text = "#FECACA" if is_critical else "#A7F3D0"
        alert_msg = "💡 High power load detected. Apply recommendations:" if is_critical else "💡 Consumption loads stable. Optimization tips:"
        
        st.markdown(clean_html(f"""
        <div style="background-color: #121620; border: 1px solid #1E293B; border-radius: 6px; padding: 20px; min-height: 275px; display: flex; flex-direction: column; justify-content: flex-start;">
            <div>
                <div style="background-color: {alert_bg}; border: 1px solid {alert_border}; color: {alert_text}; padding: 10px 14px; border-radius: 4px; font-size: 13px; font-weight: 500; margin-bottom: 15px;">
                    {alert_msg}
                </div>
                <ul style="margin: 0; padding-left: 20px;">
                    {rec_html}
                </ul>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # Model Insights & Diagnostics (Tabs at bottom of dashboard)
    st.markdown('<div class="section-header">Predictive Engine Diagnostics</div>', unsafe_allow_html=True)
    
    tab_inspect1, tab_inspect2 = st.tabs(["🧬 Model Parameter Metrics", "📈 Historical Exploration Charts"])
    
    with tab_inspect1:
        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            st.write("##### Coefficients Summary Table")
            coef_table = pd.DataFrame({
                'Feature Attribute': ['Temperature', 'Humidity', 'Occupancy', 'AC_Hours', 'Appliance_Hours', 'Day_Type'],
                'Learned Slope (Weight)': engine.coef_
            })
            st.dataframe(coef_table, use_container_width=True)
        with insight_col2:
            st.write("##### Model Formula")
            st.code(f"""
Forecasted_Load = {engine.intercept_:.4f}
                 + ({engine.coef_[0]:.4f} * Temp)
                 + ({engine.coef_[1]:.4f} * Humid)
                 + ({engine.coef_[2]:.4f} * Occupancy)
                 + ({engine.coef_[3]:.4f} * AC_Hours)
                 + ({engine.coef_[4]:.4f} * Appliance_Hours)
                 + ({engine.coef_[5]:.4f} * Day_Type)
            """, language="text")
            st.caption(f"Intercept value: {engine.intercept_:.4f}")
            
    with tab_inspect2:
        figures_path = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
        
        target_img = os.path.join(figures_path, "target_distribution.png")
        heatmap_img = os.path.join(figures_path, "correlation_heatmap.png")
        
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            if os.path.exists(target_img):
                st.image(target_img, caption="Consumption Load Target Distribution")
        with eda_col2:
            if os.path.exists(heatmap_img):
                st.image(heatmap_img, caption="Correlation Matrix Map")

# Footer brand tagline
st.markdown("---")
st.caption("Volterra Energy forecasting technology. Designed by ritesh-1918.")
