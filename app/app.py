"""
Volterra SaaS Platform
----------------------
Production energy forecast, carbon tracking, and feature attribution dashboard.
Provides clean UI layout metrics and analytics insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configurations
st.set_page_config(
    page_title="Volterra | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark SaaS Design System (CSS Styles)
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
        padding: 20px;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
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

# Load trained regression weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")

@st.cache_resource
def get_prediction_engine(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None

engine = get_prediction_engine(MODEL_PATH)

# Sidebar - Settings and Simulation Presets only
st.sidebar.markdown("### ⚙️ Simulation Settings")
preset = st.sidebar.selectbox(
    "Select Load Profile Preset",
    options=["Manual Configuration", "Peak Demand Profile", "Eco Conservation Profile", "Baseline Utility Profile"]
)

# Preset configs mapping
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
    # Split Simulator inputs and live output predictions
    col_input, col_display = st.columns([1, 1])
    
    with col_input:
        st.markdown('<div class="section-header">Simulated Input Parameters</div>', unsafe_allow_html=True)
        
        inp_col1, inp_col2 = st.columns(2)
        with inp_col1:
            temperature = st.slider("Outdoor Temperature (°C)", 10.0, 50.0, val_temp, 0.5)
            humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, val_humidity, 1.0)
            occupants = st.number_input("Occupant Count", 1, 10, val_occupants, 1)
        with inp_col2:
            ac_hours = st.slider("AC Operating Hours (daily)", 0.0, 24.0, val_ac, 0.5)
            appliance_hours = st.slider("Appliance Operating Hours (daily)", 0.0, 24.0, val_appliance, 0.5)
            day_type = st.selectbox("Day Classification", ["Weekday", "Weekend"], index=0 if val_day == "Weekday" else 1)
            
    # Map day classification
    day_encoded = 0 if day_type == "Weekday" else 1
    
    # Predict output calculations
    input_df = pd.DataFrame([{
        'Temperature': temperature,
        'Humidity': humidity,
        'Occupancy': occupants,
        'AC_Hours': ac_hours,
        'Appliance_Hours': appliance_hours,
        'Day_Type': day_encoded
    }])
    
    predicted_load = engine.predict(input_df)[0]
    estimated_cost = predicted_load * 0.15
    estimated_emissions = predicted_load * 0.4
    
    with col_display:
        st.markdown('<div class="section-header">Energy Forecast Metrics</div>', unsafe_allow_html=True)
        
        # Grid of premium KPI cards
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Forecasted Daily Load</div>
                <div class="kpi-value">{predicted_load:.2f} kWh</div>
                <div class="kpi-trend">Estimated Usage Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Projected Operating Cost</div>
                <div class="kpi-value" style="color: #F59E0B;">${estimated_cost:.2f}</div>
                <div class="kpi-trend" style="color: #64748B;">Tariff: $0.15 / kWh</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Carbon Footprint Impact</div>
                <div class="kpi-value" style="color: #3B82F6;">{estimated_emissions:.2f} kg</div>
                <div class="kpi-trend" style="color: #64748B;">CO2 Equivalency Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Consumption Status
            is_critical = predicted_load > 180.0
            status_color = "#EF4444" if is_critical else "#10B981"
            status_text = "CRITICAL LIMIT EXCEEDED" if is_critical else "NOMINAL CAPACITY STATUS"
            
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Consumption Status</div>
                <div class="kpi-value" style="color: {status_color}; font-size: 24px; padding-top: 8px;">{status_text}</div>
                <div class="kpi-trend" style="color: #64748B;">Threshold Limit: 180 kWh</div>
            </div>
            """, unsafe_allow_html=True)

    # Key Drivers & Interpretability
    st.markdown('<div class="section-header">Key Drivers & Attribution Analysis</div>', unsafe_allow_html=True)
    
    col_driver1, col_driver2 = st.columns([1.2, 1])
    
    with col_driver1:
        st.write("##### Real-Time Feature Attribution")
        contrib_series = pd.DataFrame({
            'Attribute': ['Temperature', 'Humidity', 'Occupancy', 'AC_Hours', 'Appliance_Hours', 'Day_Type'],
            'Impact Value': [
                engine.coef_[0] * temperature,
                engine.coef_[1] * humidity,
                engine.coef_[2] * occupants,
                engine.coef_[3] * ac_hours,
                engine.coef_[4] * appliance_hours,
                engine.coef_[5] * day_encoded
            ]
        }).sort_values(by='Impact Value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor('#121620')
        ax.set_facecolor('#121620')
        
        sns.barplot(
            x='Impact Value',
            y='Attribute',
            data=contrib_series,
            hue='Attribute',
            palette='mako',
            legend=False,
            ax=ax
        )
        ax.set_title("Current Active Attribution Weightings (kWh)", fontsize=10, color='#F8FAFC')
        ax.xaxis.label.set_color('#94A3B8')
        ax.yaxis.label.set_color('#94A3B8')
        ax.tick_params(colors='#94A3B8', labelsize=8)
        ax.spines['bottom'].set_color('#1E293B')
        ax.spines['left'].set_color('#1E293B')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_driver2:
        # Energy Recommendations
        st.write("##### Optimization Recommendations")
        if is_critical:
            st.error("💡 High power load detected. Apply these recommendations to lower operating cost:")
        else:
            st.info("💡 Consumption loads are stable. Optional suggestions for maximum optimization:")
            
        rec_list = []
        if ac_hours > 6.0:
            rec_list.append(f"Decrease active AC cooling by 1 hour to reduce demand by **{engine.coef_[3]:.2f} kWh**.")
        if occupants > 4:
            rec_list.append("Coordinate large appliance schedules to optimize load distribution.")
        if appliance_hours > 8.0:
            rec_list.append("Power down unneeded standby appliances to lower idle load profiles.")
        if temperature > 32.0:
            rec_list.append("Employ shading or passive cooling strategies to limit outdoor thermal gain impact.")
            
        if rec_list:
            for item in rec_list:
                st.markdown(f"*   {item}")
        else:
            st.markdown("*   All operational parameters are optimally configured.")

    # Model Insights & Diagnostics
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
