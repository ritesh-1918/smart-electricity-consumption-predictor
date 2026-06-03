"""
Volterra: Intelligent Energy Analytics Engine
---------------------------------------------
This script serves as the interactive dashboard for Volterra, a forecasting
and attribution platform predicting daily electricity consumption, projected costs,
carbon emissions, and feature attribution impacts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configurations
st.set_page_config(
    page_title="Volterra Energy Analytics",
    page_icon="⚡",
    layout="wide"
)

# Custom css styling to simulate premium SaaS dashboard
st.markdown("""
<style>
    .metric-card {
        background-color: #161A22;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .metric-value {
        font-size: 38px;
        font-weight: bold;
        color: #10B981;
        margin: 5px 0px;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .warning-card {
        border-color: #EF4444 !important;
    }
    .warning-value {
        color: #EF4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# Path to the serialized model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")

# Load model function with caching
@st.cache_resource
def load_trained_model(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.warning(f"Failed to deserialize model file: {e}")
        return None

model = load_trained_model(MODEL_PATH)

# Sidebar with Platform Specs and Scenario Presets
st.sidebar.title("⚡ Volterra Control Console")

st.sidebar.subheader("Presets")
scenario = st.sidebar.selectbox(
    "Load Profile Preset",
    options=["Manual Override", "Peak Summer Load", "Baseline Load", "Eco Mode"]
)

# Defaults mapping
if scenario == "Peak Summer Load":
    d_temp = 42.0
    d_humidity = 70.0
    d_occupancy = 5
    d_ac = 12.0
    d_appliance = 10.0
    d_day = "Weekend"
elif scenario == "Baseline Load":
    d_temp = 20.0
    d_humidity = 40.0
    d_occupancy = 2
    d_ac = 0.0
    d_appliance = 3.0
    d_day = "Weekday"
elif scenario == "Eco Mode":
    d_temp = 24.0
    d_humidity = 50.0
    d_occupancy = 3
    d_ac = 2.0
    d_appliance = 4.0
    d_day = "Weekday"
else:
    # Manual
    d_temp = 25.0
    d_humidity = 50.0
    d_occupancy = 3
    d_ac = 4.0
    d_appliance = 6.0
    d_day = "Weekday"

st.sidebar.markdown("---")
st.sidebar.subheader("Platform Specifications")
st.sidebar.markdown(f"""
*   **Engine Core**: Linear Regression v1.0
*   **Model Accuracy ($R^2$)**: `94.95%`
*   **Mean Absolute Error**: `6.38 kWh`
*   **Baseline Tariff**: `$0.15 / kWh`
*   **Emissions Rate**: `0.4 kg CO2 / kWh`
""")

# Header section
st.title("⚡ VOLTERRA: INTELLIGENT ENERGY ANALYTICS ENGINE")
st.markdown("""
*Volterra analyzes real-time environment variables and appliance statistics to forecast household electricity consumption and calculate feature attribution impacts.*
""")

if model is None:
    st.error("⚠️ Volterra prediction engine core was not found. Please verify that the training script `src/train.py` was executed successfully.")
else:
    # 2-Column layout for inputs
    st.markdown("---")
    st.subheader("📊 Load Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### 🌡️ Environmental Metrics")
        temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=50.0, value=d_temp, step=0.5)
        humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=d_humidity, step=1.0)
        occupancy = st.slider("Occupancy (Active Occupants)", min_value=1, max_value=10, value=d_occupancy, step=1)
        
    with col2:
        st.write("##### 🔌 Appliance Activity Profiles")
        ac_hours = st.slider("AC Operating Hours (daily)", min_value=0.0, max_value=24.0, value=d_ac, step=0.5)
        appliance_hours = st.slider("Appliance Operating Hours (daily)", min_value=0.0, max_value=24.0, value=d_appliance, step=0.5)
        day_type = st.segmented_control("Day Type Profile", options=["Weekday", "Weekend"], default=d_day)
        
    # Process Day Type mapping
    day_type_encoded = 0 if day_type == "Weekday" else 1
    
    # Input validation alert
    if ac_hours + appliance_hours > 24.0:
        st.warning("⚠️ High Load Alert: Combined AC and appliance operating time exceeds 24 hours.")

    # Format input row
    input_row = pd.DataFrame([{
        'Temperature': temperature,
        'Humidity': humidity,
        'Occupancy': occupancy,
        'AC_Hours': ac_hours,
        'Appliance_Hours': appliance_hours,
        'Day_Type': day_type_encoded
    }])
    
    # Run prediction
    prediction = model.predict(input_row)[0]
    
    # Financial and Ecological projections
    operating_cost = prediction * 0.15
    carbon_emissions = prediction * 0.4
    
    # Define threshold warning
    is_high_load = prediction > 180.0
    card_class = "metric-card warning-card" if is_high_load else "metric-card"
    val_class = "metric-value warning-value" if is_high_load else "metric-value"
    status_label = "🚨 CRITICAL LOAD EXCEEDED" if is_high_load else "🟢 NOMINAL LOAD PROFILE"
    
    # Hero KPI Grid (3 Columns)
    st.markdown("---")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-label">Forecasted Daily Load</div>
            <div class="{val_class}">{prediction:.2f} kWh</div>
            <div style="font-size: 11px; font-weight: bold; margin-top: 5px;">{status_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Operating Cost</div>
            <div class="metric-value" style="color: #F59E0B;">${operating_cost:.2f}</div>
            <div style="font-size: 11px; color: #64748B; font-weight: bold; margin-top: 5px;">TARIFF: $0.15 / kWh</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Carbon Footprint</div>
            <div class="metric-value" style="color: #6366F1;">{carbon_emissions:.2f} kg</div>
            <div style="font-size: 11px; color: #64748B; font-weight: bold; margin-top: 5px;">0.4 kg CO2 / kWh RATE</div>
        </div>
        """, unsafe_allow_html=True)

    # Capacity bar
    progress_val = min(max((prediction - 50) / 200, 0.0), 1.0)
    st.progress(progress_val, text="Total Simulation Load Factor (scaled 50 kWh - 250 kWh)")

    # Interpretability suite
    st.markdown("---")
    st.subheader("🔬 Attribution & Performance Suite")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dynamic Attribution Impact",
        "📈 Feature Attribution Coefficients",
        "💡 Diagnostics Engine Mechanics",
        "🔍 Historical Data Insights"
    ])
    
    with tab1:
        st.markdown("##### Real-Time Feature Attribution ($Weight \\times Input\\_Value$):")
        
        contrib_df = pd.DataFrame({
            'Feature': ['Temperature', 'Humidity', 'Occupancy', 'AC_Hours', 'Appliance_Hours', 'Day_Type'],
            'Contribution (kWh)': [
                model.coef_[0] * temperature,
                model.coef_[1] * humidity,
                model.coef_[2] * occupancy,
                model.coef_[3] * ac_hours,
                model.coef_[4] * appliance_hours,
                model.coef_[5] * day_type_encoded
            ]
        }).sort_values(by='Contribution (kWh)', ascending=True)
        
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.barplot(
            x='Contribution (kWh)', 
            y='Feature', 
            data=contrib_df, 
            hue='Feature',
            palette='viridis', 
            legend=False,
            ax=ax2
        )
        ax2.set_title("Current Attribution Contribution to Forecast", fontsize=10)
        ax2.set_xlabel("Contribution (kWh)", fontsize=8)
        ax2.set_ylabel("")
        plt.tight_layout()
        
        st.pyplot(fig2)
        st.caption("Active feature contributions display exactly which inputs drive the current prediction calculation based on their coefficients.")
        
    with tab2:
        st.markdown("##### Learned Attribution Coefficients:")
        st.info("Attribution coefficients display the static weights computed by the model during training, representing impact per unit of change.")
        
        coef_df = pd.DataFrame({
            'Feature': ['Temperature', 'Humidity', 'Occupancy', 'AC_Hours', 'Appliance_Hours', 'Day_Type'],
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#EF4444' if x < 1.0 else '#10B981' for x in coef_df['Coefficient']]
        
        sns.barplot(
            x='Coefficient', 
            y='Feature', 
            data=coef_df, 
            hue='Feature', 
            palette=dict(zip(coef_df['Feature'], colors)), 
            legend=False, 
            ax=ax
        )
        ax.set_title("Attribution Weights per Unit Feature Increase", fontsize=10)
        ax.set_xlabel("Coefficient Weight", fontsize=8)
        ax.set_ylabel("")
        plt.tight_layout()
        
        st.pyplot(fig)
        st.caption("Slopes dictate the scale and direction of predicted load fluctuations when individual variables shift.")
        
    with tab3:
        st.markdown("##### Attribution Equation Solver:")
        st.code(f"""
Electricity Consumption = Intercept + (w1 * Temp) + (w2 * Humidity) + (w3 * Occupancy) + (w4 * AC_Hours) + (w5 * Appliance_Hours) + (w6 * Day_Type)

Inputs plugged in:
Electricity Consumption = {model.intercept_:.4f}
                        + ({model.coef_[0]:.4f} * {temperature})
                        + ({model.coef_[1]:.4f} * {humidity})
                        + ({model.coef_[2]:.4f} * {occupancy})
                        + ({model.coef_[3]:.4f} * {ac_hours})
                        + ({model.coef_[4]:.4f} * {appliance_hours})
                        + ({model.coef_[5]:.4f} * {day_type_encoded})
                        
                        = {prediction:.2f} kWh
        """, language="text")
        
    with tab4:
        st.markdown("##### Exploratory Data Analysis & Historical Distributions:")
        
        figures_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
        
        # Load images
        target_dist_path = os.path.join(figures_dir, "target_distribution.png")
        heatmap_path = os.path.join(figures_dir, "correlation_heatmap.png")
        histograms_path = os.path.join(figures_dir, "histograms.png")
        boxplots_path = os.path.join(figures_dir, "boxplots.png")
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            if os.path.exists(target_dist_path):
                st.image(target_dist_path, caption="Distribution of Electricity Consumption")
            if os.path.exists(histograms_path):
                st.image(histograms_path, caption="Histograms of Dataset Features")
        with col_eda2:
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, caption="Correlation Matrix Heatmap")
            if os.path.exists(boxplots_path):
                st.image(boxplots_path, caption="Boxplots (Outlier Checks)")
                
        st.markdown("##### Feature Interactions")
        col_eda3, col_eda4 = st.columns(2)
        with col_eda3:
            temp_vs_path = os.path.join(figures_dir, "temp_vs_consumption.png")
            if os.path.exists(temp_vs_path):
                st.image(temp_vs_path, caption="Temperature vs. Consumption")
            occupancy_vs_path = os.path.join(figures_dir, "occupancy_vs_consumption.png")
            if os.path.exists(occupancy_vs_path):
                st.image(occupancy_vs_path, caption="Occupancy vs. Consumption")
        with col_eda4:
            ac_vs_path = os.path.join(figures_dir, "ac_hours_vs_consumption.png")
            if os.path.exists(ac_vs_path):
                st.image(ac_vs_path, caption="AC Active Hours vs. Consumption")
            app_vs_path = os.path.join(figures_dir, "appliance_hours_vs_consumption.png")
            if os.path.exists(app_vs_path):
                st.image(app_vs_path, caption="Appliance Active Hours vs. Consumption")

# Footer info
st.markdown("---")
st.caption("Volterra Energy Analytics Engine • Powered by Scikit-Learn and Streamlit.")
