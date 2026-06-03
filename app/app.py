"""
Streamlit Web App (Option A: Smart Electricity Predictor)
---------------------------------------------------------
This script provides a workshop-grade, highly interactive dashboard that allows B.Tech 
students to study how a Linear Regression model calculates electricity consumption.
It highlights the intercept, slopes, and feature relationships in an educational format.
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
    page_title="Smart Electricity Predictor",
    page_icon="⚡",
    layout="wide"
)

# Custom css styling to simulate premium developer environment
st.markdown("""
<style>
    .metric-card {
        background-color: #1A1D24;
        border: 2px solid #00F2FE;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 54px;
        font-weight: bold;
        color: #00F2FE;
        margin: 10px 0px;
    }
    .metric-label {
        font-size: 14px;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .warning-card {
        border-color: #FF6B6B !important;
    }
    .warning-value {
        color: #FF6B6B !important;
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
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_trained_model(MODEL_PATH)

# Sidebar with Educational content for B.Tech workshops
st.sidebar.title("📚 Workshop Syllabus")
st.sidebar.markdown("""
### Core Concepts
1. **Supervised Learning**
   The training dataset contains both inputs and correct label outputs. The model learns to map input patterns to target outputs.
2. **Regression**
   Predicting a continuous numerical quantity (like electricity load in kWh).
3. **Linear Regression**
   Fits a straight hyperplane to minimize prediction error.
   
$$y = \\beta_0 + \\sum_{i=1}^{n} \\beta_i X_i$$

* **$\\beta_0$**: Intercept (baseline constant).
* **$\\beta_i$**: Coefficients (slopes).
* **$X_i$**: Input variables.
""")

# Header section
st.title("⚡ SMART ELECTRICITY PREDICTOR")
st.markdown("""
*Production Server: v1.0.4* &nbsp;•&nbsp; **Model Accuracy ($R^2$): 94.95%** &nbsp;•&nbsp; **Algorithm: Linear Regression**
""")

# Switch modes
mode = st.radio(
    "Select Display Mode:",
    options=["🎓 Workshop Demo Mode", "📱 Standard User App"],
    horizontal=True,
    help="Demo Mode displays the live math equation solver and feature weights."
)

if model is None:
    st.error("⚠️ Trained model file `linear_regression_model.pkl` was not found in the `models/` directory. Please run the training script `src/train.py` first to generate it.")
else:
    # 2-Column layout for inputs
    st.markdown("---")
    st.subheader("🎛️ Live Parameter Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### 🌡️ Environmental Metrics")
        temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=50.0, value=25.0, step=0.5)
        humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
        occupancy = st.slider("Occupancy (Number of active occupants)", min_value=1, max_value=10, value=3, step=1)
        
    with col2:
        st.write("##### 🔌 Appliance Activity profiles")
        ac_hours = st.slider("AC Active Hours (per day)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
        appliance_hours = st.slider("Appliance Active Hours (per day)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
        day_type = st.segmented_control("Day Type Profile", options=["Weekday", "Weekend"], default="Weekday")
        
    # Process Day Type mapping
    day_type_encoded = 0 if day_type == "Weekday" else 1
    
    # Input validation alert
    if ac_hours + appliance_hours > 24.0:
        st.warning("⚠️ Combined AC and appliance hours exceed 24 hours. The model will extrapolate predictions.")

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
    
    # Define threshold warning
    is_high_load = prediction > 180.0
    card_class = "metric-card warning-card" if is_high_load else "metric-card"
    value_class = "metric-value warning-value" if is_high_load else "metric-value"
    status_label = "🚨 HIGH LOAD ALERT" if is_high_load else "🟢 NORMAL CONSUMPTION"
    
    # 3. Hero Element (Prediction Card)
    st.markdown("---")
    st.markdown(f"""
    <div class="{card_class}">
        <div class="metric-label">Estimated Household Load</div>
        <div class="{value_class}">{prediction:.2f} kWh</div>
        <div style="font-weight: bold; margin-top: 5px;">Status: {status_label}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Demand indicator bar (visual gauge)
    progress_val = min(max((prediction - 50) / 200, 0.0), 1.0)
    st.progress(progress_val, text=f"Total Capacity Usage (scaled 50 kWh to 250 kWh)")

    # 4. Interactive Interpretability Tabs (Explainable AI)
    if mode == "🎓 Workshop Demo Mode":
        st.markdown("---")
        st.subheader("🔬 Model Interpretability Suite")
        
        tab1, tab2, tab3 = st.tabs(["💡 Live Equation Solver", "📊 Feature Weights (Coefficients)", "📈 Live Feature Contributions"])
        
        with tab1:
            st.markdown("""
            **How the model computed this value:**
            Linear Regression multiplies each input feature by its learned coefficient (slope), then adds the intercept (baseline constant).
            """)
            
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
            
        with tab2:
            st.markdown("**Learned Feature Weights (Slopes):**")
            st.info("Note: The coefficients/weights are the fixed mathematical parameters learned by the model during training. They do not change when you move the sliders.")
            # Create a simple horizontal bar chart of coefficients
            coef_df = pd.DataFrame({
                'Feature': ['Temperature', 'Humidity', 'Occupancy', 'AC_Hours', 'Appliance_Hours', 'Day_Type'],
                'Coefficient': model.coef_
            }).sort_values(by='Coefficient', ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ['#FF6B6B' if x < 1.0 else '#00F2FE' for x in coef_df['Coefficient']]
            
            # Using hue to suppress deprecation warning
            sns.barplot(
                x='Coefficient', 
                y='Feature', 
                data=coef_df, 
                hue='Feature', 
                palette=dict(zip(coef_df['Feature'], colors)), 
                legend=False, 
                ax=ax
            )
            ax.set_title("Feature Weights (Impact per Unit Change)", fontsize=10)
            ax.set_xlabel("Coefficient Weight", fontsize=8)
            ax.set_ylabel("")
            plt.tight_layout()
            
            st.pyplot(fig)
            st.caption("A positive weight indicates that increasing the feature value directly raises electricity consumption. The longer the bar, the larger the feature's influence.")
            
        with tab3:
            st.markdown("**Real-Time Feature Contributions ($Weight \\times Input\\_Value$):**")
            st.success("This chart updates in real-time as you move the sliders to show exactly which input is contributing the most to the current predicted consumption!")
            
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
            
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            sns.barplot(
                x='Contribution (kWh)', 
                y='Feature', 
                data=contrib_df, 
                hue='Feature',
                palette='viridis', 
                legend=False,
                ax=ax2
            )
            ax2.set_title("Current Feature Contributions to Prediction", fontsize=10)
            ax2.set_xlabel("Contribution (kWh)", fontsize=8)
            ax2.set_ylabel("")
            plt.tight_layout()
            
            st.pyplot(fig2)
            st.caption("Contribution is calculated as: learned coefficient weight multiplied by your slider input value. This allows you to inspect what is driving the active prediction.")
            
    else:
        # Standard User Mode Recommendations
        st.markdown("---")
        st.subheader("💡 Energy Saving Recommendations")
        
        if is_high_load:
            st.warning("⚠️ High daily energy load expected. Consider the following steps to save power:")
        else:
            st.info("ℹ️ Your household energy footprint is within normal limits. Keep it up! Here are minor optimization tips:")
            
        col1, col2 = st.columns(2)
        with col1:
            if ac_hours > 6.0:
                st.write(f"- **AC Optimization**: Reducing AC active time by 1 hour saves ~**{model.coef_[3]:.2f} kWh**.")
            if temperature > 32.0:
                st.write("- **Ventilation**: Use fans instead of air conditioners when outdoor temperature permits.")
        with col2:
            if occupancy > 4:
                st.write(f"- **Shared Load**: Coordinate household appliance usage during peak periods.")
            if appliance_hours > 8.0:
                st.write(f"- **Standby Power**: Unplug standby appliances to reduce base overhead load.")
                
# Footer info
st.markdown("---")
st.caption("Designed for B.Tech Machine Learning Workshops & Laboratory Demonstrations.")
