import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from model import optimize_composition

st.set_page_config(page_title="Composite Material AI Optimizer", layout="wide")

# Title & Overview
st.title("🔬 Alumina-Jute Composite Material AI Optimizer")
st.markdown("Predict mechanical properties and perform inverse material design using Machine Learning.")

# Load Trained Models
@st.cache_resource
def load_models():
    with open('model_tensile.pkl', 'rb') as f:
        tensile_model = pickle.load(f)
    with open('model_flexural.pkl', 'rb') as f:
        flexural_model = pickle.load(f)
    return tensile_model, flexural_model

model_tensile, model_flexural = load_models()

# Create Tabs for Navigation
tab1, tab2 = st.tabs(["📊 Property Predictor", "🎯 Recipe Optimizer (Inverse Design)"])

# TAB 1: Forward Prediction
with tab1:
    st.subheader("Predict Strength from Input Formulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        alumina = st.slider("Alumina Nanoparticle (wt%)", 0.0, 1.0, 0.5, 0.05)
        jute = st.slider("Jute Fiber (wt%)", 20.0, 40.0, 30.0, 0.5)
        temp = st.slider("Curing Temperature (°C)", 80.0, 120.0, 100.0, 1.0)
        
        # Make Prediction
        input_data = [[alumina, jute, temp]]
        pred_tensile = model_tensile.predict(input_data)[0]
        pred_flexural = model_flexural.predict(input_data)[0]
        
        st.metric("Predicted Tensile Strength", f"{pred_tensile:.2f} MPa")
        st.metric("Predicted Flexural Strength", f"{pred_flexural:.2f} MPa")

    with col2:
        # Plotly chart showing Alumina wt% vs Tensile Strength
        alumina_range = np.linspace(0, 1.0, 50)
        sim_data = [[a, jute, temp] for a in alumina_range]
        tensile_curve = model_tensile.predict(sim_data)
        
        chart_df = pd.DataFrame({'Alumina wt%': alumina_range, 'Tensile Strength (MPa)': tensile_curve})
        fig = px.line(chart_df, x='Alumina wt%', y='Tensile Strength (MPa)', 
                      title=f"Impact of Alumina wt% on Tensile Strength (Jute: {jute}%, Temp: {temp}°C)")
        fig.add_vline(x=alumina, line_dash="dash", line_color="red", annotation_text="Current Selection")
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Inverse Optimization
with tab2:
    st.subheader("Automated Recipe Generator")
    st.write("Enter your desired target tensile strength, and SciPy optimization will generate the optimal composite formula.")
    
    target = st.number_input("Target Tensile Strength (MPa)", min_value=10.0, max_value=40.0, value=32.0, step=0.5)
    
    if st.button("Generate Optimal Recipe"):
        result = optimize_composition(target)
        
        st.success("Optimal Recipe Calculated!")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Recommended Alumina", f"{result['alumina_wt']} wt%")
        res_col2.metric("Recommended Jute Fiber", f"{result['jute_wt']} wt%")
        res_col3.metric("Recommended Curing Temp", f"{result['curing_temp']} °C")
