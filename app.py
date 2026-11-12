import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from model import optimize_composition

st.set_page_config(page_title="Alumina-Jute Composite System", layout="wide")
st.title("🔬 Alumina Nanoparticle–Jute Epoxy Composite Database & Optimizer")
st.markdown("Relational SQLite repository, ML mechanical property predictor, SciPy optimizer, and XRD/FTIR characterization viewer.")

@st.cache_resource
def load_models():
    with open('model_tensile.pkl', 'rb') as f:
        t_model = pickle.load(f)
    with open('model_flexural.pkl', 'rb') as f:
        f_model = pickle.load(f)
    with open('model_moduli.pkl', 'rb') as f:
        t_mod, f_mod = pickle.load(f)
    return t_model, f_model, t_mod, f_mod

model_tensile, model_flexural, model_t_mod, model_f_mod = load_models()

tab1, tab2, tab3, tab4 = st.tabs([
    "🗄️ SQL Data Explorer",
    "📈 ML Property Predictor",
    "⚙️ SciPy Ratio Optimizer",
    "🧪 XRD & FTIR Spectra"
])

# TAB 1: SQLite Database Explorer
with tab1:
    st.subheader("Experimental Trials Repository (SQLite Backend)")
    min_ts = st.slider("SQL Filter — Minimum Tensile Strength (MPa)", 0.0, 45.0, 25.0, 0.5)
    
    conn = sqlite3.connect('materials.db')
    query = "SELECT * FROM mechanical_properties WHERE tensile_strength >= ? ORDER BY al2o3_wt ASC"
    df_sql = pd.read_sql_query(query, conn, params=(min_ts,))
    conn.close()
    
    st.code(f"SELECT * FROM mechanical_properties WHERE tensile_strength >= {min_ts} ORDER BY al2o3_wt ASC;", language="sql")
    st.dataframe(df_sql, use_container_width=True)

# TAB 2: ML Property Predictor
with tab2:
    st.subheader("Predict Mechanical Properties Across Al₂O₃ Concentrations")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        alumina = st.slider("Al₂O₃ Nanoparticle Loading (wt%)", 0.00, 1.00, 0.50, 0.01)
        pred_ts = model_tensile.predict([[alumina]])[0]
        pred_fs = model_flexural.predict([[alumina]])[0]
        pred_tm = model_t_mod.predict([[alumina]])[0]
        pred_fm = model_f_mod.predict([[alumina]])[0]
        
        st.metric("Predicted Tensile Strength", f"{pred_ts:.2f} MPa")
        st.metric("Predicted Flexural Strength", f"{pred_fs:.2f} MPa")
        st.metric("Predicted Tensile Modulus", f"{pred_tm:.2f} GPa")
        st.metric("Predicted Flexural Modulus", f"{pred_fm:.2f} GPa")

    with col2:
        alumina_range = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
        curve_df = pd.DataFrame({
            'Al2O3 (wt%)': alumina_range.flatten(),
            'Tensile Strength (MPa)': model_tensile.predict(alumina_range),
            'Flexural Strength (MPa)': model_flexural.predict(alumina_range)
        })
        fig = px.line(curve_df, x='Al2O3 (wt%)', y=['Tensile Strength (MPa)', 'Flexural Strength (MPa)'],
                      title="Polynomial Regression Fit vs. Experimental Trend (Peak at ~0.50 wt%)")
        fig.add_vline(x=alumina, line_dash="dash", line_color="red", annotation_text="Selected wt%")
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: SciPy Inverse Optimization
with tab3:
    st.subheader("Automated Formulation Optimizer (SciPy L-BFGS-B)")
    mode = st.radio("Optimization Objective", ["Maximize Mechanical Strength", "Match Target Tensile Strength (MPa)"])
    
    if mode == "Maximize Mechanical Strength":
        if st.button("Find Global Optimal Al₂O₃ Loading"):
            res = optimize_composition(maximize_strength=True)
            st.success(f"**Optimal Loading:** {res['alumina_wt']} wt% Al₂O₃ → **Peak Tensile Strength:** {res['predicted_tensile']} MPa")
    else:
        target = st.number_input("Desired Tensile Strength (MPa)", min_value=28.0, max_value=45.0, value=38.0, step=0.5)
        if st.button("Solve for Required Al₂O₃ Loading"):
            res = optimize_composition(target_tensile=target, maximize_strength=False)
            st.success(f"**Recommended Loading:** {res['alumina_wt']} wt% Al₂O₃ (Predicted: {res['predicted_tensile']} MPa)")

# TAB 4: XRD & FTIR Characterization Spectra
with tab4:
    st.subheader("Material Characterization Spectra (Queried from SQLite)")
    conn = sqlite3.connect('materials.db')
    xrd_df = pd.read_sql_query("SELECT * FROM xrd_spectra", conn)
    ftir_df = pd.read_sql_query("SELECT * FROM ftir_spectra", conn)
    conn.close()

    c1, c2 = st.columns(2)
    with c1:
        xrd_samples = st.multiselect("Select XRD Samples", ['UT', '0.25% Al2O3', '0.50% Al2O3', '1.00% Al2O3'],
                                     default=['UT', '0.25% Al2O3', '0.50% Al2O3', '1.00% Al2O3'])
        if xrd_samples:
            fig_xrd = px.line(xrd_df, x='two_theta', y=xrd_samples,
                              labels={'two_theta': '2 Theta (degrees)', 'value': 'Intensity (a.u.)'},
                              title="XRD Diffraction Patterns (Peak at 22.6°)")
            st.plotly_chart(fig_xrd, use_container_width=True)

    with c2:
        ftir_samples = st.multiselect("Select FTIR Samples", ['UT', 'A1', 'B1', 'C1', 'A2', 'B2', 'C2'],
                                      default=['UT', 'A1', 'B1', 'C1'])
        if ftir_samples:
            fig_ftir = px.line(ftir_df, x='wavenumber', y=ftir_samples,
                               labels={'wavenumber': 'Wavenumber (cm⁻¹)', 'value': 'Transmittance (%)'},
                               title="FTIR Transmittance Spectra")
            fig_ftir.update_xaxes(autorange="reversed")
            st.plotly_chart(fig_ftir, use_container_width=True)