# 🔬 Alumina-Jute Composite Material AI Optimizer & Inverse Design

An end-to-end Machine Learning web application designed to predict mechanical properties (Tensile and Flexural Strength) of alumina nanoparticle-reinforced jute composites and run inverse numerical optimization to compute optimal formulations based on target performance criteria.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://composite-material-optimizer-hrqbuiutrjuh2epcdtzbyg.streamlit.app)

---

## 📌 Features

- **Forward Property Prediction:** Interactive machine learning model using `Scikit-Learn` Random Forest Regressors to predict tensile strength and flexural strength given constituent ratios and curing parameters.
- **Inverse Formulation Design:** Leverages `SciPy` numerical optimization (`L-BFGS-B` algorithm) to solve inverse problems—automatically computing the optimal alumina wt%, jute fiber wt%, and curing temperature required to achieve user-specified performance targets.
- **Dynamic Interactive UI:** High-performance dashboard built with `Streamlit` and `Plotly` for real-time parameter exploration and visualization.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (Random Forest Regression)
- **Mathematical Optimization:** SciPy (`scipy.optimize.minimize`)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Plotly Express
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📂 Project Structure

```text
composite-material-optimizer/
│
├── app.py                  # Streamlit web UI & Plotly interactive dashboard logic
├── model.py                # Synthetic dataset generator, model training, and SciPy optimizer engine
├── requirements.txt        # Production dependencies for cloud deployment
├── composite_data.csv      # Material properties dataset
├── model_tensile.pkl       # Serialized Random Forest model for Tensile Strength
└── model_flexural.pkl      # Serialized Random Forest model for Flexural Strength
