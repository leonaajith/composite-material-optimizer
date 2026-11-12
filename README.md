# 🔬 Alumina Nanoparticle–Jute Epoxy Composite Database & Optimizer

An end-to-end **Materials Informatics & Machine Learning** web application designed to store multi-modal experimental characterization data in a relational SQLite database, predict mechanical properties of $\text{Al}_2\text{O}_3$-reinforced jute–epoxy composites, and perform inverse formulation optimization.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://composite-material-optimizer-hrqbuiutrjuh2epcdtzbyg.streamlit.app)

---

## 📌 Key Features

* **Relational SQL Data Explorer (`SQLite`):** Normalizes experimental mechanical properties (`UT`, `0.25%`, `0.50%`, and `1.00 wt% Al₂O₃`), XRD diffraction patterns, and FTIR transmittance spectra into a three-table SQLite repository (`materials.db`) with interactive ANSI-SQL filtering.
* **Non-Linear Property Prediction (`Scikit-Learn`):** Uses Degree-2 Polynomial Regression surrogates ($R^2 = 0.9568$, $\text{MAE} = 1.05\text{ MPa}$ on tensile strength) to model nanoparticle reinforcement up to the $0.50\text{ wt\%}$ optimum and capture agglomeration decline at higher loadings.
* **Inverse Formulation Design (`SciPy`):** Employs bound-constrained `L-BFGS-B` numerical optimization to solve for global maximum mechanical strength or compute exact $\text{Al}_2\text{O}_3$ weight fractions for user-specified tensile targets.
* **Multi-Modal Characterization Viewer (`Plotly`):** Interactive visualization of crystalline XRD peaks ($2\theta = 22.6^\circ$) and inverted FTIR functional-group transmittance spectra ($4000\text{–}400\text{ cm}^{-1}$) queried directly from SQLite.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Database:** SQLite3 (Relational 3-table schema: `mechanical_properties`, `xrd_spectra`, `ftir_spectra`)
* **Machine Learning:** Scikit-Learn (`PolynomialFeatures` + `LinearRegression`)
* **Mathematical Optimization:** SciPy (`scipy.optimize.minimize` via `L-BFGS-B`)
* **Data Manipulation & Visualization:** Pandas, NumPy, Plotly Express
* **Web Framework & Deployment:** Streamlit, Streamlit Cloud

---

## 📁 Project Structure

```text
composite-material-optimizer/
├── app.py               # 4-tab Streamlit web dashboard & interactive Plotly visualizations
├── model.py             # SQLite database builder, polynomial model trainer & SciPy optimizer
├── materials.db         # Normalized SQLite database (Mechanical, XRD, and FTIR tables)
├── model_tensile.pkl    # Serialized Degree-2 Polynomial model for Tensile Strength
├── model_flexural.pkl   # Serialized Degree-2 Polynomial model for Flexural Strength
├── model_moduli.pkl     # Serialized models for Tensile & Flexural Moduli
├── requirements.txt     # Cloud deployment dependencies
└── README.md            # Project documentation
