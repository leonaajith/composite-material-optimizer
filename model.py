import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize

def build_database_and_train():
    conn = sqlite3.connect('materials.db')

    # 1. Real Experimental Mechanical Data (Tensile + Flexural)
    mech_df = pd.DataFrame({
        'sample_id': ['UT', '0.25% Al2O3', '0.50% Al2O3', '1.00% Al2O3'],
        'al2o3_wt': [0.00, 0.25, 0.50, 1.00],
        'tensile_strength': [28.5, 36.2, 44.8, 39.1],
        'tensile_modulus': [1.82, 2.45, 3.10, 2.78],
        'flexural_strength': [42.3, 56.8, 71.5, 62.4],
        'flexural_modulus': [2.15, 2.90, 3.85, 3.32]
    })
    mech_df.to_sql('mechanical_properties', conn, if_exists='replace', index=False)

    # 2. Real XRD Patterns Data
    xrd_df = pd.DataFrame({
        'two_theta': [10, 12, 15, 18, 20, 22.6, 25, 28, 31, 34.5, 38, 40],
        'UT': [150, 220, 450, 300, 550, 950, 400, 250, 200, 310, 180, 140],
        '0.25% Al2O3': [160, 240, 500, 260, 680, 1350, 420, 260, 210, 380, 190, 150],
        '0.50% Al2O3': [170, 250, 520, 220, 750, 1650, 410, 270, 220, 420, 200, 155],
        '1.00% Al2O3': [165, 245, 510, 240, 710, 1480, 415, 265, 215, 400, 195, 150]
    })
    xrd_df.to_sql('xrd_spectra', conn, if_exists='replace', index=False)

    # 3. Real FTIR Spectra Data
    ftir_df = pd.DataFrame({
        'wavenumber': [4000, 3700, 3350, 2900, 2500, 2000, 1730, 1500, 1250, 1000, 750, 500, 400],
        'UT': [98, 97, 63, 88, 92, 94, 76, 85, 78, 48, 70, 55, 32],
        'A1': [96, 95, 88, 92, 93, 95, 87, 89, 82, 62, 78, 68, 45],
        'B1': [99, 98, 95, 96, 97, 98, 94, 96, 90, 78, 87, 79, 60],
        'C1': [95, 94, 86, 90, 92, 93, 85, 87, 79, 58, 74, 62, 40],
        'A2': [94, 93, 80, 87, 90, 92, 82, 84, 73, 50, 68, 52, 35],
        'B2': [93, 91, 84, 88, 91, 92, 83, 85, 75, 52, 71, 56, 38],
        'C2': [92, 90, 82, 86, 89, 91, 80, 82, 70, 46, 65, 48, 30]
    })
    ftir_df.to_sql('ftir_spectra', conn, if_exists='replace', index=False)
    conn.close()

    # 4. Train Degree-2 Polynomial Models on Real Data (captures peak at ~0.50% & drop at 1.00%)
    X = mech_df[['al2o3_wt']]
    model_tensile = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X, mech_df['tensile_strength'])
    model_flexural = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X, mech_df['flexural_strength'])
    model_t_mod = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X, mech_df['tensile_modulus'])
    model_f_mod = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X, mech_df['flexural_modulus'])

    with open('model_tensile.pkl', 'wb') as f:
        pickle.dump(model_tensile, f)
    with open('model_flexural.pkl', 'wb') as f:
        pickle.dump(model_flexural, f)
    with open('model_moduli.pkl', 'wb') as f:
        pickle.dump((model_t_mod, model_f_mod), f)

    # Print paper metrics
    r2_t = r2_score(mech_df['tensile_strength'], model_tensile.predict(X))
    mae_t = mean_absolute_error(mech_df['tensile_strength'], model_tensile.predict(X))
    print(f"Database & Models Saved! Tensile R2={r2_t:.4f}, MAE={mae_t:.2f} MPa")

def optimize_composition(target_tensile=None, maximize_strength=False):
    with open('model_tensile.pkl', 'rb') as f:
        model = pickle.load(f)

    def objective(params):
        pred = model.predict([[params[0]]])[0]
        if maximize_strength:
            return -pred
        return (pred - target_tensile) ** 2

    res = minimize(objective, [0.50], bounds=[(0.0, 1.0)], method='L-BFGS-B')
    opt_alumina = round(float(res.x[0]), 3)
    pred_ts = round(float(model.predict([[opt_alumina]])[0]), 2)
    return {'alumina_wt': opt_alumina, 'predicted_tensile': pred_ts}

if __name__ == "__main__":
    build_database_and_train()