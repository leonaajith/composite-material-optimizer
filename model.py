import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

def generate_and_train():
    # 1. Create a dataset representing composite formulations
    np.random.seed(42)
    n_samples = 300

    # Features: Alumina % (0 to 10%), Jute Fiber % (20 to 40%), Curing Temp (80 to 120 C)
    alumina_wt = np.random.uniform(0, 10, n_samples)
    jute_wt = np.random.uniform(20, 40, n_samples)
    curing_temp = np.random.uniform(80, 120, n_samples)

    # Formula simulating physical property behavior (tensile & flexural strength)
    tensile_strength = 30 + 4.5 * alumina_wt - 0.35 * (alumina_wt**2) + 0.8 * jute_wt + 0.1 * curing_temp + np.random.normal(0, 2, n_samples)
    flexural_strength = 50 + 3.2 * alumina_wt - 0.25 * (alumina_wt**2) + 1.1 * jute_wt + 0.05 * curing_temp + np.random.normal(0, 2.5, n_samples)

    df = pd.DataFrame({
        'alumina_wt': alumina_wt,
        'jute_wt': jute_wt,
        'curing_temp': curing_temp,
        'tensile_strength': tensile_strength,
        'flexural_strength': flexural_strength
    })

    # Save initial dataset
    df.to_csv('composite_data.csv', index=False)
    print("Dataset saved to composite_data.csv")

    # 2. Train ML Models
    X = df[['alumina_wt', 'jute_wt', 'curing_temp']]
    y_tensile = df['tensile_strength']
    y_flexural = df['flexural_strength']

    model_tensile = RandomForestRegressor(n_estimators=100, random_state=42)
    model_tensile.fit(X, y_tensile)

    model_flexural = RandomForestRegressor(n_estimators=100, random_state=42)
    model_flexural.fit(X, y_flexural)

    # Save trained models
    with open('model_tensile.pkl', 'wb') as f:
        pickle.dump(model_tensile, f)
    with open('model_flexural.pkl', 'wb') as f:
        pickle.dump(model_flexural, f)

    print("Models trained and saved successfully!")

# Inverse Optimization Engine using SciPy
def optimize_composition(target_tensile):
    with open('model_tensile.pkl', 'rb') as f:
        model = pickle.load(f)

    # Objective function: minimize difference between prediction and user target
    def objective(params):
        alumina, jute, temp = params
        pred = model.predict([[alumina, jute, temp]])[0]
        return (pred - target_tensile) ** 2

    # Feature bounds: Alumina (0-10%), Jute (20-40%), Curing Temp (80-120 C)
    bounds = [(0, 10), (20, 40), (80, 120)]
    initial_guess = [5.0, 30.0, 100.0]

    res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return {
        'alumina_wt': round(res.x[0], 2),
        'jute_wt': round(res.x[1], 2),
        'curing_temp': round(res.x[2], 2)
    }

if __name__ == "__main__":
    generate_and_train()