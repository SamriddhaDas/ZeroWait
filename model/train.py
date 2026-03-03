"""
train.py — Train the KPT Random Forest model and save it to disk.
Run this once before starting the API server.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_data(n=2000, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        "restaurant_id":      np.random.randint(1, 50, n),
        "cuisine_type":       np.random.choice(["Indian", "Chinese", "Pizza", "Burger", "Biryani"], n),
        "order_size":         np.random.randint(1, 10, n),
        "hour_of_day":        np.random.randint(0, 24, n),
        "is_weekend":         np.random.choice([0, 1], n),
        "is_rush_hour":       np.random.choice([0, 1], n),
        "restaurant_avg_kpt": np.random.uniform(10, 40, n),
    })
    data["actual_kpt_minutes"] = (
        data["order_size"] * 2.5 +
        data["is_rush_hour"] * 8 +
        data["is_weekend"] * 3 +
        data["restaurant_avg_kpt"] * 0.5 +
        np.random.normal(0, 3, n)
    ).clip(5, 60)
    return data

def train():
    print("🔄 Generating training data...")
    data = generate_data()

    # One-hot encode cuisine
    data = pd.get_dummies(data, columns=["cuisine_type"])

    X = data.drop("actual_kpt_minutes", axis=1)
    y = data["actual_kpt_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊 Model Performance:")
    print(f"   MAE  : {mae:.2f} min")
    print(f"   RMSE : {rmse:.2f} min")
    print(f"   R²   : {r2:.4f}")

    # Save model
    model_path = os.path.join(SAVE_DIR, "kpt_random_forest.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved → {model_path}")

    # Save feature columns & metrics
    meta = {
        "feature_columns": list(X.columns),
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
        "n_estimators": 150,
        "training_samples": len(X_train),
    }
    meta_path = os.path.join(SAVE_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata saved → {meta_path}")
    return meta

if __name__ == "__main__":
    train()
