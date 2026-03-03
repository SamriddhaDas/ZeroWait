# ============================================================
# ZOMATHON - Track 1: Kitchen Prep Time (KPT) Prediction
# Beginner-Friendly ML Model
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: CREATE SAMPLE DATA
# (In real life, Zomato would give you this data)
# ============================================================

np.random.seed(42)
n = 1000  # 1000 orders

data = pd.DataFrame({
    "restaurant_id": np.random.randint(1, 50, n),
    "cuisine_type": np.random.choice(["Indian", "Chinese", "Pizza", "Burger", "Biryani"], n),
    "order_size": np.random.randint(1, 10, n),          # number of items
    "hour_of_day": np.random.randint(0, 24, n),         # 0 to 23
    "is_weekend": np.random.choice([0, 1], n),          # 0=weekday, 1=weekend
    "is_rush_hour": np.random.choice([0, 1], n),        # lunch/dinner rush
    "restaurant_avg_kpt": np.random.uniform(10, 40, n), # restaurant's historical avg prep time
})

# Create realistic KPT (target variable) with some logic + noise
data["actual_kpt_minutes"] = (
    data["order_size"] * 2.5 +
    data["is_rush_hour"] * 8 +
    data["is_weekend"] * 3 +
    data["restaurant_avg_kpt"] * 0.5 +
    np.random.normal(0, 3, n)  # random noise
).clip(5, 60)  # keep between 5 and 60 minutes

print("✅ Sample data created!")
print(f"   Total orders: {len(data)}")
print(f"   Average KPT: {data['actual_kpt_minutes'].mean():.1f} minutes\n")

# ============================================================
# STEP 2: PREPARE DATA FOR ML MODEL
# ============================================================

# Convert text columns to numbers (ML needs numbers, not words)
data = pd.get_dummies(data, columns=["cuisine_type"])

# Separate features (inputs) and target (what we want to predict)
X = data.drop("actual_kpt_minutes", axis=1)   # inputs
y = data["actual_kpt_minutes"]                 # output (KPT)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data prepared!")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}\n")

# ============================================================
# STEP 3: TRAIN THE ML MODEL
# ============================================================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!\n")

# ============================================================
# STEP 4: EVALUATE THE MODEL
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 MODEL PERFORMANCE:")
print(f"   Mean Absolute Error: {mae:.2f} minutes")
print(f"   R² Score: {r2:.2f} (closer to 1.0 = better)\n")

# ============================================================
# STEP 5: RIDER DISPATCH LOGIC
# ============================================================

def calculate_dispatch_time(order_placed_time_minutes, predicted_kpt, avg_rider_travel_time=10):
    """
    When should we dispatch the rider?
    Dispatch = Order placed + Predicted KPT - Rider travel time
    So rider arrives JUST as food is ready!
    """
    dispatch_at = order_placed_time_minutes + predicted_kpt - avg_rider_travel_time
    customer_eta = order_placed_time_minutes + predicted_kpt + avg_rider_travel_time + 2  # +2 min buffer
    return max(0, dispatch_at), customer_eta

# ============================================================
# STEP 6: DEMO - PREDICT FOR A NEW ORDER
# ============================================================

print("🍕 DEMO - Predicting for a new order:")
print("-" * 40)

# Create a sample new order (same columns as training data)
sample_order = pd.DataFrame([{
    "restaurant_id": 12,
    "order_size": 4,
    "hour_of_day": 13,       # 1 PM (lunch rush)
    "is_weekend": 1,
    "is_rush_hour": 1,
    "restaurant_avg_kpt": 25,
    "cuisine_type_Biryani": 0,
    "cuisine_type_Burger": 0,
    "cuisine_type_Chinese": 0,
    "cuisine_type_Indian": 1,  # Indian cuisine
    "cuisine_type_Pizza": 0,
}])

# Make sure columns match training data
for col in X.columns:
    if col not in sample_order.columns:
        sample_order[col] = 0
sample_order = sample_order[X.columns]

predicted_kpt = model.predict(sample_order)[0]
dispatch_time, eta = calculate_dispatch_time(
    order_placed_time_minutes=0,   # order placed at time 0
    predicted_kpt=predicted_kpt,
    avg_rider_travel_time=10
)

print(f"   Predicted Kitchen Prep Time : {predicted_kpt:.1f} minutes")
print(f"   Dispatch rider at           : {dispatch_time:.1f} minutes after order")
print(f"   Customer ETA                : {eta:.1f} minutes from order placement")
print()
print("🎯 Result: Rider arrives at restaurant exactly when food is ready!")
print("✅ No more waiting. Better ETA. Happy customers!")
