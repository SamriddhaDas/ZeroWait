# ZeroWait
# ZOMATHON - Track 1: Kitchen Prep Time (KPT) Prediction

A beginner-friendly ML model that predicts how long a restaurant will take to prepare an order, enabling smarter rider dispatch and accurate customer ETAs.

---

## Problem Statement

Food delivery platforms face a core challenge: **when should a rider be dispatched?**

- Dispatch too early → rider waits at the restaurant
- Dispatch too late → food gets cold, customer waits longer

This model predicts **Kitchen Prep Time (KPT)** so that riders can be dispatched at the optimal moment — arriving at the restaurant just as the food is ready.

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│   restaurant_id · cuisine_type · order_size · hour_of_day      │
│         is_weekend · is_rush_hour · restaurant_avg_kpt          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                            │
│   One-Hot Encode (cuisine_type)  →  Train/Test Split (80/20)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RANDOM FOREST REGRESSOR                       │
│                                                                 │
│    Tree 1 ──┐                                                   │
│    Tree 2 ──┼──▶  Average Predictions  ──▶  KPT (minutes)      │
│    Tree N ──┘                                                   │
│              (100 decision trees)                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                             │
│          Mean Absolute Error (MAE)  ·  R² Score                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DISPATCH LOGIC ENGINE                         │
│                                                                 │
│   Order Placed                                                  │
│       │                                                         │
│       ├──▶  + Predicted KPT  ──▶  Food Ready Time              │
│       │                                  │                      │
│       └──▶  Food Ready - Travel Time ──▶ Rider Dispatch Time   │
│                                          │                      │
│                                          ▼                      │
│                              Customer ETA = Food Ready          │
│                               + Travel Time + 2 min buffer      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
│     Predicted KPT  ·  Rider Dispatch Time  ·  Customer ETA     │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ZeroWait/│
├── data/
│   ├── raw/
│   │   └── orders_raw.csv              # Raw order data from Zomato
│   ├── processed/
│   │   ├── orders_processed.csv        # Cleaned & encoded data
│   │   └── train_test_split/
│   │       ├── X_train.csv             # Training features
│   │       ├── X_test.csv              # Testing features
│   │       ├── y_train.csv             # Training labels (KPT)
│   │       └── y_test.csv              # Testing labels (KPT)
│
├── model/
│   ├── train.py                        # Model training script
│   ├── predict.py                      # Inference / prediction script
│   ├── evaluate.py                     # MAE, R² evaluation metrics
│   └── saved/
│       └── kpt_random_forest.pkl       # Serialized trained model
│
├── src/
│   ├── data_preprocessing.py           # Feature engineering & encoding
│   ├── dispatch_logic.py               # Rider dispatch time calculator
│   └── utils.py                        # Helper functions
│
├── notebooks/
│   └── kpt_exploration.ipynb           # EDA & experimentation notebook
│
├── tests/
│   ├── test_preprocessing.py           # Unit tests for data pipeline
│   ├── test_model.py                   # Unit tests for model predictions
│   └── test_dispatch.py                # Unit tests for dispatch logic
│
├── kpt_model.py                        # Main end-to-end ML pipeline
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## How It Works

The pipeline follows 6 steps:

### 1. Data Generation
Simulates 1,000 food orders with realistic features:

| Feature | Description |
|---|---|
| `restaurant_id` | Unique restaurant identifier (1–50) |
| `cuisine_type` | Indian, Chinese, Pizza, Burger, Biryani |
| `order_size` | Number of items in the order (1–10) |
| `hour_of_day` | Hour the order was placed (0–23) |
| `is_weekend` | 1 = weekend, 0 = weekday |
| `is_rush_hour` | 1 = lunch/dinner rush, 0 = otherwise |
| `restaurant_avg_kpt` | Restaurant's historical average prep time (10–40 min) |

### 2. Data Preparation
- Categorical features (`cuisine_type`) are one-hot encoded
- Data is split: **80% training / 20% testing**

### 3. Model Training
Uses a **Random Forest Regressor** (100 decision trees) to learn patterns from the training data.

### 4. Evaluation

| Metric | Meaning |
|---|---|
| Mean Absolute Error (MAE) | Average prediction error in minutes |
| R² Score | How well the model explains variance (1.0 = perfect) |

### 5. Rider Dispatch Logic

```
Dispatch Time = Order Placed + Predicted KPT - Rider Travel Time
Customer ETA  = Order Placed + Predicted KPT + Rider Travel Time + 2 min buffer
```

### 6. Demo Prediction
Runs a sample order through the trained model and outputs the predicted KPT, dispatch time, and customer ETA.

---

##Running the Model

### Prerequisites

```bash
pip install pandas numpy scikit-learn
```

### Run

```bash
python kpt_model.py
```

### Expected Output

```
Sample data created!
Totaal orders: 1000
Average KPT: ~27 minutes

Data prepared!
Training samples: 800
Testing samples: 200

Model trained successfully!

MODEL PERFORMANCE:
Mean Absolute Error: ~2.xx minutes
R² Score: ~0.xx

DEMO - Predicting for a new order:
Predicted Kitchen Prep Time : ~XX.X minutes
Dispatch rider at           : ~XX.X minutes after order
Customer ETA                : ~XX.X minutes from order placement

Result: Rider arrives at restaurant exactly when food is ready!
```

---

## Real-World Extensions

To improve this model for production use, consider:

- **More features**: weather, menu item complexity, kitchen staff count
- **Live data**: replace simulated data with real order history from Zomato
- **Per-restaurant models**: train separate models for high-volume restaurants
- **Time-series features**: rolling averages of recent prep times
- **Better algorithms**: XGBoost, LightGBM, or a neural network for higher accuracy

---

## License

Built for ZOMATHON Track 1. For educational and hackathon use.
