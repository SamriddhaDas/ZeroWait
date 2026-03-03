from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ZeroWait — KPT Prediction API",
    description="Predicts kitchen prep time and optimal rider dispatch time.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ─────────────────────────────────────────────────────
# Works whether run from repo root, inside api/, or from /app in Docker
BASE_DIR   = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved", "kpt_random_forest.pkl")
META_PATH  = os.path.join(BASE_DIR, "model", "saved", "model_meta.json")

try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    FEATURE_COLS = meta["feature_columns"]
    MODEL_METRICS = meta["metrics"]
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("⚠️  Model not found. Run: python model/train.py")
    model = None
    FEATURE_COLS = []
    MODEL_METRICS = {}

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    restaurant_id:      int   = Field(default=1,    ge=1,  le=100)
    cuisine_type:       str   = Field(default="Indian")
    order_size:         int   = Field(default=3,    ge=1,  le=20)
    hour_of_day:        int   = Field(default=13,   ge=0,  le=23)
    is_weekend:         int   = Field(default=0,    ge=0,  le=1)
    is_rush_hour:       int   = Field(default=1,    ge=0,  le=1)
    restaurant_avg_kpt: float = Field(default=25.0, ge=5,  le=60)
    rider_travel_time:  float = Field(default=10.0, ge=1,  le=30)

class PredictResponse(BaseModel):
    predicted_kpt:       float
    dispatch_at_minutes: float
    customer_eta:        float
    confidence:          str

# ── Helpers ───────────────────────────────────────────────────────────────────
VALID_CUISINES = ["Indian", "Chinese", "Pizza", "Burger", "Biryani"]

def build_feature_row(req: PredictRequest) -> dict:
    row = {
        "restaurant_id":      req.restaurant_id,
        "order_size":         req.order_size,
        "hour_of_day":        req.hour_of_day,
        "is_weekend":         req.is_weekend,
        "is_rush_hour":       req.is_rush_hour,
        "restaurant_avg_kpt": req.restaurant_avg_kpt,
    }
    for c in VALID_CUISINES:
        row[f"cuisine_type_{c}"] = 1 if req.cuisine_type == c else 0
    # fill any missing columns the model expects
    return {col: row.get(col, 0) for col in FEATURE_COLS}

def dispatch_and_eta(kpt: float, rider_travel: float):
    dispatch_at = max(0.0, kpt - rider_travel)
    eta = kpt + rider_travel + 2.0
    return round(dispatch_at, 1), round(eta, 1)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return {"metrics": MODEL_METRICS, "feature_columns": FEATURE_COLS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run python model/train.py first.")
    if req.cuisine_type not in VALID_CUISINES:
        raise HTTPException(400, f"cuisine_type must be one of {VALID_CUISINES}")

    row   = build_feature_row(req)
    X     = np.array([[row[c] for c in FEATURE_COLS]])
    kpt   = float(model.predict(X)[0])
    kpt   = round(max(5.0, min(60.0, kpt)), 1)

    dispatch_at, eta = dispatch_and_eta(kpt, req.rider_travel_time)

    # Confidence based on order complexity
    if req.is_rush_hour and req.order_size > 6:
        confidence = "medium"
    elif req.order_size <= 3:
        confidence = "high"
    else:
        confidence = "high"

    return PredictResponse(
        predicted_kpt=kpt,
        dispatch_at_minutes=dispatch_at,
        customer_eta=eta,
        confidence=confidence,
    )

# ── Serve static dashboard ────────────────────────────────────────────────────
STATIC_DIR = os.environ.get("STATIC_DIR", os.path.join(BASE_DIR, "static"))
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_dashboard():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"message": "ZeroWait API running. POST /predict to use."})
