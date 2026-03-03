# ZeroWait

Predicts kitchen prep time (KPT) and optimal rider dispatch timing using a Random Forest model.

---

## Project Structure

```
zerowait/
├── api/
│   ├── __init__.py
│   └── app.py            ← FastAPI server
├── model/
│   ├── __init__.py
│   ├── train.py          ← Train & save the model
│   └── saved/            ← Auto-created on first train
│       ├── kpt_random_forest.pkl
│       └── model_meta.json
├── static/
│   └── index.html        ← Dashboard UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start.sh              ← Local one-command start
└── README.md
```

---

## Option 1 — Run Locally

**Requirements:** Python 3.10+

```bash
# From the zerowait/ root folder:
pip install -r requirements.txt

# Train the model (creates model/saved/)
python model/train.py

# Start the server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper script (does all three steps):
```bash
bash start.sh
```

Then open: http://localhost:8000

---

## Option 2 — Docker (recommended for deployment)

```bash
# Build (also trains the model inside the image)
docker build -t zerowait .

# Run
docker run -p 8000:8000 zerowait
```

Or with Docker Compose:
```bash
docker-compose up --build
```

Then open: http://localhost:8000

---

## Option 3 — Deploy to a Cloud VM (e.g. AWS EC2, DigitalOcean, GCP)

1. SSH into your server
2. Install Docker: `sudo apt install docker.io docker-compose -y`
3. Copy (or git clone) this folder onto the server
4. Run: `docker-compose up --build -d`
5. Open port 8000 in your firewall / security group

---

## Option 4 — Deploy to Railway / Render / Fly.io (free tiers)

These platforms auto-detect a `Dockerfile` and deploy it:

| Platform     | Steps |
|--------------|-------|
| **Railway**  | Connect GitHub repo → New Project → Deploy |
| **Render**   | New Web Service → Docker → set port 8000 |
| **Fly.io**   | `fly launch` → `fly deploy` |

---

## API Endpoints

| Method | Path       | Description               |
|--------|------------|---------------------------|
| GET    | `/`        | Dashboard UI              |
| GET    | `/health`  | API + model status        |
| GET    | `/metrics` | Model performance metrics |
| POST   | `/predict` | Get KPT prediction        |

### POST /predict — Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant_id": 5,
    "cuisine_type": "Indian",
    "order_size": 3,
    "hour_of_day": 13,
    "is_weekend": 0,
    "is_rush_hour": 1,
    "restaurant_avg_kpt": 25.0,
    "rider_travel_time": 10.0
  }'
```

**Response:**
```json
{
  "predicted_kpt": 28.4,
  "dispatch_at_minutes": 18.4,
  "customer_eta": 40.4,
  "confidence": "high"
}
```

---

## Environment Variables (optional)

| Variable       | Default                   | Description                    |
|----------------|---------------------------|--------------------------------|
| `APP_BASE_DIR` | Auto-detected from app.py | Override root path for model   |
| `STATIC_DIR`   | `{BASE_DIR}/static`       | Override path to dashboard     |

---

## Model Details

- **Algorithm:** RandomForestRegressor (150 trees, max_depth=12)
- **Features:** restaurant_id, cuisine_type (one-hot), order_size, hour_of_day, is_weekend, is_rush_hour, restaurant_avg_kpt
- **Target:** kitchen prep time in minutes (clipped 5–60 min)
- **Typical R²:** ~0.87–0.92
## License

Built for ZOMATHON Track 1. For educational and hackathon use.
