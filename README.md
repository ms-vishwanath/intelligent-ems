# Intelligent Emergency Response System (Intelligent EMS)

An end-to-end emergency management stack that ingests 911-like calls, predicts medical severity via ML, evaluates optimal ambulance assignments with OR-Tools, and exposes async FastAPI endpoints plus Dockerized infrastructure for production-style testing.

---

## 1. Why This Exists

- **Real-time triage** — classify emergencies by severity using a trained RandomForest model with a heuristic fallback.
- **Resource optimization** — select the best ambulance with OR-Tools (distance + availability weighting) and Haversine / OSRM ETAs.
- **Operational readiness** — asynchronous Postgres layer, background logging, Docker Compose stack, and API-level tests.

---

## 2. Highlights

| Area | Capabilities |
|------|--------------|
| API | FastAPI async routes for events, dispatch, ambulance listings, and stateless prediction endpoints (`/predict/*`). |
| ML | `ml/train.py` loads `sample_data.csv`, engineers features, trains `RandomForestClassifier`, and writes `ml/model.pkl`. |
| Optimization | OR-Tools routing solver with greedy fallback, OSRM integration, background dispatch logging. |
| Infra | Dockerfile + docker-compose (API + Postgres + OSRM), `.env` support, sample JSON payloads (`test_requests/`). |
| Tests | `pytest` suites for API smoke coverage and optimizer unit validation. |

---

## 3. Directory Walkthrough

```
intelligent_ems/
├── app/                     # FastAPI application layer
│   ├── api.py               # Routes & orchestration
│   ├── db.py                # Async SQLAlchemy models/session
│   ├── model_service.py     # Model loader + inference + fallback heuristics
│   ├── optimizer.py         # OR-Tools / fallback dispatch optimizer
│   ├── schemas.py           # Pydantic request/response models
│   └── utils.py             # OSRM helpers
├── ml/
│   ├── train.py             # Training entry point
│   ├── features.py          # Feature builder shared with inference
│   ├── sample_data.csv      # Demo dataset
│   └── model.pkl            # Generated model artifact (after training)
├── infra/
│   ├── docker-compose.yml   # API + Postgres + OSRM
│   └── Dockerfile           # FastAPI container
├── tests/
│   ├── test_api.py          # Endpoint smoke tests (gracefully handle missing DB)
│   └── test_optimizer.py    # Unit tests for optimizer logic
├── test_requests/           # Ready-made JSON payloads for predict APIs
├── requirements.txt
├── README.md
└── setup.sh                 # Opinionated local bootstrap script
```

---

## 4. Quickstart (Local)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python ml/train.py          # builds ml/model.pkl
python -c "from app.db import init_db; import asyncio; asyncio.run(init_db())"
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

API docs at `http://localhost:8000/docs`.

---

## 5. Docker Workflow

```bash
cd infra
docker-compose up --build
# Services: FastAPI (8000), Postgres (5432), OSRM (5000)
```

Stop & clean:

```bash
docker-compose down          # stop
docker-compose down -v       # stop + remove volumes
```

Logs:

```bash
docker-compose logs -f api
```

---

## 6. Environment Variables

All optional; defaults reside in code.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | Async Postgres DSN | `postgresql+asyncpg://postgres:postgres@localhost:5432/ems_db` |
| `MODEL_PATH` | Path to ML artifact | `ml/model.pkl` |
| `OSRM_URL` | Routing host | `http://localhost:5000` |

`.env` example:

```bash
cat <<'EOF' > .env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ems_db
MODEL_PATH=ml/model.pkl
OSRM_URL=http://osrm:5000
EOF
```

`setup.sh` automates venv creation, dependency install, and model training.

---

## 7. API Surface

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Basic metadata |
| `GET` | `/health` | Reports API health & model load status |
| `POST` | `/event` | Persist event, run ML prediction, optional auto-dispatch (`?auto_dispatch=true`) |
| `GET` | `/ambulances` | List resources (`?available_only=true`) |
| `POST` | `/dispatch` | Dispatch ambulance for existing event |
| `POST` | `/predict/severity` | Stateless severity inference (no DB writes) |
| `POST` | `/predict/dispatch` | Stateless severity + ambulance recommendation |

### Examples

```bash
# Health
curl http://localhost:8000/health

# Create event
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{"patient_age": 45, "patient_gender": "M", ... }'

# Stateless predict (JSON lives in test_requests/)
curl -X POST http://localhost:8000/predict/severity \
  -H "Content-Type: application/json" \
  -d @test_requests/predict_severity.json

curl -X POST http://localhost:8000/predict/dispatch \
  -H "Content-Type: application/json" \
  -d @test_requests/predict_dispatch.json
```

`/predict/dispatch` reuses the same optimizer logic but skips persistence so it’s safe for simulations.

---

## 8. ML Pipeline

1. `ml/sample_data.csv` → `ml/train.py` (RandomForestClassifier)
2. `ml/features.py` used in both training and inference
3. `ml/model.pkl` loaded by `app/model_service.py`

Training output snippet:

```
Loaded 41 samples with 14 features
Class distribution: [ 1 15  9 16]
Warning: Using non-stratified split (minimum class count: 1)
Model Accuracy: 0.7778
```

The service tracks whether predictions used the trained model or heuristic fallback (`used_fallback` flag in API responses).

---

## 9. Dispatch Optimization

- Candidate generation from `ambulances` table (status + geo).
- OR-Tools assignment minimizing weighted distance and availability penalty.
- Greedy fallback if OR-Tools unavailable.
- ETA via OSRM (`/route/v1/driving`) with Haversine fallback.
- Background dispatch logging for auditing (`DispatchLog` table).

---

## 10. Database Schema (DBML)

```dbml
Table ambulances {
  id                  int         [pk, increment]
  vehicle_id          varchar(50) [not null, unique]
  current_lat         float       [not null]
  current_lon         float       [not null]
  status              varchar(20) [not null, default: 'available']
  equipment_level     varchar(20) [not null, default: 'basic']
  crew_size           int         [not null, default: 2]
  is_available        boolean     [not null, default: true]
  created_at          timestamp   [not null, default: `now()`]
  updated_at          timestamp   [not null, default: `now()`]
}

Table events {
  id                   int         [pk, increment]
  patient_age          int         [not null]
  patient_gender       varchar(10) [not null]
  location_lat         float       [not null]
  location_lon         float       [not null]
  reported_symptoms    text        [not null]
  caller_phone         varchar(20)
  incident_type        varchar(50) [not null, default: 'medical']
  predicted_severity   float
  severity_category    varchar(20)
  assigned_ambulance_id int        [ref: > ambulances.id]
  status               varchar(20) [not null, default: 'pending']
  created_at           timestamp   [not null, default: `now()`]
  updated_at           timestamp   [not null, default: `now()`]
}

Table dispatch_logs {
  id                       int       [pk, increment]
  event_id                 int       [not null, ref: > events.id]
  ambulance_id             int       [not null, ref: > ambulances.id]
  estimated_arrival_minutes float    [not null]
  distance_km              float     [not null]
  severity_score           float     [not null]
  created_at               timestamp [not null, default: `now()`]
}
```

---

## 11. Tests

```bash
source venv/bin/activate
pytest tests/
```

`test_api.py` gracefully handles missing infrastructure (e.g., returns 500 if DB absent). `test_optimizer.py` exercises Haversine, ETA, greedy fallback, and OR-Tools wrapper.

---

## 12. Troubleshooting Cheatsheet

| Issue | Steps |
|-------|-------|
| Model not loaded | Run `python ml/train.py`, ensure `MODEL_PATH` points to generated `ml/model.pkl`. |
| DB connection errors | Verify Postgres service, check `DATABASE_URL`, run `python -c "from app.db import init_db; ..."` |
| OSRM unreachable | Confirm container `ems_osrm`, update `OSRM_URL`, API falls back to Haversine. |
| OR-Tools missing | Install via requirements (already listed). Fallback optimizer handles runtime gaps. |
| Tests failing due to infra | Ensure Postgres running or accept 500 responses in API tests (by design). |

---

## 13. Production Notes

- Run Postgres/OSRM via managed services or hardened VMs.
- Configure HTTPS, auth, rate limiting, and proper CORS policies.
- Externalize secrets and add observability (Prometheus, OpenTelemetry, etc.).
- Scale FastAPI workers via Gunicorn/Uvicorn workers or containers.

---

## 14. Support

- Docs: `http://localhost:8000/docs`
- Alternative view: `http://localhost:8000/redoc`
- For questions, open an issue or reach out to your platform team.

Happy hacking! 🚑
ntelligent EMS is a FastAPI-based emergency response platform that predicts medical severity in real time and optimizes ambulance dispatching. It combines async Postgres, OSRM travel estimates, and a trained ML model to deliver actionable insights fast.
Key Features
Async FastAPI API with event intake, resource listings, dispatch, and stateless prediction endpoints
RandomForestClassifier severity model with fallback heuristics plus reusable feature builder & training script
OR-Tools-powered ambulance optimizer with Haversine/OSRM ETA calculation and graceful fallbacks
PostgreSQL async ORM models, dispatch logging, and background tasking for auditability
Ready-to-use Docker Compose stack (API, Postgres, OSRM) and sample JSON requests for quick testing
Primary Algorithms
Severity prediction: RandomForestClassifier (scikit-learn)
Dispatch optimization: Google OR-Tools routing solver with distance/availability weighting# intelligent-ems
