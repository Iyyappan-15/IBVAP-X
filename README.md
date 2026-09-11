# IBVAP-X — Reliability-Aware Border Video Intelligence
### Smart India Hackathon | Problem Statement ID: 26187

> **Title:** AI-Based Intelligent Video Analytics Platform for Border Surveillance using Existing CCTV Infrastructure  
> **Status:** Full Runnable College / Hackathon Prototype Specification & Implementation Complete

---

## 1. Executive Summary & Core Concept
**IBVAP-X** is an intelligent, reliability-aware border video surveillance analytics platform. Unlike traditional CCTV systems that collapse confidence scores or generate constant false alarms on degraded feeds, IBVAP-X maintains **three independent decision dimensions**:

```
┌────────────────────────────────────────────────────────┐
│ EVENT PRIORITY SCORE         0 – 100                   │
│ What happened and how significant                      │
│ LOW / MEDIUM / HIGH / CRITICAL                         │
├────────────────────────────────────────────────────────┤
│ CAMERA RELIABILITY SCORE     0 – 100                   │
│ How trustworthy is the observation                     │
│ GOOD / DEGRADED / POOR / OFFLINE                       │
├────────────────────────────────────────────────────────┤
│ ACTIONABILITY                                          │
│ Recommended operational handling based on              │
│ event priority AND observation reliability             │
│ HIGH / MEDIUM / LOW                                    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Key Differentiators
- **Independent 3-Value Decision Outputs:** Never collapses camera condition into event priority.
- **Parallel Stream Architecture:** Context Engine, Camera Reliability Engine, and Anomaly Engine process streams in parallel.
- **Automatic Dual Database Engine (`DATABASE_MODE=auto`):** Tries primary PostgreSQL (`localhost:5432`); automatically falls back to local SQLite buffer if PostgreSQL is unreachable.
- **Tamper-Evident SHA-256 Evidence Vault:** Auto-captures clips/snapshots, computes cryptographic SHA-256 hashes, and includes a live `[⚠️ Demo: Simulate Tampering]` feature.
- **Local Metric FOV Coverage Engine:** Computes 3-tier camera coverage wedges in Euclidean meters to eliminate lat/lon distortion.
- **Explainable Reason Chips:** Every alert outputs structured human-readable `why_reasons` for operator auditability.

---

## 3. Technology Stack

| Layer | Library / Framework | Purpose |
|---|---|---|
| Runtime | Python 3.13 / 3.14 | Core language environment |
| Object Detection | Ultralytics YOLOv8n | Real-time object detection |
| Tracking | Supervision ByteTrack + NumPy IoU | Multi-object trajectory tracking |
| Anomaly Detection | Scikit-Learn IsolationForest | Unsupervised trajectory outlier scoring |
| Geospatial Geometry | Shapely + Folium | Metric FOV wedge & coverage mapping |
| Primary Database | PostgreSQL via psycopg (v3) | Production ORM storage |
| Fallback Database | SQLite | Local zero-config demo storage |
| User Interface | Streamlit | Operator dashboard UI |
| REST API | FastAPI + Uvicorn | External integration API |

---

## 4. System Architecture Diagram

```
                         CCTV / VIDEO
                              │
                              ▼
                       VIDEO INGESTION
             (Uploaded MP4 / Webcam / RTSP / Demo)
                              │
                              ▼
                         YOLO DETECTION
                 (configurable model & classes)
                              │
                              ▼
                         BYTETRACK
                  (stable IDs, trajectories)
                              │
                     TRACK / EVENT STREAM
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
         CONTEXT         RELIABILITY       ANOMALY
         ENGINE          ENGINE            ENGINE
         Zone/Loitering  Blur/Brightness   IsolationForest
         Time/Direction  Frame Health      (optional, parallel)
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                       EVIDENCE FUSION
                    (combines all 3 streams)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
           LOCAL EVIDENCE          CROSS-CAMERA CORRELATION
           (this camera)           (class + adjacency + time)
                 └────────────┬────────────┘
                              ▼
                       PRIORITY ENGINE
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        EVENT PRIORITY   CAMERA           ACTIONABILITY
        LOW/MED/HIGH/    RELIABILITY      (recommended
        CRITICAL         GOOD/DEGRADED/   operational
                         POOR/OFFLINE     handling)
                              ▼
                      EXPLAINABLE ALERT
                  WHAT/WHERE/WHEN/WHY/
                  HOW RELIABLE/EVIDENCE/ACTION
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
             EVIDENCE      OPERATOR     ESCALATION
             CAPTURE        REVIEW       MACHINE
             Ring Buffer   ACK/REJECT/   NEW→ACTIVE
             Pre+Event     UNCERTAIN     →ESCALATED
             +Post Clip         │
             SHA-256        ⏱️ Timer
             SEALED         Countdown
                 │              │
             AUDIT LOG      ESCALATION
             (insert-only)  BOP→Command
```

---

## 5. Quickstart & Installation

### Prerequisites
- Python 3.13 or 3.14
- Git
- pgAdmin 4 / PostgreSQL (Optional — automatic SQLite fallback active if PostgreSQL is off)

### Step 1: Clone & Environment Setup
```bash
git clone https://github.com/Iyyappan-15/IBVAP-X.git
cd IBVAP-X

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
```

### Step 2: Install Dependencies & Verify Environment
```bash
pip install -r requirements.txt
python scripts/check_compat.py
```

### Step 3: Seed Demo Data & Models
```bash
python scripts/download_model.py
python scripts/generate_test_videos.py
python scripts/train_anomaly_model.py
python scripts/seed_demo_data.py
```

---

## 6. How to Run the Presentation Demo

### One-Command Presentation Test
```bash
python scripts/run_demo.py
```

### Launch Interactive Streamlit Operator Dashboard
```bash
streamlit run frontend/dashboard.py
```
Open browser at: `http://localhost:8501`

### Launch FastAPI Backend REST Server
```bash
uvicorn backend.api.app:app --reload --port 8000
```
Open API documentation at: `http://localhost:8000/docs`

---

## 7. 9 Presentation Demo Scenarios
1. **Normal Activity Baseline:** Smooth walking; Priority LOW; No false alerts.
2. **Restricted Zone Entry:** Zone highlighted; Priority MEDIUM/HIGH.
3. **Night + Loitering Alert:** 3 reason chips stacked; Loitering timer counter.
4. **Camera Degradation (Live Toggle):** Gaussian blur applied live → Camera Reliability drops 94% → 42% (DEGRADED); Actionability drops HIGH → MEDIUM.
5. **Evidence Capture & SHA-256 Tamper Simulation:** Click `[⚠️ Demo: Simulate Tampering]` → `❌ HASH MISMATCH` displayed with recorded & computed hashes; original file untouched.
6. **Operator Timeout & Escalation:** 10s countdown bar expires → `ESCALATED TO COMMAND CENTRE` badge.
7. **Cross-Camera Correlation:** Event matched across CAM-01/02/03 within 40s window.
8. **Coverage Gap Map:** Folium map renders green/yellow/red polygons; BLIND SPOT highlighted.
9. **Anomaly Calibration & Alert:** Click `[Calibrate Normal Behaviour]` → unusual path triggers Anomaly Score 0.83+ with `⚠️ Prototype Model` label.

---

## 8. Docker Deployment
```bash
docker compose up --build
```
Access dashboard at `http://localhost:8501` and API at `http://localhost:8000`.

---

## 9. Running Automated Test Suites
```bash
pytest tests/unit/ tests/integration/ tests/api/
```

---

## 10. Project Structure
```
IBVAP-X/
├── backend/
│   ├── api/             ← FastAPI routes & app
│   ├── alerts/          ← Alert manager, ACK, escalation
│   ├── anomaly/         ← IsolationForest trajectory detector
│   ├── auth/            ← Argon2id hashing & session management
│   ├── config/          ← Pydantic settings (.env)
│   ├── context/         ← Zones, loitering, time, direction
│   ├── correlation/     ← Camera graph & temporal correlation
│   ├── coverage/        ← Local metric FOV engine
│   ├── db/              ← SQLAlchemy ORM models, session, repository
│   ├── detection/       ← YOLO detector, ByteTrack tracker, VideoSource
│   ├── evidence/        ← Ring buffer, SHA-256 hasher, audit log
│   ├── reliability/     ← Blur, brightness, frame health, obstruction
│   ├── scoring/         ← Priority engine, actionability matrix
│   ├── interfaces.py    ← Shared Pydantic data schemas
│   └── pipeline.py      ← Independent orchestrator
├── data/
│   ├── anomaly_models/  ← Pre-trained IsolationForest weights
│   ├── config/          ← cameras.json
│   ├── demo/            ← Demo videos & licensing README
│   ├── evidence/        ← Captured evidence files & audit logs
│   └── models/          ← YOLOv8n weights
├── docs/                ← Architecture, API, Demo, Limitations docs
├── frontend/
│   ├── dashboard.py     ← Streamlit main entrypoint
│   └── pages/           ← Overview, Live, Alerts, Health, Coverage, Evidence, Settings, Login
├── scripts/             ← Demo scripts & data generators
├── tests/               ← Unit, Integration, and API pytest suites
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 11. Licensing & Video Credits
See `data/demo/README.md` for full attribution and license details of free demo videos sourced from Pixabay, Pexels, and Coverr.

---

## 12. Prototype vs. Production Distinction
This software is a college/hackathon prototype developed for evaluation under Problem Statement 26187. See `docs/limitations.md` and `docs/future-enhancements.md` for complete scope details.
