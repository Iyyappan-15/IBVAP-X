<div align="center">

<h1>🛡️ IBVAP-X</h1>
<h3>Reliability-Aware Border Video Intelligence</h3>

<p><strong>AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8n-Ultralytics-red?logo=github" />
  <img src="https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-green" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-81%20Passed-brightgreen?logo=pytest" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p><em>Smart India Hackathon 2026 · Problem Statement ID: 26187</em></p>

</div>

---

## 📖 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Why IBVAP-X?](#3-why-ibvap-x)
4. [System Architecture](#4-system-architecture)
5. [Core Intelligence Pipeline](#5-core-intelligence-pipeline)
6. [Key Features](#6-key-features)
7. [Technology Stack](#7-technology-stack)
8. [Project Structure](#8-project-structure)
9. [Quick Start](#9-quick-start)
10. [Running the Dashboard](#10-running-the-dashboard)
11. [Running the API](#11-running-the-api)
12. [Demo Scenarios](#12-demo-scenarios)
13. [Configuration Reference](#13-configuration-reference)
14. [API Reference](#14-api-reference)
15. [Database](#15-database)
16. [Testing](#16-testing)
17. [Docker Deployment](#17-docker-deployment)
18. [Security & Evidence Integrity](#18-security--evidence-integrity)
19. [Camera Coverage Gap Analysis](#19-camera-coverage-gap-analysis)
20. [Limitations & Scope](#20-limitations--scope)
21. [Future Enhancements](#21-future-enhancements)
22. [Team](#22-team)

---

## 1. Project Overview

**IBVAP-X** (Intelligent Border Video Analytics Platform — eXtended) is a **real, runnable, end-to-end prototype** for AI-powered border surveillance using existing CCTV camera infrastructure.

Rather than replacing expensive hardware, IBVAP-X adds an intelligence layer on top of any video feed — uploaded recordings, webcams, RTSP streams, or demo clips — and runs every frame through a **multi-stage analysis pipeline** that produces three independent, explainable outputs:

| Output | What it means |
|--------|--------------|
| 🔴 **Event Priority** (LOW / MEDIUM / HIGH / CRITICAL) | How serious is this event? Based on zone context, time, loitering, direction, cross-camera confirmation, and anomaly score. |
| 📡 **Camera Reliability** (GOOD / DEGRADED / POOR) | How trustworthy is this camera feed right now? Based on blur, brightness, frame health, and obstruction. |
| ⚡ **Actionability** (HIGH / MEDIUM / LOW) | Should an operator act immediately? Independent of priority — a HIGH priority alert from a POOR camera is still LOW actionability. |

All three values are always computed and displayed **independently** — they are never collapsed into a single score.

---

## 2. Problem Statement

**Smart India Hackathon 2026 — Problem ID: 26187**

> *"Design and develop an AI-Based Intelligent Video Analytics Platform for Border Surveillance that works with existing CCTV camera infrastructure. The system should detect intrusion attempts, identify suspicious activities, track objects across multiple cameras, assess camera health, and generate actionable intelligence for border security operators."*

**Key challenges addressed:**
- Existing CCTV infrastructure is heterogeneous (different resolutions, FPS, codecs, ages)
- Camera feeds degrade due to weather, obstruction, hardware failure — but alerts keep coming
- Operators are overwhelmed by false positives; need **explainable** priority triage
- Evidence must be tamper-evident for legal defensibility
- Coverage blind-spots must be identified proactively

---

## 3. Why IBVAP-X?

Most video analytics systems treat every camera as perfectly reliable. IBVAP-X's core innovation is **Reliability-Aware Intelligence**:

```
Traditional System:         IBVAP-X:
─────────────────────       ─────────────────────────────────────────────
Camera → Alert              Camera → Reliability Score
                                   ↓
                            Frame → YOLO → ByteTrack → Context Engine
                                   ↓
                            Priority Score (independent)
                                   ↓
                            Actionability (= f(Priority, Reliability))
                                   ↓
                            Explainable Alert: "CRITICAL priority, but
                            DEGRADED camera — MEDIUM actionability.
                            Deploy secondary unit to verify."
```

A **CRITICAL** event from a **POOR** camera is flagged differently from the same event on a **GOOD** camera. Operators receive context-aware action recommendations, not just alarm beeps.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VIDEO INPUTS                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Upload   │  │  RTSP    │  │ Webcam   │  │  Demo    │           │
│  │ MP4/AVI  │  │ Network  │  │ Device   │  │  Videos  │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       └─────────────┴─────────────┴──────────────┘                 │
│                          │ VideoSource ABC                          │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │   Video Frame Data    │ (OpenCV, frame-by-frame,    │
│              │   (NEVER full RAM)    │  no bulk load)              │
│              └──────────┬────────────┘                             │
│                         │                                           │
│        ┌────────────────┼────────────────┐                         │
│        ▼                ▼                ▼                          │
│  ┌───────────┐  ┌──────────────┐  ┌───────────────┐               │
│  │  Camera   │  │  YOLOv8n     │  │   Pre-Event   │               │
│  │Reliability│  │  Detector    │  │  Ring Buffer  │               │
│  │  Engine   │  │  (CPU-safe)  │  │  (10s × 10fps)│               │
│  └─────┬─────┘  └──────┬───────┘  └───────────────┘               │
│        │               ▼                                           │
│        │       ┌───────────────┐                                   │
│        │       │  ByteTrack    │  (Multi-Object Tracking           │
│        │       │  Tracker      │   with IoU fallback)              │
│        │       └──────┬────────┘                                   │
│        │              │                                            │
│        │    ┌─────────┴──────────┐                                │
│        │    ▼                    ▼                                 │
│        │  ┌──────────────┐  ┌──────────────┐                      │
│        │  │Context Engine│  │ Anomaly Engine│                     │
│        │  │  Zones       │  │(IsolationForest│                    │
│        │  │  Loitering   │  │  8-dim feature)│                    │
│        │  │  Time (D/N)  │  └──────┬─────────┘                   │
│        │  │  Direction   │         │                               │
│        │  └──────┬───────┘         │                              │
│        │         │                 │                               │
│        │         └────────┬────────┘                               │
│        │                  ▼                                        │
│        │       ┌──────────────────┐                               │
│        │       │ Cross-Camera     │                               │
│        │       │  Correlator      │                               │
│        │       └──────┬───────────┘                               │
│        │              │                                            │
│        └──────────────▼                                           │
│              ┌─────────────────────────────┐                      │
│              │       Priority Engine        │                      │
│              │  + Actionability Matrix      │                      │
│              │  + Explanation Generator     │                      │
│              └──────────────┬──────────────┘                      │
│                             ▼                                      │
│                   ┌──────────────────┐                            │
│                   │   Alert Manager  │ (deduplication,            │
│                   │  (State Machine) │  cooldown, routing)        │
│                   └────────┬─────────┘                            │
│                            │                                       │
│            ┌───────────────┼───────────────┐                      │
│            ▼               ▼               ▼                      │
│     ┌────────────┐  ┌────────────┐  ┌────────────────┐           │
│     │  Evidence  │  │  Operator  │  │  Auto-Escalation│          │
│     │  Capture   │  │  ACK / REJ │  │  (BOP→Command) │          │
│     │ (SHA-256)  │  │ / UNCERTAIN│  │                │           │
│     └────────────┘  └────────────┘  └────────────────┘           │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │               INDEPENDENT PARALLEL TRACK                    │  │
│  │  Camera Config + FOV → Coverage Gap Engine → Blind-spot Map │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┴──────────────────┐
              ▼                                   ▼
     ┌─────────────────┐               ┌──────────────────┐
     │  Streamlit UI   │               │   FastAPI REST   │
     │  (Dashboard)    │               │   (Port 8000)    │
     └─────────────────┘               └──────────────────┘
```

> **Architecture Principle:** `IBVAPXPipeline` is the sole orchestrator. Streamlit and FastAPI are both just **clients** — the intelligence layer is completely UI-independent.

---

## 5. Core Intelligence Pipeline

### 5.1 Video Ingestion — `VideoSource` Abstraction

All video inputs share a common `VideoSource` ABC. The pipeline is **source-agnostic**.

```python
# backend/detection/video_stream.py
VideoSource (ABC)
├── FileVideoSource    ← uploaded MP4/AVI/MOV (primary prototype input)
├── WebcamVideoSource  ← local webcam device
├── RTSPVideoSource    ← network camera stream
└── DemoVideoSource    ← synthetic demo clips
```

**Upload rules (addendum):**
- Max file size: `200 MB` (configurable `MAX_UPLOAD_SIZE_MB`)
- Max duration: `180s` (configurable `MAX_VIDEO_DURATION_SECONDS`)
- Recommended: `MP4 H.264 @ 720p`
- Frames are **never loaded into RAM in bulk** — processed one by one via `cv2.VideoCapture`
- Temp files use UUID-based names (path traversal prevention)

### 5.2 Object Detection — YOLOv8n

```
Frame (NumPy) → ObjectDetector → List[Detection]
```

- Model: `YOLOv8n` (nano — CPU-safe, ~6 MB)
- Detects: `person`, `car`, `motorcycle`, `bus`, `truck`
- Configurable confidence threshold (default `0.50`)
- High-resolution frames auto-resized for inference only (evidence kept at full resolution)

### 5.3 Multi-Object Tracking — ByteTrack

```
List[Detection] → ObjectTracker → List[Track]
```

- Primary: **ByteTrack** (Supervision library)
- Fallback: **Pure IoU Tracker** (zero additional dependencies)
- Each track maintains full `trajectory` history (list of center coordinates)
- Track IDs persist across frames for loitering and direction analysis

### 5.4 Context Engine

Five sub-engines run in parallel on each track:

| Sub-engine | What it detects |
|-----------|----------------|
| **Zones** (`zones.py`) | Is the object inside a restricted polygon zone? (Shapely, normalized 0–1 coords) |
| **Loitering** (`loitering.py`) | Has the object stayed in an area for > `LOITERING_THRESHOLD_SECONDS`? (with cooldown spam prevention) |
| **Time Context** (`time_context.py`) | Is it DAY or NIGHT? (configurable hour boundaries) |
| **Direction** (`direction.py`) | Is the object moving TOWARD_BOUNDARY, AWAY, LATERAL, or UNCERTAIN? (cosine similarity on trajectory) |
| **Context Engine** (`context_engine.py`) | Aggregates into a `ContextEvent` schema |

### 5.5 Camera Reliability Engine

Runs **in parallel** to detection — not in the detection path. Even if no objects are detected, reliability is always computed.

| Component | Signal | Score Weight |
|-----------|--------|-------------|
| Blur (`blur.py`) | Laplacian variance — blurry = low score | 35% |
| Brightness (`brightness.py`) | Grayscale mean — too dark or washed out = low | 25% |
| Frame Health (`frame_health.py`) | Gap and frame-drop detection | 25% |
| Obstruction (`obstruction.py`) | Uniform-patch std-dev heuristic | 15% |

**Composite reliability score → CameraStatus:**
- ≥ 80 → `GOOD`
- 50–79 → `DEGRADED`
- 20–49 → `POOR`
- < 20 → `OFFLINE`

**Camera Degradation Demo:** Enable via checkbox → programmatic blur + darkness applied (no separate blurry video needed).

### 5.6 Anomaly Detection — IsolationForest

```
Track trajectory → 8-dim feature vector → IsolationForest → AnomalyResult
```

**8-dimensional feature vector:**
1. Speed (pixels/second)
2. Direction change rate
3. Trajectory length
4. Bounding box area
5. Area variance
6. Time in restricted zone
7. Distance from zone boundary
8. Loitering flag

- Score normalised to `[0.0, 1.0]`
- Non-blocking — exceptions are isolated with `logger.warning()`, never crash the pipeline
- Model retrained via `scripts/train_anomaly_model.py`
- Output clearly labelled: `⚠️ Prototype anomaly model`

### 5.7 Cross-Camera Correlation

```
ContextEvent → EventCorrelator → CorrelatedEvent | None
```

- Time-window class-based correlation (`CORRELATION_WINDOW_SECONDS = 40s`)
- Uses camera adjacency graph from `data/config/cameras.json`
- Explicitly annotated as **class match, not biometric identity re-identification**
- Correlated events contribute `+15` priority points

### 5.8 Priority Engine — Three Independent Outputs

```
ContextEvent + ReliabilityScore + AnomalyResult + CorrelationResult
    → EventPriority   (LOW / MEDIUM / HIGH / CRITICAL)     [0-100]
    → CameraStatus    (GOOD / DEGRADED / POOR / OFFLINE)   [0-100]
    → Actionability   (HIGH / MEDIUM / LOW)
    → ExplainableAlert ("Why: Zone entry at night, object loitering 45s...")
```

**Priority scoring factors:**

| Factor | Max Points |
|--------|-----------|
| In restricted zone | +30 |
| Night-time context | +20 |
| Loitering detected | +20 |
| Moving toward boundary | +15 |
| Cross-camera confirmed | +15 |
| Anomaly score > threshold | +10 |
| **Total (capped at 100)** | **100** |

**Actionability matrix:**

| Priority \ Reliability | GOOD | DEGRADED | POOR |
|----------------------|------|----------|------|
| CRITICAL | HIGH | MEDIUM | LOW |
| HIGH | HIGH | MEDIUM | LOW |
| MEDIUM | MEDIUM | LOW | LOW |
| LOW | LOW | LOW | LOW |

### 5.9 Alert Manager & Evidence

```
AlertOutput → AlertManager → Deduplication → State Machine
                   ↓
            [HIGH / CRITICAL]
                   ↓
            PreEventRingBuffer (10s pre-event frames)
                   ↓
            EvidenceCapturer → MP4 clip + JPEG snapshot + JSON metadata
                   ↓
            SHA-256 hash → chmod read-only → AuditLogger (JSONL)
```

Evidence is **tamper-evident**:
- SHA-256 computed on MP4 clip immediately after capture
- File made read-only (`os.chmod`)
- Append-only audit log
- Tampering demo: flips one byte in a temp copy, detects MISMATCH, deletes temp copy

### 5.10 Human-in-the-Loop Verification

```
Alert → Operator Dashboard
    → [ACKNOWLEDGE] → closes alert
    → [REJECT]      → marks false positive
    → [UNCERTAIN]   → flags for supervisory review
    → [timeout]     → auto-escalates BOP → Command Centre
```

- Timeout configurable (`ACK_TIMEOUT_SECONDS`)
- Escalation path: Border Outpost → Command Centre
- Full RBAC: `bop_operator` / `cmd_operator` / `admin` roles

---

## 6. Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **YOLOv8n Detection** | Real-time object detection — person, vehicle, motorcycle |
| 🔄 **ByteTrack Tracking** | Persistent multi-object tracking with trajectory history |
| 🗺️ **Zone Intelligence** | Shapely polygon restricted zones per camera |
| 🌙 **Day/Night Context** | Automatic time-of-day risk adjustment |
| 🚶 **Loitering Detection** | Threshold + cooldown to prevent alert spam |
| 🧭 **Direction Analysis** | Cosine-similarity toward-boundary detection |
| 📡 **Camera Reliability** | 4-signal real-time camera health scoring |
| 🔀 **Cross-Camera Correlation** | Time-window adjacent camera event linking |
| 🤖 **Anomaly Detection** | IsolationForest on 8-dim trajectory features |
| 🎖️ **3-Value Decision Model** | Priority + Reliability + Actionability always independent |
| 💬 **Explainable Alerts** | Human-readable "why" for every alert |
| 🔒 **Tamper-Evident Evidence** | SHA-256 + read-only + audit trail |
| 👤 **RBAC Authentication** | Argon2id hashing, role-based dashboard access |
| ↗️ **Auto-Escalation** | BOP → Command Centre on operator timeout |
| 🗺️ **Coverage Gap Map** | Folium map with 3-tier blind-spot detection |
| 📤 **MP4 Upload Flow** | Full validation pipeline (size, duration, resolution, codec) |
| 🐳 **Docker Ready** | `docker-compose up` for full stack deployment |
| 🌐 **FastAPI REST** | Full REST API for external system integration |
| 💾 **Auto DB Fallback** | PostgreSQL primary → SQLite fallback, zero config change |

---

## 7. Technology Stack

### Backend

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Object Detection | [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) |
| Multi-Object Tracking | [Supervision ByteTrack](https://github.com/roboflow/supervision) |
| Zone Geometry | [Shapely](https://shapely.readthedocs.io/) |
| Anomaly Detection | [scikit-learn IsolationForest](https://scikit-learn.org/) |
| REST API | [FastAPI](https://fastapi.tiangolo.com/) |
| Data Validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Password Hashing | [Argon2-cffi](https://argon2-cffi.readthedocs.io/) |
| Coverage Maps | [Folium](https://python-visualization.github.io/folium/) |
| Image Processing | [OpenCV](https://opencv.org/) |

### Frontend

| Component | Technology |
|-----------|-----------|
| Dashboard | [Streamlit](https://streamlit.io/) |
| Interactive Maps | Folium embedded in Streamlit |
| Real-time Metrics | Streamlit metrics + progress bars |

### Database

| Component | Technology |
|-----------|-----------|
| ORM | [SQLAlchemy 2.x](https://www.sqlalchemy.org/) |
| Primary DB | PostgreSQL (`psycopg` v3 async-compatible) |
| Fallback DB | SQLite (automatic, zero config) |
| Migrations | Schema auto-create via SQLAlchemy metadata |

### Infrastructure

| Component | Technology |
|-----------|-----------|
| Containerisation | Docker + Docker Compose |
| Config | `pydantic-settings` + `.env` file |
| Testing | pytest (81 tests) |
| Evidence Storage | Local filesystem + SHA-256 integrity |
| Audit Log | Append-only JSONL file |

---

## 8. Project Structure

```
IBVAP-X/
│
├── backend/                    # Core intelligence layer
│   ├── __init__.py
│   ├── interfaces.py           # All Pydantic schemas (Frame, Detection, Track,
│   │                           #   ContextEvent, ReliabilityScore, AlertOutput,
│   │                           #   AnomalyResult, EvidenceRecord, ...)
│   ├── pipeline.py             # IBVAPXPipeline — sole orchestrator
│   │
│   ├── config/
│   │   └── settings.py         # Pydantic Settings (reads .env)
│   │
│   ├── db/
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── session.py          # Auto PostgreSQL → SQLite fallback
│   │   ├── repository.py       # CRUD operations
│   │   └── offline_queue.py    # PENDING_SYNC offline queue
│   │
│   ├── detection/
│   │   ├── video_stream.py     # VideoSource ABC + File/Webcam/RTSP/Demo sources
│   │   ├── detector.py         # ObjectDetector (YOLOv8n wrapper)
│   │   ├── tracker.py          # ObjectTracker (ByteTrack + IoU fallback)
│   │   └── schemas.py          # VideoFrameData dataclass
│   │
│   ├── context/
│   │   ├── zones.py            # Shapely polygon zone engine
│   │   ├── loitering.py        # Loitering threshold + cooldown
│   │   ├── time_context.py     # DAY/NIGHT classifier
│   │   ├── direction.py        # Cosine similarity direction engine
│   │   └── context_engine.py   # Aggregator → ContextEvent
│   │
│   ├── reliability/
│   │   ├── blur.py             # Laplacian variance blur scorer
│   │   ├── brightness.py       # Grayscale mean brightness scorer
│   │   ├── frame_health.py     # Frame gap/drop health scorer
│   │   ├── obstruction.py      # Uniform-patch obstruction heuristic
│   │   └── reliability_score.py # Composite engine + apply_demo_degradation()
│   │
│   ├── anomaly/
│   │   ├── features.py         # 8-dim trajectory feature extractor
│   │   └── detector.py         # IsolationForest wrapper + calibration
│   │
│   ├── scoring/
│   │   ├── priority_engine.py  # Factor-based priority scorer
│   │   ├── actionability.py    # Actionability matrix (priority × reliability)
│   │   └── explanations.py     # Human-readable alert explanations
│   │
│   ├── alerts/
│   │   ├── alert_manager.py    # Deduplication + state machine
│   │   ├── acknowledgement.py  # Operator ACK/REJ/UNCERTAIN
│   │   └── escalation.py       # Auto-escalation BOP → Command Centre
│   │
│   ├── evidence/
│   │   ├── buffer.py           # PreEventRingBuffer (deque, bounded)
│   │   ├── capture.py          # MP4 + JPEG + JSON + SHA-256 capture
│   │   ├── hashing.py          # SHA-256 + verify_integrity + tamper demo
│   │   └── audit.py            # Append-only JSONL audit logger
│   │
│   ├── correlation/
│   │   ├── camera_graph.py     # Camera adjacency graph
│   │   └── event_correlator.py # Time-window cross-camera correlator
│   │
│   ├── coverage/
│   │   ├── fov.py              # Local metric FOV wedge calculator
│   │   └── coverage_engine.py  # 3-tier coverage map engine
│   │
│   ├── auth/
│   │   └── auth.py             # Argon2id auth + RBAC session manager
│   │
│   ├── upload/
│   │   ├── video_validator.py  # 8-step video validation pipeline
│   │   └── safe_temp_storage.py # UUID temp files, cleanup, path-traversal guard
│   │
│   └── api/
│       ├── app.py              # FastAPI app (CORS, lifespan, routers)
│       └── routes/
│           ├── cameras.py
│           ├── alerts.py
│           ├── evidence.py
│           ├── coverage.py
│           └── analysis.py
│
├── frontend/
│   ├── dashboard.py            # Streamlit multi-page app entry point
│   └── pages/
│       ├── live_monitoring.py  # Primary upload + analysis page
│       ├── alerts.py           # Alert management page
│       ├── evidence.py         # Evidence viewer + tamper demo
│       ├── camera_health.py    # Reliability metrics page
│       ├── coverage_map.py     # Folium blind-spot map
│       ├── overview.py         # Live metrics dashboard
│       ├── settings.py         # Runtime configuration page
│       └── login.py            # RBAC login page
│
├── data/
│   ├── config/
│   │   └── cameras.json        # 3 cameras with zone polygons + adjacency
│   ├── demo/                   # Synthetic demo video clips (gitignored)
│   ├── models/                 # YOLOv8n weights (gitignored)
│   ├── evidence/               # Captured evidence (gitignored)
│   ├── anomaly_models/         # Trained IsolationForest model
│   └── uploads_temp/           # Temporary upload storage (gitignored)
│
├── scripts/
│   ├── generate_test_videos.py # Creates 3 demo MP4 clips
│   ├── download_model.py       # Downloads YOLOv8n weights
│   ├── train_anomaly_model.py  # Trains IsolationForest on synthetic data
│   ├── seed_demo_data.py       # Seeds DB with demo users, cameras, alert
│   ├── check_compat.py         # Verifies all 14 packages are importable
│   └── run_demo.py             # One-command 9-scenario demo launcher
│
├── tests/
│   ├── unit/                   # 50+ unit tests
│   │   ├── test_detector.py
│   │   ├── test_tracker.py
│   │   ├── test_context_*.py
│   │   ├── test_reliability_*.py
│   │   ├── test_priority_*.py
│   │   ├── test_evidence_*.py
│   │   ├── test_auth.py
│   │   └── test_upload_validator.py
│   ├── integration/            # End-to-end milestone tests
│   │   ├── test_milestone_a.py
│   │   ├── test_milestone_b.py
│   │   ├── test_milestone_c.py
│   │   ├── test_milestone_d.py
│   │   ├── test_milestone_e.py
│   │   └── test_detector_real.py
│   └── api/
│       └── test_api.py
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── demo.md
│   ├── limitations.md
│   └── future-enhancements.md
│
├── .env.example                # Template — copy to .env and edit
├── .env                        # Local secrets (gitignored)
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 9. Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.13.x |
| Git | Any |
| (Optional) PostgreSQL | 15+ |
| (Optional) Docker | 24+ |

> PostgreSQL is **optional**. If it's not running, the system automatically falls back to SQLite — no configuration change needed.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Iyyappan-15/IBVAP-X.git
cd IBVAP-X
```

### Step 2 — Create a Python 3.13 virtual environment

```powershell
# Windows PowerShell
py -3.13 -m venv .venv
.venv\Scripts\activate
python --version   # Should print Python 3.13.x
```

```bash
# macOS / Linux
python3.13 -m venv .venv
source .venv/bin/activate
python --version
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Create your local `.env`

```powershell
copy .env.example .env
# Edit .env if you want to change any defaults
```

### Step 5 — Download the YOLO model

```powershell
$env:PYTHONPATH="."; python scripts/download_model.py
```

The script downloads `yolov8n.pt` (~6 MB) to `data/models/`. If it fails (no internet), place the file manually in `data/models/yolov8n.pt`.

### Step 6 — Generate demo videos

```powershell
$env:PYTHONPATH="."; python scripts/generate_test_videos.py
```

Creates:
- `data/demo/test_normal.mp4` — normal patrol activity
- `data/demo/test_zone.mp4` — restricted zone entry
- `data/demo/test_degraded.mp4` — camera degradation scenario

### Step 7 — Seed demo data (users, cameras, sample alert)

```powershell
$env:PYTHONPATH="."; python scripts/seed_demo_data.py
```

Creates demo accounts:

| Username | Password | Role |
|----------|----------|------|
| `bop_operator` | `demo1234` | Border Outpost Operator |
| `cmd_operator` | `demo5678` | Command Centre Operator |
| `admin` | `admin9012` | System Administrator |

### Step 8 — Train the anomaly model

```powershell
$env:PYTHONPATH="."; python scripts/train_anomaly_model.py
```

---

## 10. Running the Dashboard

```powershell
$env:PYTHONPATH="."; .venv\Scripts\streamlit run frontend/dashboard.py
```

Opens at: **http://localhost:8501**

### Dashboard Pages

| Page | Path | Description |
|------|------|-------------|
| 🏠 **Overview** | `/overview` | Live metrics cards, camera status grid |
| 🎥 **Live Monitoring** | `/live_monitoring` | Upload video + full analysis pipeline |
| 🚨 **Alerts** | `/alerts` | Alert feed with priority + reliability cards |
| 🔒 **Evidence** | `/evidence` | SHA-256 hashes, tamper demo button |
| 📡 **Camera Health** | `/camera_health` | Reliability score breakdown per camera |
| 🗺️ **Coverage Map** | `/coverage_map` | Folium blind-spot map |
| ⚙️ **Settings** | `/settings` | ACK timeout, reliability weights, anomaly calibration |
| 🔑 **Login** | `/login` | RBAC authentication |

---

## 11. Running the API

```powershell
$env:PYTHONPATH="."; .venv\Scripts\uvicorn backend.api.app:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## 12. Demo Scenarios

### Primary Workflow — Video Upload

1. Open the **Live Monitoring** page in the dashboard
2. Select **"📤 Upload Video File (Recommended)"** in the sidebar
3. Upload any MP4/AVI/MOV file (≤ 200 MB, ≤ 180 seconds)
4. Review the **Video Metadata Preview** (filename, size, duration, resolution, FPS, codec, estimated analysis time)
5. Optionally enable **"⚡ Simulate Camera Degradation"** in the sidebar
6. Click **▶️ START ANALYSIS**
7. Watch the annotated frame feed with live stats (frames, FPS, tracked objects, events, reliability)
8. Click **⏹ STOP ANALYSIS** at any time — partial results are preserved
9. After completion, use **VIEW ALERTS / VIEW EVIDENCE / VIEW CAMERA HEALTH / VIEW FULL REPORT** navigation

### Pre-built Demo Scenarios

Run all 9 scenarios at once:

```powershell
$env:PYTHONPATH="."; python scripts/run_demo.py
```

| Scenario | Camera | Description |
|----------|--------|-------------|
| 1 | CAM-01 | Normal patrol — person walking patrol path |
| 2 | CAM-02 | Restricted zone entry → HIGH priority alert |
| 3 | CAM-01 + CAM-02 | Cross-camera correlation between adjacent cameras |
| 4 | CAM-03 | Degraded camera → POOR reliability → LOW actionability |
| 5 | CAM-01 | Night-time loitering → CRITICAL priority |
| 6 | All | Multi-camera simultaneous event processing |
| 7 | CAM-02 | Operator acknowledgement → evidence capture |
| 8 | CAM-01 | Operator rejection → false positive logging |
| 9 | CAM-03 | ACK timeout → auto-escalation to Command Centre |

---

## 13. Configuration Reference

All settings are read from `.env` via `pydantic-settings`. Override any value by editing `.env`.

### Video Upload & Processing

```env
MAX_UPLOAD_SIZE_MB=200          # Reject uploads larger than this
MAX_VIDEO_DURATION_SECONDS=180  # Reject videos longer than this
MIN_VIDEO_DURATION_SECONDS=5    # Warn if shorter than this
PROCESS_FPS=5.0                 # Frames to process per second (skip rest)
DISPLAY_FPS=5.0                 # Max display refresh rate
MAX_INFERENCE_WIDTH=1280        # Resize for inference if wider (never upscale)
MAX_INFERENCE_HEIGHT=720        # Resize for inference if taller (never upscale)
UPLOAD_CAMERA_ID=CAM-UPLOAD-01  # Camera ID assigned to uploaded video
VIDEO_TEMP_DIR=data/uploads_temp # Temp storage for upload files
```

### Object Detection

```env
YOLO_MODEL_PATH=data/models/yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.50
YOLO_DETECT_CLASSES=person,car,motorcycle,bus,truck
```

### Context Engine

```env
DAY_START_HOUR=6
NIGHT_START_HOUR=18
LOITERING_THRESHOLD_SECONDS=30
LOITERING_COOLDOWN_SECONDS=60
DIRECTION_MIN_FRAMES=10
```

### Camera Reliability

```env
BLUR_THRESHOLD=50.0             # Below = camera flagged as blurry
BRIGHTNESS_MIN=40.0             # Below = too dark
BRIGHTNESS_MAX=220.0            # Above = washed out
RELIABILITY_WEIGHT_BLUR=0.35
RELIABILITY_WEIGHT_BRIGHTNESS=0.25
RELIABILITY_WEIGHT_FRAME=0.25
RELIABILITY_WEIGHT_OBSTRUCTION=0.15
RELIABILITY_GOOD_THRESHOLD=80.0
RELIABILITY_DEGRADED_THRESHOLD=50.0
RELIABILITY_POOR_THRESHOLD=20.0
```

### Alert & Evidence

```env
ACK_TIMEOUT_SECONDS=30
PRE_EVENT_SECONDS=10
POST_EVENT_SECONDS=10
EVIDENCE_STORAGE_DIR=data/evidence
CORRELATION_WINDOW_SECONDS=40
```

### Anomaly Detection

```env
ANOMALY_CONTAMINATION=0.10
ANOMALY_SCORE_THRESHOLD=0.50
ANOMALY_MODEL_PATH=data/anomaly_models/ibvapx_anomaly_model.pkl
```

### Database

```env
DATABASE_MODE=auto              # auto | postgres | sqlite
DB_URL_POSTGRES=postgresql+psycopg://postgres:postgres@localhost:5432/ibvapx_dev
DB_URL_SQLITE=sqlite:///./data/ibvapx_local.db
```

---

## 14. API Reference

The FastAPI backend exposes a REST API at `http://localhost:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/cameras` | List all cameras |
| `GET` | `/cameras/{id}` | Get camera details |
| `GET` | `/alerts` | List all alerts |
| `GET` | `/alerts/{id}` | Get alert details |
| `POST` | `/alerts/{id}/acknowledge` | Acknowledge alert |
| `POST` | `/alerts/{id}/reject` | Reject alert |
| `GET` | `/evidence` | List all evidence records |
| `GET` | `/evidence/{id}` | Get evidence details |
| `GET` | `/coverage` | Get coverage map data |
| `POST` | `/analysis/upload` | Submit video for analysis |

Full interactive docs: **http://localhost:8000/docs** (Swagger UI)

---

## 15. Database

### Schema (SQLAlchemy ORM)

```
User          — Authentication + RBAC roles
Camera        — Camera configurations + zone polygons
CameraHealth  — Per-frame reliability scores (time-series)
Event         — Context events (zone, loitering, direction)
Alert         — Alert records with 3-value decision output
Evidence      — SHA-256 sealed evidence records
OperatorAction — ACK / REJ / UNCERTAIN operator decisions
Escalation    — Auto-escalation records
```

### Auto-Fallback Logic

```python
DATABASE_MODE=auto   # Try PostgreSQL → fall back to SQLite automatically
DATABASE_MODE=postgres  # Force PostgreSQL (fail if unavailable)
DATABASE_MODE=sqlite    # Force SQLite always
```

No code change is needed to switch between databases.

---

## 16. Testing

### Run all tests

```powershell
$env:PYTHONPATH="."; .venv\Scripts\python -m pytest --tb=short -q
```

**Current result:** ✅ **81 tests passed**

### Run a specific test group

```powershell
# Unit tests only
$env:PYTHONPATH="."; python -m pytest tests/unit/ -v

# Integration (milestone) tests only
$env:PYTHONPATH="."; python -m pytest tests/integration/ -v

# API tests only
$env:PYTHONPATH="."; python -m pytest tests/api/ -v

# Upload validation tests only
$env:PYTHONPATH="."; python -m pytest tests/unit/test_upload_validator.py -v
```

### Test Coverage by Module

| Module | Tests | Coverage |
|--------|-------|---------|
| Video Upload Validation | 31 | Extension, size, decoder, duration, resolution, path traversal |
| Object Detection | 7 | Real YOLOv8n inference on real frames |
| Multi-Object Tracking | 11 | ByteTrack + IoU fallback |
| Context Engine | 17 | Zones, loitering, time, direction |
| Priority Engine | 25 | 3-value independence, score factors |
| Evidence | 34 | SHA-256, tamper demo, audit log |
| API | 4 | Health, cameras, alerts endpoints |
| Milestone A–E | 5 | End-to-end pipeline gates |

### Milestone Gates (CI-style validation)

| Gate | Sprint | Tests | Validates |
|------|--------|-------|-----------|
| Milestone A | Sprint 5 | 17 | Full context pipeline |
| Milestone B | Sprint 7 | 25 | 3-value independence verified |
| Milestone C | Sprint 10 | 34 | Evidence + Auth |
| Milestone D | Sprint 13 | 44 | Coverage map |
| Milestone E | Sprint 17 | 50 | Full end-to-end system |

---

## 17. Docker Deployment

### Full stack (App + PostgreSQL)

```bash
docker-compose up --build
```

- Dashboard: **http://localhost:8501**
- API: **http://localhost:8000**
- PostgreSQL: `localhost:5432` (internal)

### App only (with external Postgres or SQLite fallback)

```bash
docker build -t ibvapx .
docker run -p 8000:8000 -p 8501:8501 ibvapx
```

### `docker-compose.yml` services

```yaml
services:
  app:      # FastAPI + Streamlit
  postgres: # PostgreSQL 15
```

---

## 18. Security & Evidence Integrity

### Password Security
- Passwords hashed with **Argon2id** (memory-hard, OWASP recommended)
- Never stored in plaintext
- Session token validation on every API call

### Evidence Integrity Chain

```
1. Frame captured at event detection moment
2. Pre-event ring buffer provides 10s of prior context
3. MP4 clip + JPEG snapshot written to disk
4. SHA-256 hash computed immediately on the MP4 file
5. File set to read-only (os.chmod 0o444)
6. Hash + metadata written to append-only JSONL audit log
7. Evidence record marked is_sealed = True
```

**Tampering detection demo** (in the Evidence page):
```
Click [⚠️ Demo: Simulate Tampering]
→ Creates a temp copy of the evidence file
→ Flips 1 byte at offset 1024
→ Re-computes SHA-256 → MISMATCH detected
→ Deletes temp copy (original untouched)
→ Displays: "TAMPER DETECTED" with original vs computed hashes
```

### RBAC Roles

| Role | Access |
|------|--------|
| `BOP_OPERATOR` | View alerts, acknowledge/reject, view evidence |
| `COMMAND_OPERATOR` | All BOP access + view escalations |
| `ADMIN` | Full system access, configure cameras |

---

## 19. Camera Coverage Gap Analysis

```
Camera Config (lat/lon, heading, FOV angle, range)
    ↓
LocalFOVCalculator (Euclidean metric, GPS → meter transform)
    ↓
GeometricCoverageEngine
    ↓
Folium Map with 3-tier coverage labels:
    ✅ "2+ modeled camera FOVs" — HIGH confidence, dual coverage
    ⚠️ "1 modeled camera FOV"  — MEDIUM confidence, single coverage
    🔴 "0 modeled camera FOVs" — BLIND SPOT, no coverage
```

> **Important:** The coverage map uses **geometric approximation** with simulated GPS coordinates. The mandatory banner reads: *"⚠️ GEOMETRIC CAMERA-COVERAGE APPROXIMATION — Demo Simulation Data"*

This demonstrates the algorithmic approach for identifying coverage blind spots — in production, real survey data and camera specifications would replace the simulation inputs.

---

## 20. Limitations & Scope

This is a **college hackathon prototype** built for demonstration purposes.

| Limitation | Production Solution |
|-----------|-------------------|
| YOLOv8n detection only (no face recognition) | YOLOv8x or custom-trained model |
| Cross-camera correlation is class-based, NOT biometric re-ID | Actual re-ID model (ReID networks) |
| Coverage map uses simulated GPS data | Real survey coordinates + camera specs |
| Anomaly model trained on synthetic trajectories | Train on real patrol data over weeks |
| Single-machine, single-camera processing | Distributed stream processing (Kafka + Flink) |
| No encryption for evidence files | AES-256 + signed evidence package |
| Webcam/RTSP UI in informational mode only | Full WebRTC / RTSP reader integration |
| No geofencing or actual GPS tracking | GPS collar integration for border patrol |

---

## 21. Future Enhancements

- [ ] **Real biometric re-ID** — Replace class-based correlation with ResNet/OSNet re-identification
- [ ] **Thermal camera integration** — Night-vision + IR sensor fusion
- [ ] **Distributed processing** — Apache Kafka + Flink for multi-site, multi-camera stream ingestion
- [ ] **Drone feed integration** — UAV video input via RTSP or MAVLink
- [ ] **Mobile app for operators** — Push notifications for HIGH/CRITICAL alerts
- [ ] **Federated learning** — Train anomaly models across border posts without sharing raw footage
- [ ] **Satellite imagery fusion** — Cross-reference detected events with satellite change detection
- [ ] **Advanced coverage** — Replace geometric FOV approximation with real photogrammetric models
- [ ] **Encrypted evidence** — AES-256 signed evidence bundles for legal admissibility
- [ ] **Language support** — Hindi, regional language operator interface

---

## 22. Team

**Project:** IBVAP-X — Reliability-Aware Border Video Intelligence  
**Event:** Smart India Hackathon 2026  
**Problem ID:** 26187

---

<div align="center">

**Built for Smart India Hackathon 2026**

*"Don't just detect. Understand. Explain. Prioritise. Act."*

</div>
