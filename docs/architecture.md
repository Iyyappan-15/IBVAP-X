# IBVAP-X System Architecture & Design Specification

## Overview
IBVAP-X is a **Reliability-Aware Border Video Intelligence Platform** designed for Smart India Hackathon Problem Statement 26187.
Unlike traditional surveillance demos that collapse confidence into single threat scores, IBVAP-X maintains **three independent decision dimensions**:

1. **Event Priority:** LOW / MEDIUM / HIGH / CRITICAL (What occurred)
2. **Camera Reliability:** GOOD / DEGRADED / POOR / OFFLINE (Observation quality)
3. **Actionability:** HIGH / MEDIUM / LOW (Recommended operational handling)

---

## High-Level Pipeline Architecture

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

## Priority Scoring Policy Model

| Factor | Points | Rationale |
|---|---|---|
| Restricted zone entry | +30 | Core detection trigger |
| Night-time event | +20 | Elevated operational concern |
| Loitering above threshold | +20 | Sustained deliberate presence |
| Movement toward boundary | +15 | Direction context |
| Cross-camera confirmation | +15 | Corroborating evidence |
| Anomaly engine signal | +10 | Secondary unsupervised signal |

> **Disclaimer:** *These scoring weights are configurable prototype policy parameters. They demonstrate explainable, transparent decision logic and do not represent empirically validated threat probabilities.*
