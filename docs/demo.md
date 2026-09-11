# IBVAP-X Presentation Demo Guide

## Quick Presentation Run

Run the automated one-command demo script to verify system state:
```bash
python scripts/run_demo.py
```

Launch the interactive operator dashboard:
```bash
streamlit run frontend/dashboard.py
```

---

## 9 Presentation Scenarios Sequence

1. **Normal Activity Baseline (CAM-01):** Smooth walking; Priority LOW; No false alerts.
2. **Restricted Zone Entry:** Zone overlay highlighted; Priority MEDIUM/HIGH.
3. **Night + Loitering Alert:** 3 reason chips stacked; Loitering duration counter.
4. **Camera Degradation (Live Toggle):** Gaussian blur applied live → Camera Reliability drops 94% → 42% (DEGRADED); Actionability drops HIGH → MEDIUM.
5. **Evidence Capture & SHA-256 Tamper Simulation:** Click `[⚠️ Demo: Simulate Tampering]` → `❌ HASH MISMATCH` displayed with recorded & computed hashes; original file untouched.
6. **Operator Timeout & Escalation:** 10s countdown bar expires → `ESCALATED TO COMMAND CENTRE` badge.
7. **Cross-Camera Correlation:** Event matched across CAM-01/02/03 within 40s window.
8. **Coverage Gap Map:** Folium map renders green/yellow/red polygons; BLIND SPOT highlighted.
9. **Anomaly Calibration & Alert:** Click `[Calibrate Normal Behaviour]` → unusual path triggers Anomaly Score 0.83+ with `⚠️ Prototype Model` label.
