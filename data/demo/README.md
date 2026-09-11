# IBVAP-X Demo Video Sources & License Tracking

This document tracks the origin, license, creator, and scenario mapping for all realistic surveillance video clips used in demo presentations.

> **Note:** Video binaries (`.mp4`, `.avi`) are git-ignored to prevent repository bloat. Store videos locally in this `data/demo/` directory.

---

## Tracking Template

| Filename | Source Platform | Source URL | Creator | Download Date | License | Intended Scenario | Notes |
|---|---|---|---|---|---|---|---|
| `demo_normal.mp4` | Pixabay | https://pixabay.com/videos/... | [Creator] | YYYY-MM-DD | Pixabay License (Free use, no attribution) | Scenario 1: Normal Activity | Outdoor walking, public area |
| `demo_night.mp4` | Pexels | https://pexels.com/video/... | [Creator] | YYYY-MM-DD | Pexels License (Free use, no attribution) | Scenario 3: Night Loitering | Street scene at night |
| `demo_crowd.mp4` | Coverr | https://coverr.co/... | [Creator] | YYYY-MM-DD | Coverr License (Free commercial use) | Scenario 7: Cross-Camera | Pedestrian movement |

---

## Licensing Principles
- All demo videos must be sourced from platforms offering free, non-attribution commercial/research licenses (Pixabay, Pexels, Coverr).
- No video depicting identifiable individuals in a negative, offensive, or defamatory manner should be used.
- For unit testing, synthetic OpenCV shape videos are generated programmatically by `scripts/generate_test_videos.py`.
