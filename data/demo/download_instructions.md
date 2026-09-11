# Realistic Demo Video Download Instructions

To populate realistic demo videos for hackathon presentation:

1. **Visit Free Video Platforms:**
   - **Pixabay Videos:** https://pixabay.com/videos/
   - **Pexels Videos:** https://www.pexels.com/videos/
   - **Coverr:** https://coverr.co/

2. **Recommended Search Terms:**
   - `"people walking park"`
   - `"pedestrians street"`
   - `"night street pedestrian"`
   - `"crowd street"`

3. **Download 1–3 Short Clips (30–60 seconds, MP4 format):**
   - Save into `data/demo/` folder.
   - Rename clips to `demo_normal.mp4`, `demo_night.mp4`, etc.
   - Update `data/demo/README.md` with source URL and license details.

4. **Programmatic Degradation Note:**
   - You do NOT need separate dark or blurry video files for camera degradation testing.
   - Programmatic Gaussian blur and brightness attenuation are applied dynamically in `backend/reliability/frame_health.py` during demo execution.
