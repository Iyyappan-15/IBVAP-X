# IBVAP-X Honest Prototype Limitations

1. **Local Prototype Scope:** Designed for local/hackathon prototype evaluation; does not interface with physical military surveillance hardware.
2. **Simplified Geometric Coverage:** FOV coverage is computed in Euclidean metric coordinates assuming flat ground; does not calculate 3D terrain elevation or physical obstacles.
3. **Class-Based Cross-Camera Correlation:** Correlates events across adjacent cameras based on object class, time, and direction. Does not perform biometric re-identification.
4. **Prototype Anomaly Model:** IsolationForest model is trained on baseline trajectory distributions. Real-world deployment requires extensive domain dataset calibration.
5. **File Permissions on Windows:** `os.chmod` provides basic file-locking demonstration; enterprise immutability requires hardware WORM storage or blockchain ledger integration.
