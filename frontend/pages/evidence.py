import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from backend.evidence.hashing import EvidenceHasher
from backend.evidence.audit import AuditLogger


st.set_page_config(page_title="Evidence — IBVAP-X", page_icon="🔐", layout="wide")

st.title("🔐 Tamper-Evident Evidence Vault")
st.caption("Cryptographic SHA-256 Hash Verification & Audit Trail Inspection")

st.markdown("---")

# Demo mock evidence record
demo_evidence = {
    "evidence_id": "EVD-A8F912",
    "alert_id": "ALT-8F92A1",
    "camera_id": "CAM-03",
    "timestamp_str": "2026-09-11 19:42:12",
    "snapshot_path": "data/evidence/demo_snapshot.jpg",
    "sha256_hash": "A8C7F392B104E91A8C7F392B104E91A8C7F392B104E91A8C7F392B104E91A8C7",
    "is_sealed": True
}

# Ensure demo snapshot exists
os.makedirs("data/evidence", exist_ok=True)
if not os.path.exists(demo_evidence["snapshot_path"]):
    import cv2
    import numpy as np
    blank = np.zeros((300, 400, 3), dtype=np.uint8) + 80
    cv2.putText(blank, "SEALED EVIDENCE SNAPSHOT", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(demo_evidence["snapshot_path"], blank)
    # Compute real hash for demo file
    demo_evidence["sha256_hash"] = EvidenceHasher.compute_sha256(demo_evidence["snapshot_path"])

col1, col2 = st.columns([1, 2])

with col1:
    st.image(demo_evidence["snapshot_path"], caption=f"Snapshot ({demo_evidence['evidence_id']})", use_container_width=True)

with col2:
    st.subheader(f"Evidence Record #{demo_evidence['evidence_id']}")
    st.markdown(f"**Associated Alert:** {demo_evidence['alert_id']}")
    st.markdown(f"**Camera ID:** {demo_evidence['camera_id']}")
    st.markdown(f"**Sealed Timestamp:** {demo_evidence['timestamp_str']}")
    st.markdown(f"**Status:** `🔒 SEALED & READ-ONLY`")

    st.markdown("#### 🔑 SHA-256 Cryptographic Hash")
    st.code(demo_evidence["sha256_hash"], language="text")

    # Hash Verification Controls
    c_ver, c_tamper = st.columns(2)
    
    with c_ver:
        if st.button("✅ Verify Hash Integrity", key="btn_verify"):
            is_valid, computed = EvidenceHasher.verify_integrity(demo_evidence["snapshot_path"], demo_evidence["sha256_hash"])
            if is_valid:
                st.success("✅ **HASH VERIFIED:** File bytes match recorded cryptographic SHA-256 digest perfectly.")
            else:
                st.error("❌ **HASH MISMATCH:** File content has been altered!")

    with c_tamper:
        if st.button("⚠️ Demo: Simulate Tampering", key="btn_tamper"):
            # Trigger tamper simulation using temporary file copy
            is_valid, rec_h, comp_h = EvidenceHasher.simulate_tampering_demo(demo_evidence["snapshot_path"], demo_evidence["sha256_hash"])
            st.error("❌ **HASH MISMATCH DETECTED (Simulated Tampering):**")
            st.markdown(f"- **Recorded SHA-256:** `{rec_h[:24]}...`")
            st.markdown(f"- **Computed SHA-256:** `{comp_h[:24]}...`")
            st.warning("⚠️ **Alert:** Evidence file modification detected! Hash comparison failed. Original sealed file remains untouched.")

st.markdown("---")

st.subheader("📜 Append-Only Audit Log Trail")
audit_logger = AuditLogger()
audit_records = audit_logger.read_audit_trail(limit=10)

if audit_records:
    for rec in audit_records:
        st.markdown(f"- `[{rec['datetime_str']}]` **{rec['event_type']}** by `{rec['actor']}` (Target: `{rec['target_id']}`) — {rec['details']}")
else:
    st.info("Audit log initialized. Actions and evidence capture events will be logged here in append-only mode.")
