import pytest
import os
import tempfile
import numpy as np
from backend.evidence.buffer import PreEventRingBuffer
from backend.evidence.hashing import EvidenceHasher
from backend.evidence.audit import AuditLogger
from backend.interfaces import Frame

def test_pre_event_ring_buffer():
    buffer = PreEventRingBuffer(max_seconds=2, fps=5.0)  # Max 10 frames
    assert buffer.max_frames == 10

    blank_img = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(15):
        frame_obj = Frame(camera_id="CAM-01", frame_id=i)
        buffer.add_frame(frame_obj, blank_img)

    frames = buffer.get_buffered_frames()
    assert len(frames) == 10  # Bounded to max 10 frames!
    assert frames[0][0].frame_id == 5  # Oldest frames evicted
    assert frames[-1][0].frame_id == 14

def test_evidence_hasher_and_tamper_demo():
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(b"ORIGINAL_EVIDENCE_BYTES_12345")
        test_path = f.name

    try:
        hash_orig = EvidenceHasher.compute_sha256(test_path)
        assert len(hash_orig) == 64

        # Verify integrity against original
        is_valid, comp = EvidenceHasher.verify_integrity(test_path, hash_orig)
        assert is_valid is True
        assert comp == hash_orig

        # Test tamper simulation demo (Correction 2: original stays untouched!)
        is_valid_demo, rec_h, comp_h = EvidenceHasher.simulate_tampering_demo(test_path, hash_orig)
        assert is_valid_demo is False
        assert rec_h == hash_orig
        assert comp_h != hash_orig

        # Confirm original file remains unchanged!
        is_valid_after, _ = EvidenceHasher.verify_integrity(test_path, hash_orig)
        assert is_valid_after is True

    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

def test_audit_logger():
    log_file = "data/evidence/test_audit_log.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = AuditLogger(log_path=log_file)
    logger.log_event("TEST_EVENT", "ADMIN", "TARGET_1", {"key": "val"})

    records = logger.read_audit_trail()
    assert len(records) == 1
    assert records[0]["event_type"] == "TEST_EVENT"
    assert records[0]["actor"] == "ADMIN"

    if os.path.exists(log_file):
        os.remove(log_file)
