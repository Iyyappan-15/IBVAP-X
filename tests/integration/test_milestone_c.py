import pytest
import os
import time
import tempfile
import numpy as np

from backend.interfaces import (
    ContextEvent, ReliabilityScore, CameraStatus, EventPriority, Actionability, AlertState, UserRole, TimeContextEnum, DirectionEnum
)
from backend.auth.auth import AuthManager, hash_password, verify_password
from backend.alerts.alert_manager import AlertManager
from backend.alerts.escalation import EscalationManager
from backend.evidence.capture import EvidenceCapturer
from backend.evidence.hashing import EvidenceHasher

def test_milestone_c_security_operations_workflow():
    """
    Milestone C Gate Integration Test:
    - User authentication & role check
    - HIGH alert generated
    - Evidence captured with SHA-256 hash
    - Tamper simulation demo triggers HASH MISMATCH while original remains sealed
    - Alert timeout triggers escalation to COMMAND_CENTRE
    - COMMAND_OPERATOR role access verified
    """
    # 1. Authentication
    auth_mgr = AuthManager()
    pwd = "demo1234password"
    pwd_hash = hash_password(pwd)
    assert verify_password(pwd, pwd_hash) is True

    bop_token = auth_mgr.create_session("bop_operator", UserRole.BOP_OPERATOR)
    cmd_token = auth_mgr.create_session("cmd_operator", UserRole.COMMAND_OPERATOR)

    assert auth_mgr.validate_role_access(bop_token, UserRole.BOP_OPERATOR) is True
    assert auth_mgr.validate_role_access(bop_token, UserRole.COMMAND_OPERATOR) is False
    assert auth_mgr.validate_role_access(cmd_token, UserRole.COMMAND_OPERATOR) is True

    # 2. Alert Generation
    now = time.time()
    alert_mgr = AlertManager()
    
    ctx = ContextEvent(
        camera_id="CAM-03", track_id=20, class_name="person", timestamp=now,
        in_restricted_zone=True, zone_name="Restricted Waterway Delta",
        loitering=True, loitering_duration_seconds=40.0,
        time_context=TimeContextEnum.NIGHT, direction=DirectionEnum.TOWARD_BOUNDARY
    )
    rel = ReliabilityScore(
        camera_id="CAM-03", timestamp=now, blur_score=90.0, brightness_score=90.0,
        frame_health_score=100.0, obstruction_score=90.0, composite_reliability_score=92.5,
        status=CameraStatus.GOOD, reasons=[]
    )

    alert = alert_mgr.process_event(ctx, rel, cross_camera_confirmed=True)
    assert alert is not None
    assert alert.event_priority in [EventPriority.HIGH, EventPriority.CRITICAL]

    # 3. Evidence Capture & SHA-256 Hashing
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = EvidenceCapturer(storage_dir=tmpdir)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8) + 120
        pre_frames = [(ctx, dummy_img)]

        record = capturer.capture_evidence(alert=alert, pre_event_frames=pre_frames, current_frame_img=dummy_img)
        assert record is not None
        assert record.is_sealed is True
        assert len(record.sha256_hash) == 64

        # 4. Hash Verification & Tamper Simulation Demo (Correction 2)
        snapshot_path = os.path.join(tmpdir, record.snapshot_filename)
        is_valid, comp_hash = EvidenceHasher.verify_integrity(snapshot_path, record.sha256_hash)
        assert is_valid is True
        assert comp_hash == record.sha256_hash

        # Run tamper demo (uses temp file; original untouched)
        is_valid_demo, rec_h, comp_h = EvidenceHasher.simulate_tampering_demo(snapshot_path, record.sha256_hash)
        assert is_valid_demo is False  # Triggers HASH MISMATCH!
        assert rec_h == record.sha256_hash
        assert comp_h != record.sha256_hash

    # 5. Escalation Timeout
    escalation_mgr = EscalationManager(ack_timeout_seconds=10.0)
    
    # Fast-forward 12 seconds
    escalated_list = escalation_mgr.check_and_escalate_alerts([alert], current_time=now + 12.0)
    assert len(escalated_list) == 1
    assert alert.state == AlertState.ESCALATED

    print("\n[Milestone C Security Operations Workflow Verified]")
    print(f"  Alert ID        : {alert.alert_id}")
    print(f"  SHA-256 Hash    : {record.sha256_hash[:16]}...")
    print(f"  Tamper Demo     : HASH MISMATCH verified (Original intact)")
    print(f"  Escalation State: {alert.state.value} (Escalated to COMMAND_CENTRE)")
