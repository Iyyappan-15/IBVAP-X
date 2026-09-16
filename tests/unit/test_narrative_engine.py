"""
tests/unit/test_narrative_engine.py

Unit tests for VideoNarrativeEngine:
- Validates executive summary generation
- Validates 3-phase chronological story (Start -> Middle -> End)
- Validates actor-specific trajectory and behavioral profiles
- Validates pure offline execution with zero API keys or external services
"""
import pytest
from backend.scoring.narrative_engine import VideoNarrativeEngine


def test_narrative_engine_empty_scene():
    """Verifies narrative generation on a clean, empty perimeter scene."""
    session = {
        "camera_id": "CAM-01",
        "video_filename": "perimeter_patrol_empty.mp4",
        "processed_frames": 100,
        "source_fps": 25.0,
        "avg_luminance": 110.0,
        "avg_sharpness": 150.0,
        "blur_score": 95.0,
        "reliability_pct": 98.0,
        "seen_entities": {},
        "alerts_list": [],
        "timeline": []
    }

    result = VideoNarrativeEngine.generate_video_narrative(session)

    assert "executive_summary" in result
    assert "chronological_story" in result
    assert len(result["chronological_story"]) == 3
    assert result["highest_priority"] == "LOW"
    assert "PERIMETER SECURE" in result["security_verdict"] or "clear" in result["security_verdict"].lower()
    assert "Phase 1: Ingress" in result["chronological_story"][0]["phase"]
    assert "Phase 2: Mid-Sequence" in result["chronological_story"][1]["phase"]
    assert "Phase 3: Egress" in result["chronological_story"][2]["phase"]


def test_narrative_engine_human_breach_chronology():
    """Verifies narrative generation when a human enters the restricted buffer zone."""
    session = {
        "camera_id": "CAM-NORTH-04",
        "video_filename": "night_incursion_test.mp4",
        "processed_frames": 150,
        "source_fps": 30.0,
        "avg_luminance": 40.0,  # Night infrared
        "avg_sharpness": 85.0,
        "blur_score": 90.0,
        "reliability_pct": 94.0,
        "seen_entities": {
            "person_1": {
                "track_id": 1,
                "class_name": "person",
                "first_frame": 1,
                "last_frame": 150,
                "trajectory": [(100.0, 80.0), (250.0, 200.0), (320.0, 380.0)],
                "direction": "TOWARD_BOUNDARY",
                "zone": "Restricted Zone Alpha",
                "in_restricted_zone": True,
                "loitering": False,
                "confidence": 0.91,
                "label_stability": "HIGH"
            }
        },
        "alerts_list": [
            {
                "alert_id": "ALT-01",
                "priority": "HIGH",
                "priority_score": 82.0,
                "class_name": "person",
                "track_id": 1,
                "actionability": "HIGH",
                "action_recommendation": "Deploy rapid response unit to intercept.",
                "why_reasons": ["Restricted Zone Entry", "Forward approach toward border barrier"]
            }
        ],
        "timeline": []
    }

    result = VideoNarrativeEngine.generate_video_narrative(session)

    assert result["highest_priority"] == "HIGH"
    assert result["actionability"] == "HIGH"
    assert "1 human target" in result["executive_summary"]
    assert "night vision" in result["executive_summary"].lower() or "low-light" in result["executive_summary"].lower()

    # Verify chronological phases
    story = result["chronological_story"]
    assert len(story) == 3
    assert "PERSON" in story[0]["narrative"]
    assert "restricted perimeter buffer" in story[1]["narrative"]

    # Verify actor breakdown
    actors = result["actor_breakdowns"]
    assert len(actors) == 1
    assert actors[0]["track_id"] == 1
    assert actors[0]["class_name"] == "person"
    assert "100% of clip" in actors[0]["scene_coverage"]

    # Verify formatted report
    assert "IBVAP-X SURVEILLANCE NARRATIVE" in result["full_narrative_text"]
    assert "Deploy rapid response" in result["full_narrative_text"]


def test_narrative_engine_animal_suppression():
    """Verifies narrative generation correctly identifies benign animal movement without false breach alarms."""
    session = {
        "camera_id": "CAM-FENCE-02",
        "video_filename": "snow_dog_crossing.mp4",
        "processed_frames": 100,
        "source_fps": 25.0,
        "avg_luminance": 160.0,  # Snow daylight
        "avg_sharpness": 120.0,
        "blur_score": 96.0,
        "reliability_pct": 97.0,
        "seen_entities": {
            "dog_2": {
                "track_id": 2,
                "class_name": "dog",
                "first_frame": 10,
                "last_frame": 85,
                "trajectory": [(500.0, 200.0), (300.0, 210.0), (100.0, 220.0)],
                "direction": "LATERAL",
                "zone": "Outer Buffer",
                "in_restricted_zone": False,
                "loitering": False,
                "confidence": 0.88,
                "label_stability": "HIGH"
            }
        },
        "alerts_list": [],
        "timeline": []
    }

    result = VideoNarrativeEngine.generate_video_narrative(session)

    assert result["highest_priority"] == "LOW"
    assert "BENIGN ANIMAL ACTIVITY" in result["security_verdict"]
    assert "dog" in result["executive_summary"].lower()
    assert len(result["actor_breakdowns"]) == 1
    assert "quadruped" in result["actor_breakdowns"][0]["behavior_description"]
