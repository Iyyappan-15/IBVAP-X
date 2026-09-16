"""
backend/scoring/narrative_engine.py

IBVAP-X — 100% Offline, Deterministic Start-to-End Video Narrative Engine (Approach B).
Synthesizes ByteTrack trajectories, spatial zones, velocity vectors, dwell times,
and context events into a structured, chronological story (Start -> Middle -> End).
Zero external API calls, zero cloud latency, zero token costs.
"""
from __future__ import annotations

import math
from typing import Dict, List, Any, Optional


class VideoNarrativeEngine:
    """
    Synthesizes session telemetry, tracking trajectories, and context events into
    a comprehensive, natural-language chronological narrative describing surveillance
    video footage from start to end.
    """

    @classmethod
    def generate_video_narrative(cls, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Generates structured start-to-end narrative from session summary.
        """
        camera_id = session.get("camera_id", "CAM-01")
        video_name = session.get("video_filename", session.get("source_label", "Surveillance Stream"))
        total_frames = max(1, session.get("processed_frames", 1))
        fps = max(1.0, float(session.get("source_fps", 25.0)))
        total_duration = total_frames / fps

        # 1. Environmental & Sensor Context Analysis
        avg_lum = float(session.get("avg_luminance", 50.0))
        is_night = avg_lum < 65.0
        lighting_desc = "Low-light / Night vision infrared" if is_night else ("High-contrast daylight" if avg_lum > 140.0 else "Normal daylight")
        
        avg_sharp = float(session.get("avg_sharpness", 100.0))
        blur_score = float(session.get("blur_score", 95.0))
        rel_pct = float(session.get("reliability_pct", 95.0))
        rel_status = "GOOD" if rel_pct >= 80.0 else ("DEGRADED" if rel_pct >= 50.0 else "POOR")
        sensor_desc = f"Sensor health is {rel_status} ({rel_pct:.0f}% reliability, optical sharpness {blur_score:.0f}%)"

        # 2. Entity & Trajectory Analysis
        seen_entities: Dict[str, Dict[str, Any]] = session.get("seen_entities", {})
        non_fence_entities = [
            e for e in seen_entities.values()
            if str(e.get("track_id")) != "PERIMETER" and str(e.get("class_name", "")).lower() != "fence"
        ]

        # Categorize actors
        humans = [e for e in non_fence_entities if "person" in e.get("class_name", "").lower()]
        animals = [e for e in non_fence_entities if e.get("class_name", "").lower() in ("dog", "cat", "bird", "horse", "cow", "sheep", "animal")]
        vehicles = [e for e in non_fence_entities if e.get("class_name", "").lower() in ("car", "truck", "bus", "motorcycle", "bicycle")]

        alerts_list: List[Dict[str, Any]] = session.get("alerts_list", [])
        highest_alert = max(alerts_list, key=lambda a: a.get("priority_score", 0.0)) if alerts_list else None
        prio_label = highest_alert["priority"] if highest_alert else "LOW"
        act_label = highest_alert["actionability"] if highest_alert else "LOW"
        action_rec = highest_alert.get("action_recommendation", "Maintain standard automated perimeter surveillance.") if highest_alert else "Maintain standard perimeter surveillance."

        # 3. Construct Executive Summary
        actor_summary_parts = []
        if humans:
            actor_summary_parts.append(f"{len(humans)} human target{'s' if len(humans) > 1 else ''}")
        if animals:
            actor_summary_parts.append(f"{len(animals)} quadruped animal{'s' if len(animals) > 1 else ''} ({', '.join(set(a.get('class_name', 'animal') for a in animals))})")
        if vehicles:
            actor_summary_parts.append(f"{len(vehicles)} vehicle{'s' if len(vehicles) > 1 else ''}")
        if not actor_summary_parts:
            actor_summary_str = "Zero unauthorized targets or anomalous entities"
        else:
            actor_summary_str = ", ".join(actor_summary_parts)

        exec_summary = (
            f"Surveillance recording '{video_name}' captured by sector '{camera_id}' across a duration of {total_duration:.1f} seconds "
            f"({total_frames} frames at {fps:.1f} FPS) under {lighting_desc.lower()} conditions. "
            f"{sensor_desc}. Analysis identified {actor_summary_str} within the monitored perimeter buffer."
        )

        # 4. Construct Chronological Story Phases (Start -> Middle -> End)
        chronological_story = cls._build_chronological_phases(
            non_fence_entities, total_duration, fps, highest_alert, is_night, lighting_desc
        )

        # 5. Actor-Specific Behavioral & Trajectory Profiles
        actor_breakdowns = cls._build_actor_breakdowns(non_fence_entities, fps, total_duration)

        # 6. Environmental Assessment
        env_verdict = {
            "lighting": lighting_desc,
            "is_night": is_night,
            "camera_reliability_pct": rel_pct,
            "camera_status": rel_status,
            "optical_quality": f"{blur_score:.0f}%",
            "glitch_resistance": "Temporal persistence filter active (noise flickers suppressed)",
        }

        # 7. Security Verdict & Directive
        if highest_alert and prio_label in ("CRITICAL", "HIGH"):
            threat_desc = (
                f"POTENTIAL BREACH / PERIMETER RISK: A confirmed {highest_alert.get('class_name', 'target').upper()} "
                f"(Track #{highest_alert.get('track_id', 1)}) was observed advancing into restricted security zones. "
                f"Tactical Actionability is {act_label}. Recommended action: {action_rec}"
            )
        elif animals and not humans:
            threat_desc = (
                f"BENIGN ANIMAL ACTIVITY: Activity in Sector {camera_id} is attributable exclusively to natural animal movement "
                f"({', '.join(set(a.get('class_name', 'animal') for a in animals))}). Zero human breach indicators detected. "
                f"Alert priority classified as LOW. Tactical directive: Log and suppress false alarm."
            )
        elif not non_fence_entities:
            threat_desc = (
                f"PERIMETER SECURE: Monitored sector {camera_id} showed zero unauthorized movement or perimeter barrier tampering. "
                f"All security zones remain clear and nominal."
            )
        else:
            threat_desc = (
                f"ROUTINE MONITORING: Detected activity consists of nominal transit with no hostile trajectory vectors or barrier breaches. "
                f"Threat priority: {prio_label}. Tactical directive: {action_rec}"
            )

        # 8. Full Text Compilation
        full_text_lines = [
            f"================================================================================",
            f"IBVAP-X SURVEILLANCE NARRATIVE & INCIDENT CHRONOLOGY",
            f"================================================================================",
            f"Camera Sector : {camera_id}",
            f"Source Video  : {video_name}",
            f"Duration      : {total_duration:.1f}s ({total_frames} frames @ {fps:.1f} FPS)",
            f"Environment   : {lighting_desc} | Health: {rel_pct:.0f}% ({rel_status})",
            f"Threat Level  : {prio_label} (Actionability: {act_label})",
            f"================================================================================",
            f"",
            f"[1. EXECUTIVE SCENE SUMMARY]",
            f"{exec_summary}",
            f"",
            f"[2. CHRONOLOGICAL EVENT PROGRESSION]",
        ]

        for phase in chronological_story:
            full_text_lines.append(f"• {phase['phase']} (Timestamp: {phase['timestamp_range']}):")
            full_text_lines.append(f"  {phase['narrative']}")
            full_text_lines.append(f"")

        full_text_lines.append(f"[3. ACTOR TRAJECTORY & BEHAVIORAL PROFILES]")
        if actor_breakdowns:
            for actor in actor_breakdowns:
                full_text_lines.append(f"• Target Track #{actor['track_id']} [{actor['class_name'].upper()}]:")
                full_text_lines.append(f"  - Active Presence : {actor['active_window']} ({actor['scene_coverage']})")
                full_text_lines.append(f"  - Spatial Heading : {actor['spatial_heading']}")
                full_text_lines.append(f"  - Behavior Profile: {actor['behavior_description']}")
                full_text_lines.append(f"")
        else:
            full_text_lines.append(f"  No persistent dynamic actors detected in evaluated video sequence.\n")

        full_text_lines.append(f"[4. TACTICAL SECURITY VERDICT]")
        full_text_lines.append(f"{threat_desc}")
        full_text_lines.append(f"================================================================================")

        full_narrative_text = "\n".join(full_text_lines)

        return {
            "executive_summary": exec_summary,
            "chronological_story": chronological_story,
            "actor_breakdowns": actor_breakdowns,
            "environmental_verdict": env_verdict,
            "security_verdict": threat_desc,
            "highest_priority": prio_label,
            "actionability": act_label,
            "action_recommendation": action_rec,
            "full_narrative_text": full_narrative_text
        }

    @classmethod
    def _build_chronological_phases(
        cls,
        entities: List[Dict[str, Any]],
        total_duration: float,
        fps: float,
        highest_alert: Optional[Dict[str, Any]],
        is_night: bool,
        lighting_desc: str
    ) -> List[Dict[str, Any]]:
        """
        Divides the video duration into 3 structured temporal phases (Start, Middle, End)
        and narrates the evolving situation across each window.
        """
        t_phase1_end = max(1.0, total_duration * 0.30)
        t_phase2_end = max(t_phase1_end + 1.0, total_duration * 0.75)

        if not entities:
            return [
                {
                    "phase": "Phase 1: Ingress / Initial Scene",
                    "timestamp_range": f"0.0s – {t_phase1_end:.1f}s",
                    "narrative": "Camera begins surveillance over the designated sector. Perimeter fencing and physical buffer zones are completely clear with zero detected motion or targets."
                },
                {
                    "phase": "Phase 2: Mid-Sequence Observation",
                    "timestamp_range": f"{t_phase1_end:.1f}s – {t_phase2_end:.1f}s",
                    "narrative": "Sector remains static and undisturbed. No intrusions, approaching vehicles, or anomalous entities enter the field of view."
                },
                {
                    "phase": "Phase 3: Egress / Resolution",
                    "timestamp_range": f"{t_phase2_end:.1f}s – {total_duration:.1f}s",
                    "narrative": "Footage concludes with perimeter barrier intact and zero security incidents logged."
                }
            ]

        # Classify actors present in Phase 1 (start)
        phase1_actors = [
            e for e in entities
            if (e.get("first_frame", 1) / fps) <= t_phase1_end
        ]
        phase1_desc = []
        for e in phase1_actors:
            cname = e.get("class_name", "object")
            tid = e.get("track_id", 1)
            t_start = (e.get("first_frame", 1) - 1) / fps
            traj = e.get("trajectory", [])
            pos_desc = cls._describe_position(traj[0] if traj else None)
            if t_start <= 0.5:
                phase1_desc.append(f"A **{cname.upper()}** (Track #{tid}) is already present in the {pos_desc}")
            else:
                phase1_desc.append(f"A **{cname.upper()}** (Track #{tid}) appears at {t_start:.1f}s entering from the {pos_desc}")

        if not phase1_desc:
            phase1_narrative = f"Perimeter begins in nominal condition under {lighting_desc.lower()}. No active targets in early sequence."
        else:
            phase1_narrative = f"At the opening of the sequence: {'; '.join(phase1_desc)}. Baseline telemetry confirms normal optical visibility."

        # Phase 2: Middle actions (trajectories, zones, dwell)
        phase2_actions = []
        for e in entities:
            cname = e.get("class_name", "object")
            tid = e.get("track_id", 1)
            traj = e.get("trajectory", [])
            heading = cls._describe_heading(traj)
            is_in_zone = bool(e.get("in_restricted_zone", False)) or ("restricted" in e.get("zone", "").lower())
            is_loiter = bool(e.get("loitering", False))

            action_bits = []
            if heading:
                action_bits.append(f"moves along a {heading.lower()}")
            if is_in_zone:
                action_bits.append("crosses into the restricted perimeter buffer")
            if is_loiter:
                action_bits.append("exhibits sustained dwell / loitering behavior")

            if action_bits:
                phase2_actions.append(f"**{cname.upper()}** (Track #{tid}) {', '.join(action_bits)}")

        if not phase2_actions:
            phase2_narrative = "Targets in frame maintain stationary positions without advancing toward security perimeter zones."
        else:
            phase2_narrative = f"During mid-sequence progression: {'; '.join(phase2_actions)}."

        # Phase 3: Culmination (final position, exit or sustained presence)
        phase3_actions = []
        for e in entities:
            cname = e.get("class_name", "object")
            tid = e.get("track_id", 1)
            last_t = e.get("last_frame", int(total_duration * fps)) / fps
            traj = e.get("trajectory", [])
            end_pos = cls._describe_position(traj[-1] if traj else None)

            if last_t < (total_duration - 1.0):
                phase3_actions.append(f"**{cname.upper()}** (Track #{tid}) exits the camera's field of view at {last_t:.1f}s")
            else:
                phase3_actions.append(f"**{cname.upper()}** (Track #{tid}) remains visible near the {end_pos} at sequence completion")

        phase3_narrative = f"As the clip concludes: {'; '.join(phase3_actions)}."

        return [
            {
                "phase": "Phase 1: Ingress / Initial Scene",
                "timestamp_range": f"0.0s – {t_phase1_end:.1f}s",
                "narrative": phase1_narrative
            },
            {
                "phase": "Phase 2: Mid-Sequence Trajectory & Interactions",
                "timestamp_range": f"{t_phase1_end:.1f}s – {t_phase2_end:.1f}s",
                "narrative": phase2_narrative
            },
            {
                "phase": "Phase 3: Egress / Culmination",
                "timestamp_range": f"{t_phase2_end:.1f}s – {total_duration:.1f}s",
                "narrative": phase3_narrative
            }
        ]

    @classmethod
    def _build_actor_breakdowns(
        cls,
        entities: List[Dict[str, Any]],
        fps: float,
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Creates granular behavioral and movement profiles for each tracked actor.
        """
        profiles = []
        for e in entities:
            tid = e.get("track_id", 1)
            cname = e.get("class_name", "object")
            f_start = e.get("first_frame", 1)
            f_end = e.get("last_frame", 1)
            t_start = (f_start - 1) / fps
            t_end = f_end / fps
            duration_actor = max(0.1, t_end - t_start)
            coverage_pct = min(100.0, (duration_actor / max(total_duration, 0.1)) * 100.0)

            traj = e.get("trajectory", [])
            heading_desc = cls._describe_heading(traj)
            pos_start = cls._describe_position(traj[0] if traj else None)
            pos_end = cls._describe_position(traj[-1] if traj else None)

            # Behavior characterization
            behaviors = []
            if "person" in cname.lower():
                if bool(e.get("loitering", False)) or duration_actor >= 10.0:
                    behaviors.append("Sustained perimeter presence / loitering")
                else:
                    behaviors.append("Active human patrol / transit")
            elif cname.lower() in ("dog", "cat", "animal"):
                behaviors.append("Low-altitude quadruped movement (non-hostile wildlife/domestic animal)")
            elif cname.lower() in ("car", "truck"):
                behaviors.append("Motorized vehicular transit")
            else:
                behaviors.append("Monitored physical object")

            if bool(e.get("in_restricted_zone", False)):
                behaviors.append("Entered restricted security buffer")

            profiles.append({
                "track_id": tid,
                "class_name": cname,
                "active_window": f"{t_start:.1f}s -> {t_end:.1f}s",
                "scene_coverage": f"{coverage_pct:.0f}% of clip",
                "spatial_heading": f"From {pos_start} to {pos_end} ({heading_desc})",
                "behavior_description": "; ".join(behaviors)
            })

        return profiles

    @staticmethod
    def _describe_position(point: Optional[tuple]) -> str:
        """Translates pixel coordinates (x, y) on normalized/typical 640x480 frame into quadrant descriptors."""
        if not point or len(point) < 2:
            return "perimeter sector"
        x, y = point[0], point[1]
        
        # Quadrant estimation (assuming typical frame width ~640, height ~480)
        horiz = "left / western flank" if x < 210 else ("right / eastern flank" if x > 430 else "central corridor")
        vert = "upper background" if y < 160 else ("lower foreground" if y > 320 else "mid-ground sector")
        return f"{vert} ({horiz})"

    @staticmethod
    def _describe_heading(trajectory: List[tuple]) -> str:
        """Determines movement vector from trajectory points."""
        if not trajectory or len(trajectory) < 2:
            return "Stationary / hovering in place"
        
        p_start = trajectory[0]
        p_end = trajectory[-1]
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dist = math.hypot(dx, dy)

        if dist < 20.0:
            return "Stationary position (minimal displacement)"
        
        # Determine dominant direction
        if abs(dy) > 1.4 * abs(dx):
            return "Forward frontal advance toward camera / border line" if dy > 0 else "Retreating toward background"
        elif abs(dx) > 1.4 * abs(dy):
            return "Lateral traversal (west to east)" if dx > 0 else "Lateral traversal (east to west)"
        else:
            diag_h = "eastward" if dx > 0 else "westward"
            diag_v = "advancing southward" if dy > 0 else "retreating northward"
            return f"Diagonal trajectory ({diag_v}, {diag_h})"
