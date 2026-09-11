from typing import Dict, Tuple, Optional
from backend.config import settings

class LoiteringDetector:
    """Tracks time spent inside zones per track ID and prevents duplicate alert spam using cooldown logic."""

    def __init__(
        self,
        threshold_seconds: float = None,
        cooldown_seconds: float = None
    ):
        self.threshold_seconds = threshold_seconds or settings.LOITERING_THRESHOLD_SECONDS
        self.cooldown_seconds = cooldown_seconds or settings.LOITERING_COOLDOWN_SECONDS

        # Structure: track_id -> {zone_name: {"entry_time": float, "last_alert_time": float}}
        self.track_zone_timers: Dict[int, Dict[str, Dict[str, float]]] = {}

    def update_track_zone(
        self,
        track_id: int,
        zone_name: Optional[str],
        current_time: float
    ) -> Tuple[bool, float]:
        """
        Updates time spent in zone for track_id.
        Returns Tuple[is_loitering: bool, duration_seconds: float].
        """
        if track_id not in self.track_zone_timers:
            self.track_zone_timers[track_id] = {}

        track_timers = self.track_zone_timers[track_id]

        if not zone_name:
            # Track left all zones — reset timers
            self.track_zone_timers[track_id] = {}
            return False, 0.0

        if zone_name not in track_timers:
            # Track entered new zone
            track_timers[zone_name] = {
                "entry_time": current_time,
                "last_alert_time": 0.0
            }

        z_data = track_timers[zone_name]
        duration = current_time - z_data["entry_time"]

        is_loitering = False
        if duration >= self.threshold_seconds:
            # Check cooldown to prevent duplicate alert spam
            if (current_time - z_data["last_alert_time"]) >= self.cooldown_seconds:
                is_loitering = True
                z_data["last_alert_time"] = current_time

        return is_loitering, duration
