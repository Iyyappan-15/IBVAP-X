import time
import uuid
import logging
from typing import List, Dict, Optional

from backend.config import settings
from backend.interfaces import ContextEvent, CorrelatedEvent
from backend.correlation.camera_graph import CameraGraph

logger = logging.getLogger(__name__)

class EventCorrelator:
    """Correlates events across adjacent cameras within a configured time window (without biometric re-ID)."""

    def __init__(self, config_path: str = "data/config/cameras.json"):
        self.camera_graph = CameraGraph(config_path=config_path)
        self.time_window = settings.CORRELATION_WINDOW_SECONDS
        
        # Recent events cache: List[ContextEvent]
        self.recent_events: List[ContextEvent] = []

    def process_event(self, event: ContextEvent) -> Optional[CorrelatedEvent]:
        """
        Checks if event matches recent events on adjacent cameras within the time window.
        Returns CorrelatedEvent if match found.
        """
        now = event.timestamp
        
        # Prune old events outside time window
        self.recent_events = [e for e in self.recent_events if (now - e.timestamp) <= self.time_window]

        matched_cameras = set([event.camera_id])

        for prev_evt in self.recent_events:
            if prev_evt.camera_id != event.camera_id:
                # Check same object class and spatial adjacency
                if prev_evt.class_name.lower() == event.class_name.lower():
                    if self.camera_graph.is_adjacent(event.camera_id, prev_evt.camera_id):
                        matched_cameras.add(prev_evt.camera_id)

        self.recent_events.append(event)

        if len(matched_cameras) > 1:
            corr_id = f"CORR-{uuid.uuid4().hex[:8].upper()}"
            cam_list = list(matched_cameras)
            
            explanation = (
                f"Class-based correlation across {len(cam_list)} cameras ({', '.join(cam_list)}) "
                f"within {self.time_window}s window (not identity-confirmed)."
            )

            logger.info(f"[EventCorrelator] Correlated event created across cameras: {cam_list}")
            return CorrelatedEvent(
                correlation_id=corr_id,
                camera_ids=cam_list,
                event_type="CROSS_CAMERA_MOVEMENT",
                start_time=now - 20.0,
                end_time=now,
                object_class=event.class_name,
                explanation=explanation,
                confirmed=True
            )

        return None
