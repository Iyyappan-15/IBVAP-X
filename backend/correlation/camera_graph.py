import json
import os
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

class CameraGraph:
    """Manages spatial adjacency graph between border surveillance cameras."""

    def __init__(self, config_path: str = "data/config/cameras.json"):
        self.config_path = config_path
        # Structure: camera_id -> Set[adjacent_camera_ids]
        self.adjacency: Dict[str, Set[str]] = {}
        self._load_graph()

    def _load_graph(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"[CameraGraph] Config file '{self.config_path}' not found. Adjacency graph empty.")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cam in data.get("cameras", []):
            cid = cam["camera_id"]
            adj_list = cam.get("adjacent_cameras", [])
            self.adjacency[cid] = set(adj_list)

    def is_adjacent(self, cam1: str, cam2: str) -> bool:
        """Returns True if cam1 and cam2 are configured as adjacent cameras."""
        if cam1 == cam2:
            return True
        return cam2 in self.adjacency.get(cam1, set()) or cam1 in self.adjacency.get(cam2, set())

    def get_adjacent_cameras(self, camera_id: str) -> List[str]:
        return list(self.adjacency.get(camera_id, set()))
