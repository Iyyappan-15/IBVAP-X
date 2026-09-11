from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time
import numpy as np

@dataclass
class VideoFrameData:
    camera_id: str
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    frame: Optional[np.ndarray] = None
    source_type: str = "file"  # file | webcam | rtsp | demo
    source_metadata: Dict[str, Any] = field(default_factory=dict)
