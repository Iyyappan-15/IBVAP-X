from datetime import datetime
from backend.config import settings
from backend.interfaces import TimeContextEnum

class TimeContextClassifier:
    """Classifies operational hours into DAY or NIGHT context."""

    def __init__(self, day_start_hour: int = None, night_start_hour: int = None):
        self.day_start_hour = day_start_hour or settings.DAY_START_HOUR
        self.night_start_hour = night_start_hour or settings.NIGHT_START_HOUR

    def classify(self, timestamp: float) -> TimeContextEnum:
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour

        if self.day_start_hour <= hour < self.night_start_hour:
            return TimeContextEnum.DAY
        else:
            return TimeContextEnum.NIGHT
