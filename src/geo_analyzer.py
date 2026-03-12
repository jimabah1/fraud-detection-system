import math
import threading
from datetime import datetime
from typing import Optional

from .models import Transaction, Location, DetectorResult


def haversine_km(loc1: Location, loc2: Location) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in kilometres.
    """
    R = 6371.0
    lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
    lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# Maximum realistic travel speed in km/h (commercial flight + airport time)
MAX_TRAVEL_SPEED_KMH = 900.0
# Minimum hours before geo anomaly applies (allow same-city variance)
MIN_TIME_GAP_SECONDS = 300


class GeoAnalyzer:
    """
    Detects impossible travel: two transactions in locations that can't be
    reached within the elapsed time, even by plane.

    Also flags high-risk country transitions and international transactions
    with suspicious timing.
    """

    HIGH_RISK_COUNTRIES = {
        "NG", "RU", "CN", "KP", "IR", "VE", "UA", "BY",
    }

    def __init__(self):
        # user_id -> (timestamp, location)
        self._last_location: dict[str, tuple[datetime, Location]] = {}
        self._lock = threading.Lock()

    def check(self, transaction: Transaction) -> DetectorResult:
        triggered = []
        score = 0.0
        metadata = {}

        location = transaction.location

        if location is None:
            return DetectorResult(
                detector_name="geo_analyzer",
                score=0.0,
                weight=1.2,
                triggered_rules=[],
                metadata={"skipped": "no_location_data"},
            )

        with self._lock:
            last = self._last_location.get(transaction.user_id)

            if last is not None:
                last_time, last_loc = last
                time_gap_seconds = (transaction.timestamp - last_time).total_seconds()

                if time_gap_seconds > 0:
                    distance_km = haversine_km(last_loc, location)
                    required_speed = (distance_km / (time_gap_seconds / 3600)) if time_gap_seconds >= MIN_TIME_GAP_SECONDS else 0

                    metadata["distance_km"] = round(distance_km, 1)
                    metadata["required_speed_kmh"] = round(required_speed, 1)
                    metadata["time_gap_seconds"] = time_gap_seconds

                    if time_gap_seconds >= MIN_TIME_GAP_SECONDS and required_speed > MAX_TRAVEL_SPEED_KMH:
                        triggered.append("impossible_travel")
                        # Score scales with how impossible the travel is
                        impossibility = min(required_speed / MAX_TRAVEL_SPEED_KMH, 10.0)
                        score = max(score, min(40 + (impossibility - 1) * 10, 95))

                    if last_loc.country and location.country and last_loc.country != location.country:
                        if time_gap_seconds < 3600:
                            triggered.append("rapid_country_change")
                            score = max(score, 50.0)

            self._last_location[transaction.user_id] = (transaction.timestamp, location)

        if location.country in self.HIGH_RISK_COUNTRIES:
            triggered.append("high_risk_country")
            score = max(score, 35.0)

        if transaction.is_international:
            score = max(score, score + 10.0) if triggered else score

        return DetectorResult(
            detector_name="geo_analyzer",
            score=min(score, 100.0),
            weight=1.2,
            triggered_rules=triggered,
            metadata=metadata,
        )

    def reset_user(self, user_id: str) -> None:
        with self._lock:
            self._last_location.pop(user_id, None)

    def get_last_location(self, user_id: str) -> Optional[tuple[datetime, Location]]:
        with self._lock:
            return self._last_location.get(user_id)
