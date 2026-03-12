from datetime import datetime, timedelta
import pytest

from src.geo_analyzer import GeoAnalyzer, haversine_km
from src.models import Location
from tests.conftest import make_transaction, make_location


class TestHaversine:
    def test_same_location_is_zero(self):
        loc = make_location(51.5, -0.1)
        assert haversine_km(loc, loc) == pytest.approx(0.0, abs=0.01)

    def test_london_to_new_york(self):
        london = make_location(51.5074, -0.1278)
        new_york = make_location(40.7128, -74.0060)
        distance = haversine_km(london, new_york)
        assert 5500 < distance < 5600  # ~5570 km

    def test_london_to_paris(self):
        london = make_location(51.5074, -0.1278)
        paris = make_location(48.8566, 2.3522)
        distance = haversine_km(london, paris)
        assert 340 < distance < 360  # ~343 km

    def test_symmetry(self):
        a = make_location(51.5, -0.1)
        b = make_location(40.7, -74.0)
        assert haversine_km(a, b) == pytest.approx(haversine_km(b, a), rel=0.001)


class TestGeoAnalyzer:
    def setup_method(self):
        self.analyzer = GeoAnalyzer()

    def test_first_transaction_no_score(self):
        t = make_transaction(location=make_location())
        result = self.analyzer.check(t)
        assert result.score == 0.0
        assert result.triggered_rules == []

    def test_no_location_skipped(self):
        t = make_transaction(location=None)
        result = self.analyzer.check(t)
        assert result.score == 0.0
        assert "skipped" in result.metadata

    def test_impossible_travel_detected(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        london = make_location(51.5074, -0.1278, "GB")
        new_york = make_location(40.7128, -74.0060, "US")

        t1 = make_transaction(user_id="traveller", location=london, timestamp=base)
        t2 = make_transaction(
            user_id="traveller",
            location=new_york,
            timestamp=base + timedelta(minutes=10)  # 10 min for ~5570km = impossible
        )
        self.analyzer.check(t1)
        result = self.analyzer.check(t2)

        assert "impossible_travel" in result.triggered_rules
        assert result.score > 40

    def test_realistic_travel_not_flagged(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        london = make_location(51.5074, -0.1278, "GB")
        new_york = make_location(40.7128, -74.0060, "US")

        t1 = make_transaction(user_id="flyer", location=london, timestamp=base)
        # 10 hours later is fine for a flight
        t2 = make_transaction(
            user_id="flyer",
            location=new_york,
            timestamp=base + timedelta(hours=10)
        )
        self.analyzer.check(t1)
        result = self.analyzer.check(t2)
        assert "impossible_travel" not in result.triggered_rules

    def test_high_risk_country_flagged(self):
        t = make_transaction(location=make_location(55.7, 37.6, "RU"))
        result = self.analyzer.check(t)
        assert "high_risk_country" in result.triggered_rules
        assert result.score >= 35

    def test_low_risk_country_not_flagged(self):
        t = make_transaction(location=make_location(51.5, -0.1, "GB"))
        result = self.analyzer.check(t)
        assert "high_risk_country" not in result.triggered_rules

    def test_rapid_country_change_flagged(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        uk_loc = make_location(51.5, -0.1, "GB")
        fr_loc = make_location(48.8, 2.3, "FR")

        t1 = make_transaction(user_id="country_jumper", location=uk_loc, timestamp=base)
        t2 = make_transaction(
            user_id="country_jumper",
            location=fr_loc,
            timestamp=base + timedelta(minutes=10)
        )
        self.analyzer.check(t1)
        result = self.analyzer.check(t2)
        assert "rapid_country_change" in result.triggered_rules

    def test_different_users_isolated(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        london = make_location(51.5, -0.1, "GB")
        ny = make_location(40.7, -74.0, "US")

        t1 = make_transaction(user_id="user_geo_1", location=london, timestamp=base)
        t2 = make_transaction(
            user_id="user_geo_2",  # different user
            location=ny,
            timestamp=base + timedelta(minutes=5)
        )
        self.analyzer.check(t1)
        result = self.analyzer.check(t2)
        assert "impossible_travel" not in result.triggered_rules

    def test_reset_user_clears_location(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        london = make_location(51.5, -0.1, "GB")
        ny = make_location(40.7, -74.0, "US")

        t1 = make_transaction(user_id="geo_reset", location=london, timestamp=base)
        self.analyzer.check(t1)
        self.analyzer.reset_user("geo_reset")

        t2 = make_transaction(
            user_id="geo_reset",
            location=ny,
            timestamp=base + timedelta(minutes=5)
        )
        result = self.analyzer.check(t2)
        assert "impossible_travel" not in result.triggered_rules

    def test_detector_name(self):
        result = self.analyzer.check(make_transaction())
        assert result.detector_name == "geo_analyzer"
