import pytest
from backend.coverage.fov import LocalFOVCalculator
from backend.coverage.coverage_engine import GeometricCoverageEngine

def test_local_fov_calculator_metric():
    origin = (0.0, 0.0)  # Metric origin
    heading = 0.0        # North (+Y)
    fov = 60.0
    range_m = 100.0

    poly = LocalFOVCalculator.calculate_fov_wedge_metric(origin, heading, fov, range_m)
    assert poly is not None
    assert poly.is_valid is True
    assert poly.area > 0.0

def test_gps_metric_transforms():
    ref_lat, ref_lon = 31.6215, 74.8752
    
    # Transform GPS point to metric and back
    x, y = GeometricCoverageEngine.gps_to_metric(31.6230, 74.8780, ref_lat, ref_lon)
    assert isinstance(x, float)
    assert isinstance(y, float)

    lat_back, lon_back = GeometricCoverageEngine.metric_to_gps(x, y, ref_lat, ref_lon)
    assert abs(lat_back - 31.6230) < 1e-4
    assert abs(lon_back - 74.8780) < 1e-4

def test_geometric_coverage_engine():
    engine = GeometricCoverageEngine(config_path="data/config/cameras.json")
    result = engine.compute_coverage()

    assert "single_fov_polygons" in result
    assert "coverage_tier_labels" in result
    assert result["coverage_tier_labels"]["single"] == "1 modeled camera FOV"
    assert result["coverage_tier_labels"]["none"] == "0 modeled camera FOVs"
