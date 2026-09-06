"""Unit tests for tsp_utils module."""

import sys
import os
import warnings

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsp_utils import haversine_distance, calculate_route_distance, solve_tsp


# ---------------------------------------------------------------------------
# haversine_distance tests
# ---------------------------------------------------------------------------

class TestHaversineDistance:
    """Tests for the haversine_distance function."""

    def test_same_point_returns_zero(self):
        """Distance from a point to itself should be 0."""
        point = (17.385, 78.4867)  # Hyderabad
        assert haversine_distance(point, point) == pytest.approx(0.0, abs=1e-6)

    def test_known_city_pair_mumbai_delhi(self):
        """Mumbai (19.076, 72.878) → Delhi (28.644, 77.216) ≈ 1,153 km."""
        mumbai = (19.076, 72.878)
        delhi = (28.644, 77.216)
        dist = haversine_distance(mumbai, delhi)
        assert dist == pytest.approx(1153, abs=30)  # within 30 km tolerance

    def test_known_city_pair_hyderabad_bangalore(self):
        """Hyderabad (17.385, 78.487) → Bangalore (12.972, 77.595) ≈ 501 km."""
        hyderabad = (17.385, 78.487)
        bangalore = (12.972, 77.595)
        dist = haversine_distance(hyderabad, bangalore)
        assert dist == pytest.approx(501, abs=20)

    def test_symmetry(self):
        """distance(A, B) == distance(B, A)."""
        a = (17.385, 78.487)
        b = (12.972, 77.595)
        assert haversine_distance(a, b) == pytest.approx(haversine_distance(b, a))

    def test_small_distance(self):
        """Two nearby points (~1 km apart) should give a small distance."""
        p1 = (17.385, 78.487)
        p2 = (17.394, 78.487)  # ~1 km north
        dist = haversine_distance(p1, p2)
        assert 0.5 < dist < 2.0  # roughly 1 km


# ---------------------------------------------------------------------------
# calculate_route_distance tests
# ---------------------------------------------------------------------------

class TestCalculateRouteDistance:
    """Tests for the calculate_route_distance function."""

    def test_empty_route(self):
        assert calculate_route_distance([]) == 0.0

    def test_single_point(self):
        assert calculate_route_distance([(17.385, 78.487)]) == 0.0

    def test_two_points(self):
        """Should equal haversine between the two points."""
        p1 = (17.385, 78.487)
        p2 = (12.972, 77.595)
        assert calculate_route_distance([p1, p2]) == pytest.approx(
            haversine_distance(p1, p2)
        )

    def test_multi_point_route(self):
        """Total should be sum of consecutive pairwise distances."""
        points = [(17.385, 78.487), (17.4, 78.5), (17.42, 78.52)]
        expected = (
            haversine_distance(points[0], points[1])
            + haversine_distance(points[1], points[2])
        )
        assert calculate_route_distance(points) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# solve_tsp tests
# ---------------------------------------------------------------------------

class TestSolveTsp:
    """Tests for the TSP solver."""

    def test_two_points(self):
        """Two-point route should just return both points."""
        pts = [(17.385, 78.487), (17.4, 78.5)]
        route, dist = solve_tsp(pts)
        assert len(route) == 2
        assert dist == pytest.approx(calculate_route_distance(pts))

    def test_all_points_visited(self):
        """Every input point must appear exactly once in the output route."""
        pts = [
            (17.385, 78.487),
            (17.4, 78.5),
            (17.42, 78.52),
            (17.35, 78.46),
            (17.39, 78.49),
        ]
        route, _ = solve_tsp(pts)
        assert len(route) == len(pts)
        # All points present (order may differ)
        assert set(map(tuple, route)) == set(map(tuple, pts))

    def test_optimized_beats_naive(self):
        """Optimized distance should be ≤ the naive input-order distance."""
        pts = [
            (17.385, 78.487),
            (17.42, 78.52),   # far
            (17.39, 78.49),   # close to start
            (17.35, 78.46),   # another direction
            (17.4, 78.5),     # middle
        ]
        naive_dist = calculate_route_distance(pts)
        _, opt_dist = solve_tsp(pts)
        assert opt_dist <= naive_dist + 1e-9  # allow floating point tolerance

    def test_max_points_truncation_warns(self):
        """When len(points) > max_points, a RuntimeWarning should be issued."""
        pts = [(17.0 + i * 0.01, 78.0 + i * 0.01) for i in range(10)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            route, _ = solve_tsp(pts, max_points=5)
            assert len(route) == 5
            assert any(issubclass(warning.category, RuntimeWarning) for warning in w)

    def test_single_point(self):
        """Single point should return it with zero distance."""
        pts = [(17.385, 78.487)]
        route, dist = solve_tsp(pts)
        assert route == pts
        assert dist == 0.0

    def test_max_iterations_respected(self):
        """Should not hang even with max_iterations=1."""
        pts = [(17.0 + i * 0.01, 78.0 + i * 0.01) for i in range(6)]
        route, dist = solve_tsp(pts, max_iterations=1)
        assert len(route) == 6
        assert dist > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
