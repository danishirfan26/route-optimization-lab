from pathlib import Path
from random import Random
from subprocess import run
from sys import executable
import unittest
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from route_lab import (
    distance_matrix,
    exact_search,
    genetic_search,
    make_stops,
    nearest_neighbor,
    order_crossover,
    route_length,
    two_opt,
    verify_route,
)


class RouteLabTests(unittest.TestCase):
    def test_seed_creates_distinct_stops_and_is_repeatable(self):
        stops = make_stops(24, 4)
        self.assertEqual(len(set(stops)), 25)
        self.assertEqual(stops, make_stops(24, 4))
        self.assertNotEqual(stops, make_stops(24, 3))

    def test_distance_includes_return_leg(self):
        distances = distance_matrix([(0, 0), (3, 0), (3, 4)])
        self.assertEqual(route_length((0, 1, 2), distances), 12)

    def test_two_opt_uncrosses_a_square(self):
        distances = distance_matrix([(0, 0), (1, 1), (1, 0), (0, 1)])
        initial = (0, 1, 2, 3)
        improved = two_opt(initial, distances)
        verify_route(improved, len(distances))
        self.assertGreater(route_length(initial, distances), 4)
        self.assertAlmostEqual(route_length(improved, distances), 4)

    def test_order_crossover_keeps_each_stop_once(self):
        first_parent = (1, 2, 3, 4, 5)
        second_parent = tuple(reversed(first_parent))
        for seed in range(20):
            child = order_crossover(first_parent, second_parent, Random(seed))
            self.assertCountEqual(child, first_parent)

    def test_genetic_search_is_seeded_and_preserves_baseline(self):
        distances = distance_matrix(make_stops(12, 7))
        route = genetic_search(distances, 7, population_size=20, generations=30)
        repeated = genetic_search(distances, 7, population_size=20, generations=30)
        self.assertEqual(route, repeated)
        verify_route(route, len(distances))
        baseline = nearest_neighbor(distances)
        self.assertLessEqual(route_length(route, distances), route_length(baseline, distances))

    def test_exact_matches_a_known_optimum(self):
        distances = distance_matrix([(0, 0), (2, 0), (2, 3), (0, 3)])
        result = exact_search(distances)
        verify_route(result, len(distances))
        self.assertAlmostEqual(route_length(result, distances), 10)

    def test_duplicate_stop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "visit each stop once"):
            verify_route((0, 1, 1), 3)

    def test_cli_writes_an_svg_with_two_routes(self):
        script = Path(__file__).with_name("route_lab.py")
        with TemporaryDirectory() as directory:
            svg = Path(directory, "routes.svg")
            result = run(
                [executable, str(script), "--stops", "4", "--exact", "--svg", str(svg)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("Exact search", result.stdout)
            document = ElementTree.parse(svg)
            tours = document.findall(".//{http://www.w3.org/2000/svg}polyline")
            self.assertEqual(len(tours), 2)
            for tour in tours:
                stops = tour.get("points").split()
                self.assertEqual(len(stops), 6)
                self.assertEqual(stops[0], stops[-1])

    def test_cli_rejects_large_exact_search(self):
        script = Path(__file__).with_name("route_lab.py")
        result = run(
            [executable, str(script), "--stops", "9", "--exact"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--exact requires --stops of at most 8", result.stderr)


if __name__ == "__main__":
    unittest.main()
