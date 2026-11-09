"""Compare search strategies on a synthetic delivery routing problem."""

import argparse
from itertools import permutations
from math import dist
from pathlib import Path
from random import Random


def make_stops(count, seed):
    """Return a central depot followed by unique synthetic delivery locations."""
    rng = Random(seed)
    depot = (50, 50)
    grid = [(x, y) for x in range(5, 96) for y in range(5, 96) if (x, y) != depot]
    return [depot, *rng.sample(grid, count)]


def distance_matrix(points):
    return [[dist(a, b) for b in points] for a in points]


def route_length(route, distances):
    """Measure a tour starting at the depot, including its return leg."""
    return sum(distances[a][b] for a, b in zip(route, (*route[1:], 0)))


def nearest_neighbor(distances):
    """Visit the closest unvisited delivery stop, breaking ties by stop ID."""
    route = [0]
    remaining = set(range(1, len(distances)))
    while remaining:
        last = route[-1]
        next_stop = min(remaining, key=lambda stop: (distances[last][stop], stop))
        route.append(next_stop)
        remaining.remove(next_stop)
    return tuple(route)


def two_opt(route, distances):
    """Reverse the segment for the best improving pair of edges until stable."""
    route = list(route)
    while True:
        best_move = None
        best_delta = -1e-9
        for first in range(1, len(route) - 1):
            a, b = route[first - 1:first + 1]
            for last in range(first + 1, len(route)):
                c, d = route[last], route[(last + 1) % len(route)]
                delta = distances[a][c] + distances[b][d] - distances[a][b] - distances[c][d]
                if delta < best_delta:
                    best_move = first, last
                    best_delta = delta
        if best_move is None:
            return tuple(route)
        first, last = best_move
        route[first:last + 1] = reversed(route[first:last + 1])


def order_crossover(first_parent, second_parent, rng):
    """Keep a slice from one parent and fill the remaining slots in donor order."""
    start, end = sorted(rng.sample(range(len(first_parent) + 1), 2))
    segment = first_parent[start:end]
    retained = set(segment)
    donor = [stop for stop in second_parent if stop not in retained]
    return (*donor[:start], *segment, *donor[start:])


def genetic_search(distances, seed, population_size=80, generations=120):
    """Use tournament selection, order-preserving crossover, reversal, and one elite."""
    rng = Random(seed)
    stops = tuple(range(1, len(distances)))
    population = [nearest_neighbor(distances)[1:]]
    population.extend(
        tuple(rng.sample(stops, len(stops))) for _ in range(population_size - 1)
    )

    def score(chromosome):
        return route_length((0, *chromosome), distances)

    for _ in range(generations):
        ranked = sorted((score(chromosome), chromosome) for chromosome in population)
        next_generation = [ranked[0][1]]
        while len(next_generation) < population_size:
            first_parent = min(rng.sample(ranked, 3))[1]
            second_parent = min(rng.sample(ranked, 3))[1]
            child = order_crossover(first_parent, second_parent, rng)
            if rng.random() < 0.25:
                first, last = sorted(rng.sample(range(len(child)), 2))
                child = (*child[:first], *reversed(child[first:last + 1]), *child[last + 1:])
            next_generation.append(child)
        population = next_generation
    return (0, *min(population, key=score))


def exact_search(distances):
    """Enumerate all delivery orders for small tours only."""
    if len(distances) > 9:
        raise ValueError("Exact search is limited to eight delivery stops.")
    routes = ((0, *order) for order in permutations(range(1, len(distances))))
    return min(routes, key=lambda route: route_length(route, distances))


def verify_route(route, count):
    if len(route) != count or route[0] != 0 or set(route) != set(range(count)):
        raise ValueError("A route must start at the depot and visit each stop once.")


def write_svg(path, points, results, distances):
    """Draw the baseline and the shortest computed tour on identical maps."""
    baseline = next(iter(results))
    shortest = min(results, key=lambda name: route_length(results[name], distances))
    panels = []
    for left, name, color in [(20, baseline, "#a3400a"), (390, shortest, "#007767")]:
        route = results[name]
        locations = [(left + 22 + x * 3, 438 - y * 3) for x, y in points]
        tour = " ".join(f"{locations[stop][0]},{locations[stop][1]}" for stop in (*route, 0))
        dots = "".join(
            f'<circle cx="{x}" cy="{y}" r="4.2" fill="{color}" stroke="white" stroke-width="1.5"/>'
            for x, y in locations[1:]
        )
        depot_x, depot_y = locations[0]
        length = route_length(route, distances)
        panels.append(f"""
    <rect x="{left}" y="100" width="350" height="347" rx="15" fill="#f3f7f9" stroke="#cfdae2"/>
    <text x="{left + 18}" y="127" fill="#17304a" font-size="15" font-weight="bold">{name}</text>
    <text x="{left + 330}" y="127" text-anchor="end" fill="{color}" font-size="15" font-weight="bold">{length:.2f} units</text>
    <polyline points="{tour}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>
    {dots}
    <rect x="{depot_x - 5}" y="{depot_y - 5}" width="10" height="10" fill="#17304a" stroke="white" stroke-width="1.5"/>""")
    chart = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 494" role="img" aria-labelledby="title description" width="760" height="494">
  <title id="title">Route Optimization Lab benchmark</title>
  <desc id="description">Two maps of the same synthetic stops. The nearest-neighbor route on the left is compared with the shortest computed route on the right.</desc>
  <rect width="760" height="494" rx="20" fill="#ffffff"/>
  <g font-family="system-ui, sans-serif">
    <text x="20" y="35" fill="#00675a" font-size="13" font-weight="bold" letter-spacing="1.4">ROUTE OPTIMIZATION LAB</text>
    <text x="20" y="66" fill="#17304a" font-size="27" font-weight="bold">Same stops, different tours</text>
    <text x="20" y="87" fill="#4a5f71" font-size="13">Synthetic delivery benchmark • {len(points) - 1} stops • Euclidean distance</text>
    {''.join(panels)}
    <rect x="20" y="467" width="10" height="10" fill="#17304a"/>
    <text x="38" y="476" fill="#4a5f71" font-size="13">Depot</text>
    <circle cx="104" cy="472" r="4.2" fill="#4a5f71"/>
    <text x="117" y="476" fill="#4a5f71" font-size="13">Delivery stop</text>
  </g>
</svg>
"""
    path.write_text(chart, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stops", type=int, default=24, help="number of delivery stops, excluding depot"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for both the map and the genetic search"
    )
    parser.add_argument(
        "--exact", action="store_true", help="include the true optimum for at most eight stops"
    )
    parser.add_argument("--svg", type=Path, help="write a side-by-side route comparison to this path")
    args = parser.parse_args()
    if not 3 <= args.stops <= 250:
        parser.error("--stops must be between 3 and 250.")
    if args.exact and args.stops > 8:
        parser.error("--exact requires --stops of at most 8.")

    points = make_stops(args.stops, args.seed)
    distances = distance_matrix(points)
    baseline = nearest_neighbor(distances)
    results = {
        "Nearest neighbor": baseline,
        "Nearest neighbor + 2-opt": two_opt(baseline, distances),
        "Genetic search": genetic_search(distances, args.seed),
    }
    if args.exact:
        results["Exact search"] = exact_search(distances)

    baseline_length = route_length(baseline, distances)
    print(f"Delivery stops: {args.stops} | Seed: {args.seed}")
    print(f"{'Algorithm':<26} {'Distance':>10} {'vs. baseline':>14}")
    for name, route in results.items():
        verify_route(route, len(points))
        length = route_length(route, distances)
        change = (length / baseline_length - 1) * 100
        print(f"{name:<26} {length:10.2f} {change:13.2f}%")

    if args.svg:
        write_svg(args.svg, points, results, distances)
        print(f"Saved {args.svg}")


if __name__ == "__main__":
    main()
