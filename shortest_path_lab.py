"""
Shortest Path and Route Optimization
Computational Mathematics Laboratory project

Part 1: shortest path on a weighted graph (Dijkstra, Bellman-Ford, Floyd-Warshall)
Part 2: shortest path on a grid (BFS, Dijkstra, A*)
Part 3: route optimization / travelling salesman (nearest neighbour, 2-opt, brute force)

Run:  python shortest_path_lab.py        (prints results and saves PNG figures to ./figures)
"""
import heapq
import itertools
import math
import os
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")  # lets the script save figures without a display
import matplotlib.pyplot as plt

INF = float("inf")


# ----------------------------------------------------------------------
# Part 1: weighted graph algorithms.  graph = {u: [(v, weight), ...]}
# ----------------------------------------------------------------------
def build_path(prev, src, dst):
    path, node = [], dst
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path and path[0] == src else []


def dijkstra(graph, src, dst):
    """Binary-heap Dijkstra. Needs non-negative weights. O((V+E) log V)."""
    dist = {u: INF for u in graph}
    prev = {src: None}
    dist[src] = 0
    heap, settled = [(0, src)], 0
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        settled += 1
        if u == dst:
            break
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
    return dist[dst], build_path(prev, src, dst), settled


def bellman_ford(graph, src, dst):
    """Allows negative weights and detects negative cycles. O(V*E)."""
    dist = {u: INF for u in graph}
    prev = {src: None}
    dist[src] = 0
    for _ in range(len(graph) - 1):
        changed = False
        for u in graph:
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v], prev[v], changed = dist[u] + w, u, True
        if not changed:
            break
    for u in graph:
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                raise ValueError("graph contains a negative cycle")
    return dist[dst], build_path(prev, src, dst)


def floyd_warshall(graph):
    """All-pairs shortest distances. O(V^3)."""
    nodes = list(graph)
    d = {u: {v: (0 if u == v else INF) for v in nodes} for u in nodes}
    for u in graph:
        for v, w in graph[u]:
            d[u][v] = min(d[u][v], w)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


# ----------------------------------------------------------------------
# Part 2: grid search.  grid[r][c] = 0 free, 1 wall, 2 mud (cost 5)
# ----------------------------------------------------------------------
def cell_cost(grid, cell):
    return 5 if grid[cell[0]][cell[1]] == 2 else 1


def neighbours(grid, cell):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != 1:
            yield (nr, nc)


def grid_search(grid, start, goal, method="dijkstra"):
    """method: 'bfs' (ignores weights), 'dijkstra' or 'astar' (Manhattan heuristic)."""
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    g, prev, explored = {start: 0}, {start: None}, []
    done = set()
    if method == "bfs":
        queue = deque([start])
        seen = {start}
        while queue:
            u = queue.popleft()
            explored.append(u)
            if u == goal:
                break
            for v in neighbours(grid, u):
                if v not in seen:
                    seen.add(v)
                    prev[v] = u
                    queue.append(v)
    else:
        heap = [(h(start) if method == "astar" else 0, start)]
        while heap:
            _, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            explored.append(u)
            if u == goal:
                break
            for v in neighbours(grid, u):
                ng = g[u] + cell_cost(grid, v)
                if ng < g.get(v, INF):
                    g[v], prev[v] = ng, u
                    heapq.heappush(heap, (ng + (h(v) if method == "astar" else 0), v))
    path, node = [], goal if goal in prev else None
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    cost = sum(cell_cost(grid, c) for c in path[1:]) if path else INF
    return path, cost, explored


def random_grid(rows, cols, wall=0.25, mud=0.08, seed=1, start=(0, 0), goal=None):
    rng = random.Random(seed)
    goal = goal or (rows - 1, cols - 1)
    grid = [[1 if rng.random() < wall else (2 if rng.random() < mud else 0)
             for _ in range(cols)] for _ in range(rows)]
    grid[start[0]][start[1]] = grid[goal[0]][goal[1]] = 0
    return grid


# ----------------------------------------------------------------------
# Part 3: route optimization (TSP).  cities = [(x, y), ...]
# ----------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_length(cities, tour):
    return sum(dist(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def nearest_neighbour(cities, start=0):
    left = set(range(len(cities))) - {start}
    tour = [start]
    while left:
        nxt = min(left, key=lambda j: dist(cities[tour[-1]], cities[j]))
        tour.append(nxt)
        left.remove(nxt)
    return tour


def two_opt(cities, tour):
    """Repeat 2-opt swaps until no swap shortens the tour. Returns (tour, swaps)."""
    tour, swaps, n = tour[:], 0, len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = cities[tour[i - 1]], cities[tour[i]]
                c, d = cities[tour[j]], cities[tour[(j + 1) % n]]
                if dist(a, c) + dist(b, d) - dist(a, b) - dist(c, d) < -1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    swaps += 1
                    improved = True
    return tour, swaps


def brute_force(cities):
    """Exact optimum by trying every permutation. Only practical for n <= 9."""
    best, best_len = None, INF
    for perm in itertools.permutations(range(1, len(cities))):
        t = [0, *perm]
        length = tour_length(cities, t)
        if length < best_len:
            best, best_len = t, length
    return best


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def plot_graph(graph, pos, path, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    for u in graph:
        for v, w in graph[u]:
            if u < v:
                ax.plot(*zip(pos[u], pos[v]), color="#b8c2bd", lw=1.5, zorder=1)
                mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
                ax.text(*mid, str(w), fontsize=8, ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=1))
    if path:
        xs, ys = zip(*(pos[n] for n in path))
        ax.plot(xs, ys, color="#d4155a", lw=3.5, zorder=2)
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=500, color="#16232e", zorder=3)
        ax.text(x, y, n, color="white", ha="center", va="center", fontsize=10, zorder=4)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(filename, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_grid(grid, start, goal, explored, path, title, ax):
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#16232e"))
            elif grid[r][c] == 2:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#c9a86a"))
    for r, c in explored:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#9fd3cb"))
    if path:
        ax.plot([c + .5 for r, c in path], [r + .5 for r, c in path], color="#d4155a", lw=2.5)
    ax.scatter(start[1] + .5, start[0] + .5, s=60, color="#1f7a4d", zorder=3)
    ax.scatter(goal[1] + .5, goal[0] + .5, s=60, color="#2b4fd8", zorder=3)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_tour(cities, tour, title, ax):
    pts = [cities[i] for i in tour] + [cities[tour[0]]]
    ax.plot(*zip(*pts), color="#d4155a", lw=1.8, zorder=1)
    ax.scatter(*zip(*cities), color="#16232e", s=25, zorder=2)
    ax.scatter(*cities[tour[0]], color="#1f7a4d", s=80, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
def sample_graph():
    edges = [("A", "B", 4), ("A", "C", 2), ("B", "C", 1), ("B", "D", 5), ("C", "D", 8),
             ("C", "E", 10), ("D", "E", 2), ("D", "F", 6), ("E", "F", 3)]
    graph = {n: [] for n in "ABCDEF"}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    pos = {"A": (0, 1), "B": (2, 2), "C": (2, 0), "D": (4, 2), "E": (4, 0), "F": (6, 1)}
    return graph, pos


def main():
    os.makedirs("figures", exist_ok=True)

    print("=== Part 1: weighted graph, A to F ===")
    graph, pos = sample_graph()
    d, path, settled = dijkstra(graph, "A", "F")
    print(f"Dijkstra     : cost {d}, path {' -> '.join(path)}, vertices settled {settled}")
    d2, path2 = bellman_ford(graph, "A", "F")
    print(f"Bellman-Ford : cost {d2}, path {' -> '.join(path2)}")
    fw = floyd_warshall(graph)
    print(f"Floyd-Warshall A to F: {fw['A']['F']}")
    assert d == d2 == fw["A"]["F"], "algorithms disagree"
    plot_graph(graph, pos, path, f"Shortest path A to F (cost {d})", "figures/graph_shortest_path.png")

    print("\n=== Part 2: grid search (mud costs 5) ===")
    rows, cols = 16, 32
    start, goal = (2, 2), (13, 29)
    grid = random_grid(rows, cols, seed=4, start=start, goal=goal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    print(f"{'method':10} {'explored':>9} {'steps':>6} {'cost':>6}")
    for ax, m in zip(axes, ("bfs", "dijkstra", "astar")):
        p, cost, ex = grid_search(grid, start, goal, m)
        steps = len(p) - 1 if p else None
        print(f"{m:10} {len(ex):>9} {steps!s:>6} {cost:>6}")
        plot_grid(grid, start, goal, ex, p, f"{m}: explored {len(ex)}, cost {cost}", ax)
    fig.savefig("figures/grid_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Part 3: route optimization ===")
    rng = random.Random(7)
    cities = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(40)]
    nn = nearest_neighbour(cities)
    opt, swaps = two_opt(cities, nn)
    rand = list(range(len(cities)))
    rng.shuffle(rand)
    L = lambda t: tour_length(cities, t)
    print(f"random order : {L(rand):8.1f}")
    print(f"nearest nbr  : {L(nn):8.1f}")
    print(f"+ 2-opt      : {L(opt):8.1f}  ({swaps} swaps, {100 * (1 - L(opt) / L(nn)):.1f}% shorter)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_tour(cities, nn, f"Nearest neighbour: {L(nn):.1f}", axes[0])
    plot_tour(cities, opt, f"After 2-opt: {L(opt):.1f}", axes[1])
    fig.savefig("figures/tsp_tours.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    small = cities[:8]
    ex = brute_force(small)
    h, _ = two_opt(small, nearest_neighbour(small))
    print(f"8 cities, exact optimum {tour_length(small, ex):.1f} vs NN+2-opt {tour_length(small, h):.1f}")
    print("\nFigures saved in ./figures")


if __name__ == "__main__":
    main()
