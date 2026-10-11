"""Web interface. Run with:  streamlit run app.py"""
import random

import matplotlib.pyplot as plt
import streamlit as st

import shortest_path_lab as lab

st.set_page_config(page_title="Shortest Path and Route Optimization", layout="wide")
st.title("Shortest Path and Route Optimization")
st.caption("Computational Mathematics Laboratory")

tab1, tab2, tab3 = st.tabs(["Graph shortest path", "Grid search", "Route optimization"])

with tab1:
    graph, pos = lab.sample_graph()
    c1, c2 = st.columns(2)
    src = c1.selectbox("From", list(graph), index=0)
    dst = c2.selectbox("To", list(graph), index=5)
    d, path, settled = lab.dijkstra(graph, src, dst)
    st.write(f"**Dijkstra:** cost {d}, path {' → '.join(path)}")
    lab.plot_graph(graph, pos, path, f"{src} to {dst} (cost {d})", "/tmp/g.png")
    st.image("/tmp/g.png")
    st.write("**All-pairs distances (Floyd-Warshall)**")
    fw = lab.floyd_warshall(graph)
    st.dataframe({v: [fw[u][v] for u in graph] for v in graph}, hide_index=False)

with tab2:
    seed = st.slider("Random map (seed)", 1, 50, 4)
    wall = st.slider("Wall density", 0.0, 0.4, 0.25)
    start, goal = (2, 2), (13, 29)
    grid = lab.random_grid(16, 32, wall=wall, seed=seed, start=start, goal=goal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rows = []
    for ax, m in zip(axes, ("bfs", "dijkstra", "astar")):
        p, cost, ex = lab.grid_search(grid, start, goal, m)
        rows.append({"method": m, "explored": len(ex), "cost": cost if p else "no path"})
        lab.plot_grid(grid, start, goal, ex, p, m, ax)
    st.pyplot(fig)
    st.table(rows)

with tab3:
    n = st.slider("Number of cities", 5, 60, 25)
    seed = st.slider("Random cities (seed)", 1, 50, 7, key="s2")
    rng = random.Random(seed)
    cities = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    nn = lab.nearest_neighbour(cities)
    opt, swaps = lab.two_opt(cities, nn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    lab.plot_tour(cities, nn, f"Nearest neighbour: {lab.tour_length(cities, nn):.1f}", axes[0])
    lab.plot_tour(cities, opt, f"After 2-opt: {lab.tour_length(cities, opt):.1f}", axes[1])
    st.pyplot(fig)
    st.write(f"2-opt made {swaps} swaps.")
