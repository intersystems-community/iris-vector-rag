#!/usr/bin/env python3
"""
Graph Analytics Benchmark: Arno vs COS vs NetworkX vs Neo4j GDS

Four engines, same synthetic graphs, same five algorithms.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _make_connection(host: str, port: int, namespace: str, user: str, password: str) -> Any:
    try:
        import iris as _iris  # type: ignore[import]
        return _iris.connect(f"{host}:{port}/{namespace}", user, password)
    except Exception:
        pass
    import intersystems_iris as _iris2  # type: ignore[import]
    return _iris2.createConnection(host, port, namespace, user, password)


def _call_class_method(conn: Any, class_name: str, method_name: str, *args: Any) -> str:
    try:
        import iris as _iris  # type: ignore[import]
        return str(_iris.createIRIS(conn).classMethodValue(class_name, method_name, *args))
    except Exception:
        pass
    import intersystems_iris as _iris2  # type: ignore[import]
    return str(_iris2.createIRIS(conn).classMethodValue(class_name, method_name, *args))


def seed_graph_to_kg(conn: Any, n_nodes: int, branching: int = 3, prefix: str = "BN") -> List[str]:
    rng = random.Random(42)
    node_ids = [f"{prefix}_{i}" for i in range(n_nodes)]
    n = len(node_ids)

    try:
        import iris as _iris
        ir = _iris.createIRIS(conn)
    except Exception:
        import intersystems_iris as _iris2
        ir = _iris2.createIRIS(conn)

    ir.kill("^KG")
    ir.kill("^ArnoKG")

    for src_idx in range(n):
        src = node_ids[src_idx]
        targets: set = set()
        for _ in range(branching):
            dst_idx = rng.randint(0, n - 1)
            if dst_idx != src_idx:
                targets.add(dst_idx)
        if targets:
            ir.set(len(targets), "^KG", "deg", src)
            for dst_idx in targets:
                dst = node_ids[dst_idx]
                ir.set(1, "^KG", "out", src, "REL", dst)
                ir.set(1, "^KG", "in",  dst, "REL", src)

    ir.set(1, "^KG", "__version")
    return node_ids


def teardown_kg(conn: Any) -> None:
    try:
        import iris as _iris
        ir = _iris.createIRIS(conn)
    except Exception:
        import intersystems_iris as _iris2
        ir = _iris2.createIRIS(conn)
    ir.kill("^KG")
    ir.kill("^ArnoKG")


def _build_nx_graph(node_ids: List[str], branching: int) -> Any:
    import networkx as nx
    rng = random.Random(42)
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    n = len(node_ids)
    for src_idx, src in enumerate(node_ids):
        targets: set = set()
        for _ in range(branching):
            dst_idx = rng.randint(0, n - 1)
            if dst_idx != src_idx:
                targets.add(dst_idx)
        for dst_idx in targets:
            G.add_edge(src, node_ids[dst_idx])
    return G


def _build_neo4j_graph(driver: Any, node_ids: List[str], branching: int) -> None:
    rng = random.Random(42)
    n = len(node_ids)
    edges: List[Tuple[str, str]] = []
    for src_idx, src in enumerate(node_ids):
        targets: set = set()
        for _ in range(branching):
            dst_idx = rng.randint(0, n - 1)
            if dst_idx != src_idx:
                targets.add(dst_idx)
        for dst_idx in targets:
            edges.append((src, node_ids[dst_idx]))

    with driver.session() as s:
        s.run("MATCH (n:BenchNode) DETACH DELETE n")
        s.run(
            "UNWIND $ids AS id CREATE (:BenchNode {id: id})",
            ids=node_ids,
        )
        s.run(
            "UNWIND $edges AS e "
            "MATCH (a:BenchNode {id: e[0]}), (b:BenchNode {id: e[1]}) "
            "CREATE (a)-[:REL]->(b)",
            edges=edges,
        )
        try:
            s.run("CALL gds.graph.drop('bench', false)")
        except Exception:
            pass
        s.run(
            "CALL gds.graph.project('bench', 'BenchNode', "
            "{REL: {orientation: 'NATURAL'}})"
        )


def _teardown_neo4j(driver: Any) -> None:
    with driver.session() as s:
        try:
            s.run("CALL gds.graph.drop('bench', false)")
        except Exception:
            pass
        s.run("MATCH (n:BenchNode) DETACH DELETE n")


def _timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def benchmark_iris(conn: Any, class_name: str, method_name: str,
                   method_args: Tuple, runs: int) -> Dict[str, float]:
    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _call_class_method(conn, class_name, method_name, *method_args)
        times.append(time.perf_counter() - t0)
    return {"mean": statistics.mean(times), "stdev": statistics.stdev(times) if runs > 1 else 0.0}


def benchmark_nx(nx_graph: Any, algo: str, seeds: List[str], runs: int) -> Dict[str, float]:
    import networkx as nx

    def run_once() -> None:
        if algo == "pagerank_global":
            nx.pagerank(nx_graph, alpha=0.85, max_iter=20)
        elif algo == "ppr":
            seed_dict = {s: 1.0 / len(seeds) for s in seeds if s in nx_graph}
            nx.pagerank(nx_graph, alpha=0.85, max_iter=20, personalization=seed_dict)
        elif algo == "wcc":
            list(nx.weakly_connected_components(nx_graph))
        elif algo == "cdlp":
            list(nx.community.label_propagation_communities(nx_graph.to_undirected()))
        elif algo == "subgraph":
            seed = seeds[0] if seeds else next(iter(nx_graph.nodes()))
            nx.single_source_shortest_path(nx_graph, seed, cutoff=2)
        elif algo == "khop_sample":
            seed = seeds[0] if seeds else next(iter(nx_graph.nodes()))
            # BFS with fanout=16 for 2 hops — equivalent to Kumo's GraphSAGE sampler
            frontier = {seed}
            for _ in range(2):
                next_f = set()
                for n in frontier:
                    nbrs = list(nx_graph.successors(n))[:16]
                    next_f.update(nbrs)
                frontier = next_f
        elif algo == "random_walk":
            seed = seeds[0] if seeds else next(iter(nx_graph.nodes()))
            for _ in range(10):
                cur = seed
                for _ in range(20):
                    nbrs = list(nx_graph.successors(cur))
                    if not nbrs:
                        break
                    cur = random.choice(nbrs)
        elif algo == "neighbor_agg":
            seed = seeds[0] if seeds else next(iter(nx_graph.nodes()))
            for hop in range(2):
                for n in list(nx.single_source_shortest_path_length(
                    nx_graph, seed, cutoff=hop + 1
                ).keys()):
                    nbrs = list(nx_graph.successors(n))
                    if nbrs:
                        _ = sum(nx_graph.degree(nb) for nb in nbrs) / len(nbrs)

    times = [_timed(run_once) for _ in range(runs)]
    return {"mean": statistics.mean(times), "stdev": statistics.stdev(times) if runs > 1 else 0.0}


def benchmark_neo4j(driver: Any, algo: str, seeds: List[str], runs: int) -> Dict[str, float]:
    def run_once() -> None:
        with driver.session() as s:
            if algo == "pagerank_global":
                s.run(
                    "CALL gds.pageRank.stream('bench', "
                    "{maxIterations: 20, dampingFactor: 0.85}) "
                    "YIELD nodeId, score RETURN count(*)"
                ).consume()
            elif algo == "ppr":
                s.run(
                    "CALL gds.pageRank.stream('bench', "
                    "{maxIterations: 20, dampingFactor: 0.85}) "
                    "YIELD nodeId, score RETURN count(*)"
                ).consume()
            elif algo == "wcc":
                s.run(
                    "CALL gds.wcc.stream('bench') "
                    "YIELD nodeId, componentId RETURN count(*)"
                ).consume()
            elif algo == "cdlp":
                s.run(
                    "CALL gds.labelPropagation.stream('bench') "
                    "YIELD nodeId, communityId RETURN count(*)"
                ).consume()
            elif algo == "subgraph":
                seed = seeds[0] if seeds else None
                if seed:
                    s.run(
                        "MATCH (src:BenchNode {id: $id}) "
                        "CALL gds.bfs.stream('bench', "
                        "{sourceNode: id(src), maxDepth: 2}) "
                        "YIELD path RETURN count(*)",
                        id=seed,
                    ).consume()
            elif algo == "khop_sample":
                seed = seeds[0] if seeds else None
                if seed:
                    s.run(
                        "MATCH (src:BenchNode {id: $id}) "
                        "CALL gds.bfs.stream('bench', "
                        "{sourceNode: id(src), maxDepth: 2}) "
                        "YIELD path RETURN count(*)",
                        id=seed,
                    ).consume()
            elif algo == "random_walk":
                seed = seeds[0] if seeds else None
                if seed:
                    s.run(
                        "MATCH (src:BenchNode {id: $id}) "
                        "CALL gds.randomWalk.stream('bench', "
                        "{sourceNodes: [id(src)], walkLength: 20, walksPerNode: 10}) "
                        "YIELD nodeIds RETURN count(*)",
                        id=seed,
                    ).consume()
            elif algo == "neighbor_agg":
                seed = seeds[0] if seeds else None
                if seed:
                    s.run(
                        "MATCH (src:BenchNode {id: $id})-[*1..2]->(nbr:BenchNode) "
                        "RETURN avg(size([(nbr)-[:REL]->() | 1])) AS mean_deg",
                        id=seed,
                    ).consume()

    times = [_timed(run_once) for _ in range(runs)]
    return {"mean": statistics.mean(times), "stdev": statistics.stdev(times) if runs > 1 else 0.0}


ALGO_SPECS = [
    ("pagerank_global", "Graph.KG.PageRank", "PageRankGlobalJson",
     "Graph.KG.ArnoAccel", "PageRankGlobalJson", False, (0.85, 20)),
    ("ppr",             "Graph.KG.PageRank", "RunJson",
     "Graph.KG.ArnoAccel", "PPRJson",             True,  (0.85, 20)),
    ("wcc",             "Graph.KG.Algorithms", "WCCJson",
     "Graph.KG.ArnoAccel", "WCCJson",             False, (10,)),
    ("cdlp",            "Graph.KG.Algorithms", "CDLPJson",
     "Graph.KG.ArnoAccel", "CDLPJson",            False, (10,)),
]


def run_size_benchmark(
    conn: Any,
    neo4j_driver: Optional[Any],
    n_nodes: int,
    branching: int,
    runs: int,
) -> Dict[str, Any]:
    print(f"  Seeding ^KG ({n_nodes} nodes, branching={branching}) ...", flush=True)
    node_ids = seed_graph_to_kg(conn, n_nodes, branching=branching)
    rng = random.Random(42)
    ppr_seeds = rng.sample(node_ids, min(3, len(node_ids)))
    ppr_seeds_json = json.dumps(ppr_seeds)

    print(f"  Building NetworkX graph ...", flush=True)
    nx_graph = _build_nx_graph(node_ids, branching)

    if neo4j_driver:
        print(f"  Loading Neo4j graph ...", flush=True)
        _build_neo4j_graph(neo4j_driver, node_ids, branching)

    size_results: Dict[str, Any] = {"n_nodes": n_nodes, "branching": branching, "algos": {}}

    for algo, cos_cls, cos_meth, arno_cls, arno_meth, needs_seeds, extra_args in ALGO_SPECS:
        print(f"    {algo} ...", end=" ", flush=True)

        if needs_seeds:
            cos_args = (ppr_seeds_json,) + extra_args
            arno_args = cos_args
        else:
            cos_args = extra_args
            arno_args = extra_args

        cos_st   = benchmark_iris(conn, cos_cls, cos_meth, cos_args, runs)
        arno_st  = benchmark_iris(conn, arno_cls, arno_meth, arno_args, runs)
        nx_st    = benchmark_nx(nx_graph, algo, ppr_seeds, runs)
        neo4j_st = benchmark_neo4j(neo4j_driver, algo, ppr_seeds, runs) if neo4j_driver else None

        size_results["algos"][algo] = {
            "cos": cos_st, "arno": arno_st, "nx": nx_st,
            "neo4j": neo4j_st,
        }

        parts = [
            f"COS={cos_st['mean']*1000:.1f}ms",
            f"Arno={arno_st['mean']*1000:.1f}ms",
            f"NX={nx_st['mean']*1000:.1f}ms",
        ]
        if neo4j_st:
            parts.append(f"Neo4j={neo4j_st['mean']*1000:.1f}ms")
        print("  ".join(parts), flush=True)

    print(f"    subgraph ...", end=" ", flush=True)
    sg_seed_json = json.dumps([node_ids[0]])
    cos_sg   = benchmark_iris(conn, "Graph.KG.Subgraph", "SubgraphJson",
                              (sg_seed_json, 2, "", 1000), runs)
    arno_sg  = benchmark_iris(conn, "Graph.KG.ArnoAccel", "SubgraphJson",
                              (sg_seed_json, 2, "", 1000), runs)
    nx_sg    = benchmark_nx(nx_graph, "subgraph", [node_ids[0]], runs)
    neo4j_sg = benchmark_neo4j(neo4j_driver, "subgraph", [node_ids[0]], runs) if neo4j_driver else None

    size_results["algos"]["subgraph"] = {
        "cos": cos_sg, "arno": arno_sg, "nx": nx_sg, "neo4j": neo4j_sg,
    }
    parts = [f"COS={cos_sg['mean']*1000:.1f}ms", f"Arno={arno_sg['mean']*1000:.1f}ms",
             f"NX={nx_sg['mean']*1000:.1f}ms"]
    if neo4j_sg:
        parts.append(f"Neo4j={neo4j_sg['mean']*1000:.1f}ms")
    print("  ".join(parts), flush=True)

    kumo_seeds_json = json.dumps([node_ids[0]])
    kumo_seed_list = [node_ids[0]]

    for kumo_algo, arno_meth, arno_extra in [
        ("khop_sample",  "KhopSampleJson",  (2, 16)),
        ("random_walk",  "RandomWalkJson",  (20, 10)),
        ("neighbor_agg", "NeighborAggJson", (2,)),
    ]:
        print(f"    {kumo_algo} ...", end=" ", flush=True)
        arno_st = benchmark_iris(conn, "Graph.KG.ArnoAccel", arno_meth,
                                 (kumo_seeds_json,) + arno_extra, runs)
        nx_st = benchmark_nx(nx_graph, kumo_algo, kumo_seed_list, runs)
        neo4j_st = (benchmark_neo4j(neo4j_driver, kumo_algo, kumo_seed_list, runs)
                    if neo4j_driver else None)
        size_results["algos"][kumo_algo] = {
            "cos": None, "arno": arno_st, "nx": nx_st, "neo4j": neo4j_st,
        }
        parts = [f"Arno={arno_st['mean']*1000:.1f}ms", f"NX={nx_st['mean']*1000:.1f}ms"]
        if neo4j_st:
            parts.append(f"Neo4j={neo4j_st['mean']*1000:.1f}ms")
        print("  ".join(parts), flush=True)

    if neo4j_driver:
        _teardown_neo4j(neo4j_driver)

    return size_results


def _ms(v: float) -> str:
    if v < 0.001:
        return f"{v*1e6:.0f}µs"
    if v < 1.0:
        return f"{v*1000:.2f}ms"
    return f"{v:.3f}s"


def _sp(baseline: float, other: float) -> str:
    if other <= 0:
        return "-"
    r = baseline / other
    return f"**{r:.1f}x**" if r > 1.05 else f"{r:.2f}x"


def generate_markdown(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    ts = results["timestamp"]
    has_neo4j = any(
        results["sizes"][0]["algos"].get(a, {}).get("neo4j") is not None
        for a in ["pagerank_global", "wcc"]
    )

    lines.append("# Graph Analytics Benchmark: COS vs Arno vs NetworkX vs Neo4j GDS")
    lines.append("")
    lines.append(f"**Generated:** {ts}  ")
    lines.append(f"**IRIS:** {results['connection']['host']}:{results['connection']['port']}  ")
    lines.append(f"**Arno callout loaded:** {'Yes ✓' if results['arno_available'] else 'No'}  ")
    lines.append(f"**Repetitions:** {results['runs']}  ")
    lines.append("")
    lines.append("> Arno times include `BuildGraphJson()` on the **first** call (cold); "
                 "subsequent calls with the same graph hit the `^ArnoKG` shadow-global cache (warm).")
    lines.append("")

    algo_names = ["pagerank_global", "ppr", "wcc", "cdlp", "subgraph",
                  "khop_sample", "random_walk", "neighbor_agg"]
    for size_data in results["sizes"]:
        n = size_data["n_nodes"]
        b = size_data["branching"]
        lines.append(f"## {n} nodes  (branching={b})")
        lines.append("")

        hdr = "| Algorithm | COS | Arno | NX | vs COS | vs NX |"
        sep = "|-----------|-----|------|----|--------|-------|"
        if has_neo4j:
            hdr += " Neo4j GDS | vs COS |"
            sep += "-----------|--------|"
        lines.append(hdr)
        lines.append(sep)

        algos = size_data["algos"]
        for algo in algo_names:
            if algo not in algos:
                continue
            d = algos[algo]
            arno_m = d["arno"]["mean"]
            nx_m   = d["nx"]["mean"]
            if d["cos"] is None:
                cos_cell = "—"
                arno_vs_cos = "—"
                n4j_vs_cos = "—"
                cos_m = 0.0
            else:
                cos_m = d["cos"]["mean"]
                cos_cell = _ms(cos_m)
                arno_vs_cos = _sp(cos_m, arno_m)
                n4j_vs_cos = _sp(cos_m, d["neo4j"]["mean"]) if has_neo4j and d.get("neo4j") else "—"
            row = (
                f"| {algo} "
                f"| {cos_cell} "
                f"| {_ms(arno_m)} "
                f"| {_ms(nx_m)} "
                f"| {arno_vs_cos} "
                f"| {_sp(nx_m, arno_m)} |"
            )
            if has_neo4j and d.get("neo4j"):
                n4j_m = d["neo4j"]["mean"]
                row += f" {_ms(n4j_m)} | {n4j_vs_cos} |"
            lines.append(row)
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- **COS** — pure ObjectScript (`Graph.KG.PageRank` / `Algorithms`), no caching")
    lines.append("- **Arno** — Rust engine via `$ZF(-5)`, reads pre-serialized JSON from `^ArnoKG` "
                 "shadow global; warm-cache path amortizes serialization cost across repeated calls")
    lines.append("- **NX** — NetworkX in-process Python, graph already in memory (no I/O)")
    lines.append("- **Neo4j GDS** — bolt round-trip to local Neo4j 5, graph projected in-memory")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=32769)
    parser.add_argument("--namespace", default="USER")
    parser.add_argument("--user", default="_SYSTEM")
    parser.add_argument("--password", default="SYS")
    parser.add_argument("--sizes", default="100,500,1000,5000")
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/graph_analytics_benchmark")
    parser.add_argument("--arno-lib", default="/tmp/libarno_callout.so")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="benchmark")
    parser.add_argument("--no-neo4j", action="store_true")
    parser.add_argument("--no-teardown", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print(f"Connecting to IRIS {args.host}:{args.port} ...", flush=True)
    conn = _make_connection(args.host, args.port, args.namespace, args.user, args.password)
    print("Connected.", flush=True)

    try:
        _call_class_method(conn, "Graph.KG.ArnoAccel", "Load", args.arno_lib)
    except Exception:
        pass
    try:
        arno_available = _call_class_method(
            conn, "Graph.KG.ArnoAccel", "IsAvailable"
        ).strip() not in ("", "0", "false", "False")
    except Exception:
        arno_available = False
    print(f"Arno callout: {'available' if arno_available else 'unavailable'}", flush=True)

    neo4j_driver = None
    if not args.no_neo4j:
        try:
            from neo4j import GraphDatabase  # type: ignore[import]
            import warnings
            neo4j_driver = GraphDatabase.driver(
                args.neo4j_uri,
                auth=(args.neo4j_user, args.neo4j_password),
                notifications_min_severity="OFF",
            )
            neo4j_driver.verify_connectivity()
            print(f"Neo4j: connected at {args.neo4j_uri}", flush=True)
        except Exception as e:
            print(f"Neo4j: unavailable ({e}) — skipping", flush=True)
            neo4j_driver = None

    print()
    all_results: List[Dict[str, Any]] = []

    for n_nodes in sizes:
        branching = min(args.branching, n_nodes - 1)
        print(f"=== n_nodes={n_nodes} ===", flush=True)
        teardown_kg(conn)

        size_data = run_size_benchmark(conn, neo4j_driver, n_nodes, branching, args.runs)
        all_results.append(size_data)

        if not args.no_teardown:
            teardown_kg(conn)
        print()

    if neo4j_driver:
        neo4j_driver.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_results = {
        "timestamp": timestamp,
        "connection": {"host": args.host, "port": args.port, "namespace": args.namespace},
        "arno_available": arno_available,
        "neo4j_available": neo4j_driver is not None,
        "runs": args.runs,
        "sizes": all_results,
    }

    json_path = output_dir / f"graph_analytics_benchmark_{timestamp}.json"
    json_path.write_text(json.dumps(full_results, indent=2))
    md = generate_markdown(full_results)
    md_path = output_dir / f"graph_analytics_benchmark_{timestamp}.md"
    md_path.write_text(md)
    (output_dir / "latest.json").write_text(json.dumps(full_results, indent=2))
    (output_dir / "latest.md").write_text(md)
    print(f"Results: {json_path}")
    print(f"Report:  {md_path}")
    print()

    cols = ["COS", "Arno", "NX", "Neo4j"]
    keys = ["cos", "arno", "nx", "neo4j"]
    algo_names = ["pagerank_global", "ppr", "wcc", "cdlp", "subgraph",
                  "khop_sample", "random_walk", "neighbor_agg"]
    size_map = {r["n_nodes"]: r for r in all_results}
    header = f"{'Algorithm':<20}" + "".join(f"{f'1k {c}':>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for algo in algo_names:
        row = [f"{algo:<20}"]
        if 1000 in size_map and algo in size_map[1000]["algos"]:
            d = size_map[1000]["algos"][algo]
            for k in keys:
                v = d.get(k)
                row.append(f"{_ms(v['mean']) if v else '-':>12}")
        print("".join(row))


if __name__ == "__main__":
    main()
