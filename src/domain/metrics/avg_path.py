from __future__ import annotations

import networkx as nx

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric

@register_metric("avg_path_length")
class AvgPathLength(Metric):
    INFO = MetricInfo(
        name="average path length",
        description="average shortest path length on largest connected component",
    )

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        g = graph.to_networkx(copy=False)
        weight_arg = "weight" if graph.is_weighted() else None

        g_undirected = g.to_undirected() if g.is_directed() else g

        if g_undirected.number_of_nodes() <= 1:
            val = 0.0
        elif nx.is_connected(g_undirected):
            val = nx.average_shortest_path_length(g_undirected, weight=weight_arg)
        else:
            largest_cc = max(nx.connected_components(g_undirected), key=len)
            subgraph = g_undirected.subgraph(largest_cc)
            try:
                val = nx.average_shortest_path_length(subgraph, weight=weight_arg)
            except Exception:
                val = -1.0

        return MetricResult(
            metric=self.INFO.name,
            summary={"avg": float(val), "weighted": bool(weight_arg)}
        )
