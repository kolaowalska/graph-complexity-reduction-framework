from __future__ import annotations

import networkx as nx

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric


@register_metric("diameter")
class Diameter(Metric):
    INFO = MetricInfo(
        name="diameter",
        type="absolute",
        description="graph diameter; if disconnected uses largest connected component."
    )

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        g = graph.to_networkx(copy=False)
        weight_arg = "weight" if graph.is_weighted() else None
        g_undirected = g.to_undirected() if g.is_directed() else g

        if g_undirected.number_of_nodes() == 0:
            return MetricResult(metric=self.INFO.name, summary={"diameter": 0.0})

        if nx.is_connected(g_undirected):
            subgraph = g_undirected
        else:
            largest_cc = max(nx.connected_components(g_undirected), key=len)
            subgraph = g_undirected.subgraph(largest_cc)

        return MetricResult(
            metric=self.INFO.name,
            summary={
                "diameter": float(nx.diameter(subgraph, weight=weight_arg)),
                "component_nodes": subgraph.number_of_nodes(),
                "total_nodes": g_undirected.number_of_nodes(),
                "weighted": bool(weight_arg),
            },
        )
