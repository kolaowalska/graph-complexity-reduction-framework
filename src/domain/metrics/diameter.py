from __future__ import annotations

import networkx as nx

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric

# TODO: decide whether diameter should be inf or the diameter of the largest connected component
@register_metric("diameter")
class Diameter(Metric):
    INFO = MetricInfo(
        name="diameter",
        version="0.1.0",
        description="graph diameter; if disconnected uses largest connected component."
    )

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        G = graph.to_networkx(copy=False)
        UG = G.to_undirected() if G.is_directed() else G

        component_size = 0
        if UG.number_of_nodes() == 0:
            val = 0.0
        elif nx.is_connected(UG):
            val = float(nx.diameter(UG))
            component_size = UG.number_of_nodes()
        else:
            largest_cc = max(nx.connected_components(UG), key=len)
            subgraph = UG.subgraph(largest_cc)
            val = float(nx.diameter(subgraph))
            component_size = len(largest_cc)

        return MetricResult(
            metric=self.INFO.name,
            summary={
                "diameter": val,
                # "component_nodes": component_size,
                # "total_nodes": UG.number_of_nodes()
            }
        )
