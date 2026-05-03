from __future__ import annotations

import networkx as nx

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric

@register_metric("connectivity")
class Connectivity(Metric):
    INFO = MetricInfo(
        name="connectivity",
        type="absolute",
        description=("structural connectivity profile:"
                    "number of connected components, "
                    "size of the largest component as a fraction of total nodes, "
                    "and algebraic connectivity (fiedler value) of the largest connected component"
        ),
    )

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        g = graph.to_networkx(copy=False)
        g_undirected = g.to_undirected() if g.is_directed() else g

        n = g_undirected.number_of_nodes()
        if n == 0:
            return MetricResult(
                metric=self.INFO.name,
                summary={"n_components": 0, "largest_component_ratio": 0.0, "fiedler": 0.0},
            )

        ccs = list(nx.connected_components(g_undirected))
        n_components = len(ccs)
        largest_cc = max(ccs, key=len)
        largest_ratio = len(largest_cc) / n

        subgraph = g_undirected.subgraph(largest_cc)
        try:
            fiedler = float(nx.algebraic_connectivity(subgraph))
        except Exception:
            fiedler = 0.0

        return MetricResult(
            metric=self.INFO.name,
            summary={
                "n_components": n_components,
                "largest_component_ratio": float(largest_ratio),
                "fiedler": fiedler,
            },
        )
