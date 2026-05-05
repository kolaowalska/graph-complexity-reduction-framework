from __future__ import annotations

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric


@register_metric("edge_density")
class EdgeDensity(Metric):
    INFO = MetricInfo(
        name="edge density",
        type="absolute",
        description="ratio of present edges to maximum possible edges (2|E| / |V|(|V|-1) for undirected graphs)."
    )

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        g = graph.to_networkx(copy=False)
        n = g.number_of_nodes()
        e = g.number_of_edges()
        max_edges = n * (n - 1) / 2 if not g.is_directed() else n * (n - 1)
        density = float(e / max_edges) if max_edges > 0 else 0.0

        return MetricResult(
            metric=self.INFO.name,
            summary={"density": density, "edges": e, "nodes": n},
        )
