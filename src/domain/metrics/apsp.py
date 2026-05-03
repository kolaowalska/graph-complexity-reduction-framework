from __future__ import annotations

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric

@register_metric("apsp")
class APSPMetric(Metric):
    INFO = MetricInfo(
        name="all pairs shortest paths",
        type="absolute",
        description="todo"
    )

    # TODO
    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        pass

