from __future__ import annotations

from src.domain.graph_model import Graph, RunParams
from src.domain.metrics.base import Metric, MetricInfo, MetricResult
from src.domain.metrics.registry import register_metric


@register_metric("avg_stretch")
class AvgStretch(Metric):
    INFO = MetricInfo(
        name="average stretch",
        type="absolute",
        description="todo"
    )

    # TODO
    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        G = graph.to_networkx(copy=False)

        return MetricResult(
            metric=self.INFO.name,
            summary={
            }
        )
