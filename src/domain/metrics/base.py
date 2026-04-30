from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.domain.graph_model import Graph, RunParams, ArtifactHandle


@dataclass(frozen=True)
class MetricInfo:
    name: str
    version: str = "0.1.0"
    description: str = ""

@dataclass(frozen=True)
class MetricResult:
    """results returned by Metric.compute()"""
    metric: str
    summary: Mapping[str, float | int | str] = field(default_factory=dict)
    artifacts: Mapping[str, ArtifactHandle] = field(default_factory=dict)


class Metric(ABC):
    INFO: MetricInfo

    @abstractmethod
    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        pass


class RelativeMetric(ABC):
    INFO: MetricInfo

    @abstractmethod
    def compute(self, G: Graph, H: Graph, params: RunParams) -> MetricResult:
        pass

class DeltaMetric(Metric):
    """
    decorator that wraps any Metric and computes the delta
    between the original and reduced graph
    """
    INFO: MetricInfo

    def __init__(self, base_metric: Metric):
        self.base_metric = base_metric
        self.INFO = MetricInfo(
            name=f"{base_metric.INFO.name}_delta",
            description=f"delta of {base_metric.INFO.name} between two graphs",
        )

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        """
        single-graph call - delegates to the wrapped metric
        """
        return self.base_metric.compute(graph, params)

    def compute_delta(self, g: Graph, h: Graph, params: RunParams) -> MetricResult:
        """
        two-graph call - computes the before and after diff
        """
        g_summary = self.base_metric.compute(g, params).summary
        h_summary = self.base_metric.compute(h, params).summary

        delta_summary = {}
        for key in g_summary.keys():
            g_val = g_summary[key]
            h_val = h_summary[key]

            if isinstance(g_val, (int, float)) and isinstance(h_val, (int, float)):
                delta_summary[f"{key}_original"] = g_val
                delta_summary[f"{key}_reduced"] = h_val
                delta_summary[f"{key}_delta"] = h_val - g_val

        return MetricResult(
            metric=self.INFO.name,
            summary=delta_summary
        )
