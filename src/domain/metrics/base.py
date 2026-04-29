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

    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        pass

class AbsoluteMetric(ABC):
    INFO: MetricInfo

    @abstractmethod
    def compute(self, graph: Graph, params: RunParams) -> MetricResult:
        pass

class RelativeMetric(ABC):
    INFO: MetricInfo

    @abstractmethod
    def compute(self, G: Graph, H: Graph, params: RunParams) -> MetricResult:
        pass

class DeltaMetric(ABC):
    '''
    takes any AbsoluteMetric and automatically converts it into a RelativeMetric
    that calculates the delta between the original and sparsified graph
    '''

    def __init__(self, base_metric: AbsoluteMetric):
        self.base_metric = base_metric
        self.INFO = MetricInfo(
            name=f"{base_metric.INFO.name} delta",
            description=f"calculates the change in {base_metric.INFO.name}",
        )

    def compute(self, G: Graph, H: Graph, params: RunParams) -> MetricResult:
        G_result = self.base_metric.compute(G, params).summary
        H_result = self.base_metric.compute(H, params).summary

        delta_summary = {}
        for key in G_result.keys():
            if isinstance(G_result[key], (int, float)) and isinstance(H_result[key], (int, float)):
                delta_summary[f"{key}_original"] = G_result[key]
                delta_summary[f"{key}_reduced"] = H_result[key]
                delta_summary[f"{key}_delta"] = H_result[key] - G_result[key]

        return MetricResult(
            metric=self.INFO.name,
            summary=delta_summary
        )
