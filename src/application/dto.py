from dataclasses import dataclass
from typing import Any, Dict, List
from src.domain.metrics.base import MetricResult


# DATA TRANSFER OBJECT

@dataclass(frozen=True)
class ExperimentDTO:
    """
    a container to move experiment data from the service layer to cli/ui without exposing domain entities
    """
    graph_name: str
    reduced_graph_key: str
    nodes_before: int
    edges_before: int
    nodes_after: int
    edges_after: int
    algorithm_name: str
    metric_results: List[MetricResult]
    metadata: Dict[str, Any]

