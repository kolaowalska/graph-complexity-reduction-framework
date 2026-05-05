from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import logging

from src.domain.graph_model import Graph, RunParams, OperationDescriptor


@dataclass
class TransformInfo:
    """
    metadata for a GraphTransform plugin
    """
    name: str
    abbrev: str
    version: str = "1.0.0"
    supports_directed: bool = True
    supports_weighted: bool = True
    deterministic: bool = False
    # param_schema: Mapping[str, ParamSpec] = field(default_factory=dict)

    def descriptor(self) -> OperationDescriptor:
        return OperationDescriptor(kind="transform", name=self.name, version=self.version)


class GraphTransform(ABC):
    INFO: TransformInfo

    def execute(self, graph: Graph, params: RunParams) -> Graph:
        if graph.node_count == 0:
            logging.warning(f"[{self.__class__.__name__}] no nodes found")

        print(f"\n[{self.__class__.__name__}] starting transformation on '{graph.name}'")
        start_time = time.time()

        result_graph = self.run(graph, params)
        duration = time.time() - start_time

        result_graph.metadata['algorithm'] = self.__class__.__name__
        result_graph.metadata['execution_time'] = duration
        result_graph.metadata['parent_graph'] = graph.name

        print(f"[{self.__class__.__name__}] finished transformation on '{graph.name}' in {duration:.5f}s")
        return result_graph

    @abstractmethod
    def run(self, graph: Graph, params: RunParams) -> Graph:
        pass