from __future__ import annotations

import time
from typing import Any, Optional, Dict

from src.domain.transforms.registry import TransformRegistry
from src.domain.graph_model import Graph, RunParams
from src.domain.experiment import Experiment
from src.domain.sparsifiers.registry import SparsifierRegistry
from src.domain.metrics.registry import MetricRegistry
from src.domain.metrics.base import MetricResult

from src.infrastructure.graph_gateway import GraphGateway, GraphSource
from src.infrastructure.persistence.repo import GraphRepository, ExperimentRepository
from src.infrastructure.persistence.unit_of_work import UnitOfWork
from src.infrastructure.persistence.stubs import InMemoryExperimentRepository
from src.application.dto import ExperimentDTO


class ExperimentService:
    def __init__(
            self,
            graph_repo: GraphRepository,
            experiment_repo: ExperimentRepository,
            gateway: Optional[GraphGateway] = None):
        self.graph_repo = graph_repo
        self.experiment_repo = experiment_repo or InMemoryExperimentRepository()
        self.gateway = gateway or GraphGateway()

    def import_graph(self, source: GraphSource) -> str:
        """
        imports a graph into the service store and returns an internal handle
        """
        graph = self.gateway.load(source)
        key = graph.name

        # TODO: collision logic
        if self.graph_repo.get(key) is not None:
            i = 2
            while self.graph_repo.get(f"{key}_{i}") is not None:
                i += 1
            key = f"{key}_{i}"
            graph = Graph.from_networkx(
                graph.to_networkx(copy=True),
                name=key,
                metadata=dict(graph.metadata)
            )
        self.graph_repo.save(graph)
        return key

    def get_graph(self, graph_key: str) -> Graph:
        graph = self.graph_repo.get(graph_key)
        if graph is None:
            available = ", ".join(self.graph_repo.list_names())
            raise KeyError(f"graph not found: {graph_key}. available graphs: [{available}]")
        return graph

    def list_graphs(self) -> list[str]:
        return self.graph_repo.list_names()

    def run_sparsifier(
        self,
        graph_key: str,
        sparsifier_name: str,
        params: Dict[str, Any],
    ) -> Graph:
        """
        applies a sparsifier using [LAYER SUPERTYPE] execute method and returns a graph object
        """
        SparsifierRegistry.discover()

        G = self.get_graph(graph_key)
        sparsifier = SparsifierRegistry.get(sparsifier_name)

        # calls execute() (layer supertype) not run()!
        return sparsifier.execute(G, RunParams(params))


    def run_transform(
        self,
        graph_key: str,
        transform_name: str,
        params: Dict[str, Any],
    ) -> Graph:
        """
        applies a transformation using [LAYER SUPERTYPE] execute method and returns a graph object
        """
        TransformRegistry.discover()

        G = self.get_graph(graph_key)
        transform = TransformRegistry.get(transform_name)
        return transform.execute(G, RunParams(params))


    def compute_metrics(
        self,
        graph: Graph,
        metric_names: list[str],
    ) -> list[MetricResult]:
        MetricRegistry.discover()
        results = []

        for name in metric_names:
            metric = MetricRegistry.get(name)
            start = time.perf_counter()
            result = metric.compute(graph, RunParams({}))
            duration = time.perf_counter() - start

            new_summary = dict(result.summary)
            new_summary['execution_time'] = duration

            # results.append(metric.compute(graph, RunParams({})))

            updated_result = MetricResult(
                metric=result.metric,
                summary=new_summary,
                artifacts=result.artifacts
            )
            results.append(updated_result)

        return results


# SERVICE LAYER ORCHESTRATION

    def run_experiment(
        self,
        graph_key: str,
        algorithm_name: str,
        metric_names: list[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> ExperimentDTO:
        """
        uses the [DTO] to orchestrate an experiment within a [UNIT OF WORK]
        """
        # 0. start UOW
        uow = UnitOfWork(self.graph_repo, self.experiment_repo)
        run_params = params or {}

        with uow:
            # 1. discovery
            SparsifierRegistry.discover()
            TransformRegistry.discover()
            start = time.perf_counter()

            # 2. polymorphic execution
            if algorithm_name in SparsifierRegistry.list():
                h = self.run_sparsifier(graph_key, algorithm_name, run_params)
            elif algorithm_name in TransformRegistry.list():
                h = self.run_transform(graph_key, algorithm_name, run_params)
            else:
                all_algos = sorted(SparsifierRegistry.list() + TransformRegistry.list())
                raise KeyError(f"algorithm '{algorithm_name}' not found. available: {all_algos}")

            transform_time = time.perf_counter() - start
            if isinstance(h.metadata, dict):
                h.metadata['execution_time'] = transform_time

            # 3. compute metrics
            metric_results = self.compute_metrics(h, metric_names)

            # 4. create experiment entity (domain object)
            experiment = Experiment()
            experiment.start()
            for m in metric_results:
                experiment.add_result(m.metric, m)
            experiment.finish()

            # 5. register new objects
            uow.register_new_graph(h)
            uow.register_new_experiment(experiment)

            # context manager __exit__ should call uow.commit() automatically here (?)

        # 6. return DTO for UI/console
        original_graph = self.get_graph(graph_key)

        return ExperimentDTO(
            graph_name=graph_key,
            reduced_graph_key=h.name,
            nodes_before=original_graph.node_count,
            edges_before=original_graph.edge_count,
            nodes_after=h.node_count,
            edges_after=h.edge_count,
            algorithm_name=algorithm_name,
            metric_results=metric_results,
            metadata=h.metadata
        )

