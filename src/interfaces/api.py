from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any

from src.application.experiment_service import ExperimentService
from src.infrastructure.graph_gateway import GraphSource
from src.infrastructure.persistence.stubs import InMemoryGraphRepository, InMemoryExperimentRepository


class ExperimentFacade:
    """
    [REMOTE FACADE] provides a coarse interface for interacting with experiments
    """
    def __init__(self):
        # in a real app this would be injected
        self.graph_repo = InMemoryGraphRepository()
        self.experiment_repo = InMemoryExperimentRepository()
        self._service = ExperimentService(self.graph_repo, self.experiment_repo)

    def upload_graph(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        simulates POST /graphs/upload
        input: {"path": "...", "name": "..."}
        output: {"status": "ok", "id": "..."}
        """
        source = GraphSource(
            kind=request_json.get("kind", "file"),
            value=request_json["path"],
            name=request_json["name"],
            directed = request_json.get("directed", False),
            weighted = request_json.get("weighted", False)
        )
        try:
            key = self._service.import_graph(source)
            return {"status": "success", "graph_key": key}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_job(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        simulates POST /jobs/run
        input: {"graph_key": "...", "algorithm": "...", "metrics": [...], "params": {...}}
        """
        try:
            # delegating to [SERVICE LAYER]
            dto = self._service.run_experiment(
                graph_key=request_json["graph_key"],
                algorithm_name=request_json["algorithm"],
                metric_names=request_json.get("metrics", []),
                params=request_json.get("params", {})
            )

            # serializes [DTO] to a JSON-compatible dict
            return {
                "status": "success",
                "data": asdict(dto)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def service(self):
        return self._service