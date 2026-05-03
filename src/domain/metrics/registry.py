from __future__ import annotations

from typing import Callable, Dict, Type

from .base import Metric, RelativeMetric
from src.domain.common.plugin_discovery import discover_modules

_METRICS: Dict[str, Type[Metric]] = {}
_DISCOVERED = False


class MetricRegistry:
    @staticmethod
    def register(name: str) -> Callable[[Type[Metric | RelativeMetric]], Type[Metric | RelativeMetric]]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric name must be a non-empty string!!")
        key = name.strip()

        def _decorator(cls: Type[Metric]) -> Type[Metric]:
            if key in _METRICS and _METRICS[key] is not cls:
                raise ValueError(f"metric '{key}' already registered by {_METRICS[key].__name__}")
            _METRICS[key] = cls
            return cls

        return _decorator

    @staticmethod
    def discover() -> None:
        global _DISCOVERED
        if _DISCOVERED:
            return
        discover_modules("src.domain.metrics")
        _DISCOVERED = True

    @staticmethod
    def ensure_discovered() -> None:
        if not _DISCOVERED:
            MetricRegistry.discover()

    @staticmethod
    def get(name: str) -> Metric | RelativeMetric:
        MetricRegistry.ensure_discovered()
        key = name.strip()
        try:
            cls = _METRICS[key]
        except KeyError as e:
            available = ", ".join(sorted(_METRICS.keys()))
            raise KeyError(f"unknown metric '{key}'. available: [{available}]") from e
        return cls()

    @staticmethod
    def list() -> list[str]:
        MetricRegistry.ensure_discovered()
        return sorted(_METRICS.keys())


register_metric = MetricRegistry.register
