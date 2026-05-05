from __future__ import annotations

from typing import Dict, Type, Callable, Iterable

from .base import Sparsifier
from src.domain.common.plugin_discovery import discover_modules

_SPARSIFIERS: Dict[str, Type[Sparsifier]] = {}
_DISCOVERED = False


class SparsifierRegistry:
    @staticmethod
    def register(name: str) -> Callable[[Type[Sparsifier]], Type[Sparsifier]]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("sparsifier name must be a non-empty string!!")

        key = name.strip()

        def _decorator(cls: Type[Sparsifier]) -> Type[Sparsifier]:
            if key in _SPARSIFIERS and _SPARSIFIERS[key] is not cls:
                raise ValueError(f"sparsifier '{key}' already registered by {_SPARSIFIERS[key].__name__}")
            _SPARSIFIERS[key] = cls
            return cls

        return _decorator

    @staticmethod
    def discover() -> None:
        """
        imports all modules under graph_pipeline.domain.sparsifiers
        each plugin module should self-register using @SparsifierRegistry.register("name").
        """
        global _DISCOVERED
        if _DISCOVERED:
            return
        discover_modules("src.domain.sparsifiers")
        _DISCOVERED = True

    @staticmethod
    def ensure_discovered() -> None:
        if not _DISCOVERED:
            SparsifierRegistry.discover()

    @staticmethod
    def get(name: str) -> Sparsifier:
        SparsifierRegistry.ensure_discovered()
        key = name.strip()
        try:
            cls = _SPARSIFIERS[key]
        except KeyError as e:
            available = ", ".join(sorted(_SPARSIFIERS.keys()))
            raise KeyError(f"unknown sparsifier '{key}'. available: [{available}]") from e
        return cls()

    @staticmethod
    def list() -> list[str]:
        SparsifierRegistry.ensure_discovered()
        return sorted(_SPARSIFIERS.keys())

    @staticmethod
    def items() -> Iterable[tuple[str, Type[Sparsifier]]]:
        SparsifierRegistry.ensure_discovered()
        return _SPARSIFIERS.items()

register_sparsifier = SparsifierRegistry.register
