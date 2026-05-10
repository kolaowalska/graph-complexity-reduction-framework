from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import networkx as nx
import os
from pathlib import Path

from src.domain.graph_model import Graph


@dataclass
class GraphSource:
    """
    [DTO] specifying where to find a graph and how it should be interpreted
    """
    kind: str
    name: str
    value: Any = None
    directed: bool = False
    weighted: bool = False
    # fmt: Optional[str] = "edgelist"


class GraphGateway:
    """
    [GATEWAY] to external graph data
    """
    def load(self, source: GraphSource) -> Graph:
        print(f"\n[GATEWAY] loading graph '{source.name}' from {source.kind}...")

        if source.kind == "file":
            path = source.value

            if path is None:
                raise ValueError(f"error: source path is None for graph '{source.name}'")

            if not isinstance(path, (str, os.PathLike)):
                raise TypeError(f"error: expected path string, got {type(path)}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"file not found: {path}")

            def lazy_loader():
                print(f"\n[LAZY LOAD] reading file {path}")

                create_using = nx.DiGraph if source.directed else nx.Graph

                if source.weighted:
                    return nx.read_edgelist(
                        str(path),
                        nodetype=int,
                        create_using=create_using,
                        data=(('weight', float),)
                    )
                else:
                    return nx.read_edgelist(
                        str(path),
                        nodetype=int,
                        create_using=create_using,
                        data=False
                    )

            return Graph.from_loader(name=source.name, loader_f=lazy_loader)

        elif source.kind == "memory":
            if source.value is None:
                return nx.DiGraph() if source.directed else nx.Graph()

            # TODO: strict conversion (source.value.to_directed()) if source.directed else ...
            return Graph.from_networkx(source.value, name=source.name)

        else:
            raise ValueError(f"unknown source kind: {source.kind}")
