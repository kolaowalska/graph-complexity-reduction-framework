from __future__ import annotations

import random
import networkx as nx

from domain.transforms.base import TransformInfo
from src.domain.graph_model import Graph, RunParams
from src.domain.sparsifiers.base import Sparsifier
from src.domain.sparsifiers.registry import register_sparsifier


@register_sparsifier("random")
class RandomSparsifier(Sparsifier):
    INFO = TransformInfo(name="random sparsifier", abbrev="rndm")

    def run(self, graph: Graph, params: RunParams) -> Graph:
        p = params.get("p", 0.5)
        seed = params.get("seed", 420)

        rng = random.Random(seed)
        G = graph.to_networkx()

        H = nx.DiGraph() if graph.is_directed() else nx.Graph()
        H.add_nodes_from(G.nodes())

        kept_edges = [
            (u, v, d)
            for u, v, d in G.edges(data=True)
            if rng.random() <= p
        ]

        H.add_edges_from(kept_edges)

        return Graph.from_networkx(
            H,
            name=f"{graph.name}_random",
            metadata={"p": p}
        )