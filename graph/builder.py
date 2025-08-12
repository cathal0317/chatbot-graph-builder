from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from .schema import GraphDef


def build_nx_graph(graph_def: GraphDef) -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()

    # add nodes
    for node_name, node_def in graph_def.nodes.items():
        attrs: Dict[str, Any] = {
            "stage": node_def.stage,
            "visible": node_def.visible,
            **node_def.attrs,
        }
        g.add_node(node_name, **attrs)

    # add edges
    for edge in graph_def.edges:
        g.add_edge(edge.source, edge.target, context=edge.context or "")

    return g 