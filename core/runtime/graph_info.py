from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import networkx as nx

from graph.graph_builder import GraphBuilder
from graph.validator import validate_graph


@dataclass
class GraphInfo:
    graph: nx.DiGraph
    nodes_info: Dict[str, Dict[str, Any]]
    start_nodes: List[str]
    end_nodes: List[str]


def load_and_validate(json_path: str) -> GraphInfo:
    """Load nodes config from JSON, build a directed graph, validate, and return a graph info bundle."""
    gb = GraphBuilder()
    if not gb.load_from_json(json_path):
        raise ValueError(f"Invalid JSON or failed to load: {json_path}")
    if not gb.build_graph():
        raise ValueError("Graph build/validation failed")

    report = validate_graph(gb.graph)
    # Proceed even with warnings; errors were already considered in build_graph()
    return GraphInfo(
        graph=gb.graph,
        nodes_info=gb.nodes_info,
        start_nodes=report.start_nodes,
        end_nodes=report.end_nodes,
    ) 