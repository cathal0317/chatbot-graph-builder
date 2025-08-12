from __future__ import annotations

from typing import List

import networkx as nx

from .schema import ValidationReport
from .toposort import kahn_toposort


def validate_graph(g: nx.DiGraph) -> ValidationReport:
    warnings: List[str] = []
    errors: List[str] = []

    topos = kahn_toposort(g)
    start_nodes = topos.start_nodes
    end_nodes = topos.end_nodes

    if not start_nodes:
        errors.append("시작 노드(in_degree=0)가 없습니다.")
    
    if not end_nodes:
        warnings.append("종료 노드(out_degree=0)가 없습니다.")

    isolated = [n for n in g.nodes() if g.in_degree(n) == 0 and g.out_degree(n) == 0]
    if isolated:
        warnings.append(f"고립 노드: {sorted(isolated)}")

    # reachability from the (single) start if present
    unreachable: List[str] = []
    if start_nodes:
        start = start_nodes[0]
        reachable = _reachable_from(g, start)
        unreachable = [n for n in g.nodes() if n not in reachable]
        if unreachable:
            warnings.append(f"시작 '{start}'에서 도달 불가: {sorted(unreachable)}")

    ok = len(errors) == 0
    return ValidationReport(
        ok=ok,
        warnings=warnings,
        errors=errors,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        isolated_nodes=isolated,
        unreachable_nodes=unreachable,
    )


def _reachable_from(g: nx.DiGraph, start: str) -> List[str]:
    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for nb in g.successors(cur):
            if nb not in visited:
                stack.append(nb)
    return list(visited) 

    