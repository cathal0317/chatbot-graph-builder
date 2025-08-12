from __future__ import annotations

from collections import deque
from typing import Dict, List

import networkx as nx

from .schema import CycleDetectionResult


def kahn_toposort(g: nx.DiGraph) -> CycleDetectionResult:
    in_deg: Dict[str, int] = dict(g.in_degree())
    out_deg: Dict[str, int] = dict(g.out_degree())

    starts: List[str] = [n for n, d in in_deg.items() if d == 0 and g.out_degree(n) > 0]
    ends: List[str] = [n for n, d in out_deg.items() if d == 0 and g.in_degree(n) > 0]

    q: deque[str] = deque(n for n, d in in_deg.items() if d == 0)
    order: List[str] = []
    local = dict(in_deg)

    while q:
        cur = q.popleft()
        order.append(cur)
        for nb in g.successors(cur):
            local[nb] -= 1
            if local[nb] == 0:
                q.append(nb)

    success = len(order) == len(g.nodes())
    cyclic_nodes = [] if success else [n for n, d in local.items() if d > 0]
    return CycleDetectionResult(
        success=success,
        order=order,
        cyclic_nodes=cyclic_nodes,
        start_nodes=starts,
        end_nodes=ends,
    ) 