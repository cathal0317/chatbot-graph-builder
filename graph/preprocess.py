from __future__ import annotations

import json
from typing import Any, Dict, List

from .schema import EdgeDef, GraphDef, NodeDef


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_raw_to_graphdef(raw: Dict[str, Any]) -> GraphDef:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("JSON 파일이 비어있거나 딕셔너리 형태가 아닙니다.")

    nodes: Dict[str, NodeDef] = {}
    edges: list[EdgeDef] = []

    for node_name, node_cfg in raw.items():
        # Build node definition
        attrs = {
            k: v for k, v in node_cfg.items()
            if k not in ("next_nodes")
        }
        nodes[node_name] = NodeDef(name=node_name, attrs=attrs)

        # Edges: source=node_name -> target=each next
        next_nodes = node_cfg.get("next_nodes", [])
        if not next_nodes:
            continue

        for nxt in next_nodes:
            if isinstance(nxt, dict):
                target = nxt.get("name")
                context = nxt.get("context")
            else:
                target = str(nxt)
                context = None

            if not target or target == node_name:
                continue  # 빈 타겟/자기참조 방지

            edges.append(EdgeDef(source=node_name, target=target, context=context))

    return GraphDef(nodes=nodes, edges=edges)