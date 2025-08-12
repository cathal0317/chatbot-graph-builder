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
    edges: List[EdgeDef] = []

    for node_name, node_info in raw.items():
        if not isinstance(node_info, dict):
            node_info = {}

        # Normalize attributes
        stage = node_info.get("stage", "")
        visible = bool(node_info.get("visible", True))

        attrs = dict(node_info)

        if "koName" in attrs:
            attrs["ko_name"] = attrs.pop("koName")
        else:
            attrs.setdefault("ko_name", node_name)

        # remove fields that are promoted to NodeDef fields
        attrs.pop("name", None)
        attrs.pop("stage", None)
        attrs.pop("visible", None)
        prev_nodes = attrs.pop("prev_nodes", [])

        nodes[node_name] = NodeDef(name=node_name, stage=stage, visible=visible, attrs=attrs)

        if prev_nodes:
            for prev in prev_nodes:
                if isinstance(prev, dict):
                    source = prev.get("name")
                    context = prev.get("context")
                else:
                    source = str(prev)
                    context = None
                if not source:
                    continue
                edges.append(EdgeDef(source=source, target=node_name, context=context))

    return GraphDef(nodes=nodes, edges=edges) 