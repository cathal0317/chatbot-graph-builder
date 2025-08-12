from __future__ import annotations

import json
from typing import Any, Dict, List

import networkx as nx

from .preprocess import load_json, normalize_raw_to_graphdef
from .builder import build_nx_graph
from .validator import validate_graph
from .toposort import kahn_toposort
from .visualize import draw_with_legend


class GraphBuilder:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes_info: Dict[str, Dict[str, Any]] = {}
        
    def load_from_json(self, json_path: str) -> bool:
        try:
            raw_config = load_json(json_path)
            if not isinstance(raw_config, dict) or not raw_config:
                print("❌ JSON 파일이 비어있거나 딕셔너리 형태로 입력되지 않았습니다.")
                return False

            self.nodes_info = raw_config

            print(f"✅ JSON 설정 로드 완료: {json_path}")
            print(f" 로드된 노드 수: {len(self.nodes_info)}")
            return True

        except FileNotFoundError:
            print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            return False
        except Exception as e:
            print(f"❌ 설정 로드 중 알 수 없는 오류: {e}")
            return False

    def build_graph(self) -> bool:
        if not self.nodes_info:
            print("❌ JSON 파일이 로드되지 않았습니다.")
            return False

        self.graph.clear()

        graph_def = normalize_raw_to_graphdef(self.nodes_info)
        self.graph = build_nx_graph(graph_def)

        print(f"{self.graph.number_of_nodes()}개 노드 추가 완료")
        print("✅ 그래프 생성 완료!")
        print(f"노드 수: {self.graph.number_of_nodes()}")
        print(f"엣지 수: {self.graph.number_of_edges()}")

        report = validate_graph(self.graph)
        for w in report.warnings:
            print(f"{w}")
        for e in report.errors:
            print(f"{e}")
        return report.ok

    def detect_cycles(self) -> Dict[str, Any]:
        result = kahn_toposort(self.graph)
        if result.success:
            print("\n✅ 순환 없음 - 그래프가 DAG입니다.")
            if result.order:
                print("위상 정렬 순서: " + " -> ".join(result.order))
        else:
            print("\n❌ 순환이 감지되었습니다. DAG가 아닙니다.")
            if result.cyclic_nodes:
                print(f"순환 관련 노드(완전히 정렬되지 않음): {sorted(result.cyclic_nodes)}")
        return {
            "success": result.success,
            "order": result.order,
            "cyclic_nodes": result.cyclic_nodes,
        }

    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        if node_name not in self.graph:
            print(f"⚠️ 노드를 찾을 수 없습니다: {node_name}")
            return {}
        original = self.nodes_info.get(node_name, {})
        attrs = self.graph.nodes[node_name]
        merged = {**original, **attrs}
        return merged


    def export_graph_info(self) -> Dict[str, Any]:
        nodes_payload = []
        for node in self.graph.nodes():
            attrs = dict(self.graph.nodes[node])
            nodes_payload.append({"name": node, **attrs})

        edges_payload = []
        for u, v, attrs in self.graph.edges(data=True):
            edges_payload.append({"from": u, "to": v, **attrs})

        cycle_result = kahn_toposort(self.graph)
        graph_stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_dag": cycle_result.success,
        }

        stage_groups = {}
        for node_name, node_info in self.nodes_info.items():
            stage = node_info.get("stage", "unknown")
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append(node_name)

        return {
            "nodes": nodes_payload,
            "edges": edges_payload,
            "graph_stats": graph_stats,
            "stage_groups": stage_groups,
        }

    def visualize_graph(self, save_path: str) -> None:
        draw_with_legend(self.graph, save_path) 


    def get_predecessors(self, node_name: str) -> List[str]:
        if node_name not in self.graph:
            return []
        return list(self.graph.predecessors(node_name))

    def get_successors(self, node_name: str) -> List[str]:
        if node_name not in self.graph:
            return []
        return list(self.graph.successors(node_name))