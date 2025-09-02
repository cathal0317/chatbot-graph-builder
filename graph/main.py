from __future__ import annotations

import os
from graph.graph_builder import GraphBuilder


def main() -> None:
    print("=" * 60)
    print("GraphBuilder 테스트")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', 'card_issuance_chatbot.json')

    builder = GraphBuilder()

    # JSON 설정 로드
    if not builder.load_from_json(config_path):
        return

    # 그래프 생성
    if not builder.build_graph():
        return

    # 사이클 탐지
    cycle_result = builder.detect_cycles()

    # 그래프 정보 출력
    graph_info = builder.export_graph_info()
    print(f"\n총 노드 수: {graph_info['graph_stats']['nodes']}")
    print(f"총 엣지 수: {graph_info['graph_stats']['edges']}")
    print(f"DAG 여부: {graph_info['graph_stats']['is_dag']}")


if __name__ == "__main__":
    main() 