# Chatbot Graph Builder

JSON 설정의 `prev_nodes`만으로 대화 그래프의 엣지를 자동 생성하는 Python 시스템입니다. NetworkX `DiGraph`로 구성되며 Kahn's 알고리즘으로 사이클을 탐지하고, 유효성 검증 및 시각화를 제공합니다.

## 요구 사항
- Python 3.9+
- networkx>=3.0
- matplotlib>=3.5.0

## 설치
```bash
cd chatbot-graph-builder
python3 -m pip install -r requirements.txt
```

## 실행
```bash
python3 main.py
```
- `dialog_graph.png` 파일로 그래프 시각화가 저장됩니다.

## 주요 기능
- JSON 설정 로드 (`config/chatbot_config.json`)
- `prev_nodes` 기반 자동 엣지 생성 (수동 `next_nodes` 연결 불필요)
- 방향 그래프(DiGraph) 구성 및 검증
- Kahn's 알고리즘을 통한 사이클 탐지 및 위상 정렬
- 시작/종료 노드 검증, 고립 노드 탐지, 도달 가능성 확인
- 그래프 시각화 (스테이지별 색상 매핑)
