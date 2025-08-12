from __future__ import annotations

from typing import Dict, Tuple, List

import networkx as nx


def _try_graphviz_layout(g: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    try:
        # Prefer pygraphviz if available
        from networkx.drawing.nx_agraph import graphviz_layout  # type: ignore
        return graphviz_layout(g, prog="dot")  # top-to-bottom layering for DAGs
    except Exception:
        try:
            # Fallback to pydot if available
            from networkx.drawing.nx_pydot import graphviz_layout  # type: ignore
            return graphviz_layout(g, prog="dot")
        except Exception:
            return {}


def _stage_layered_layout(g: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    # Arrange nodes by their 'stage' attribute in columns (left-to-right)
    stage_order: List[str] = [
        "start",
        "nlu",
        "data_collection",
        "selection",
        "processing",
        "end",
        "error",
    ]
    stage_to_col = {stage: idx for idx, stage in enumerate(stage_order)}

    # Group nodes by stage
    grouped: Dict[int, List[str]] = {}
    for n, a in g.nodes(data=True):
        col = stage_to_col.get(a.get("stage", ""), len(stage_order))
        grouped.setdefault(col, []).append(n)

    # Sort nodes in each column for stable layout
    for col in grouped:
        grouped[col].sort()

    # Compute positions: columns on x, evenly spaced y within each column
    pos: Dict[str, Tuple[float, float]] = {}
    col_gap = 3.0
    row_gap = 1.5
    for col, nodes in sorted(grouped.items()):
        x = col * col_gap
        # Center around 0 vertically
        offset = (len(nodes) - 1) * row_gap / 2.0
        for i, n in enumerate(nodes):
            y = -offset + i * row_gap
            pos[n] = (x, y)
    return pos


def draw_with_legend(g: nx.DiGraph, save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        print("⚠️ matplotlib이 설치되지 않았습니다. 시각화를 건너뜁니다.")
        return

    # Stage → color map
    color_map: Dict[str, str] = {
        "start": "#90EE90",
        "nlu": "#FFB6C1",
        "data_collection": "#87CEEB",
        "selection": "#DDA0DD",
        "processing": "#F0E68C",
        "end": "#FFA07A",
        "error": "#FF6B6B",
    }

    # Layout on full graph to keep structure
    pos = _try_graphviz_layout(g)
    if not pos:
        pos = _stage_layered_layout(g)
    if not pos:
        pos = nx.spring_layout(g, seed=42, k=0.7)

    # Split nodes by visibility
    visible_nodes = [n for n, a in g.nodes(data=True) if a.get("visible", True)]
    invisible_nodes = [n for n in g.nodes() if n not in visible_nodes]

    # Prepare labels and colors
    labels_visible: Dict[str, str] = {}
    colors_visible: List[str] = []
    for n in visible_nodes:
        stage = g.nodes[n].get("stage", "")
        labels_visible[n] = g.nodes[n].get("ko_name") or n
        colors_visible.append(color_map.get(stage, "#D3D3D3"))

    labels_invisible: Dict[str, str] = {n: (g.nodes[n].get("ko_name") or n) for n in invisible_nodes}

    # Draw
    plt.figure(figsize=(16, 10))

    # Invisible nodes (faded but present)
    if invisible_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=invisible_nodes,
            node_color="#D3D3D3",
            alpha=0.5,
            node_size=1800,
            edgecolors="#BBBBBB",
            linewidths=1.5,
        )

    # Visible nodes
    if visible_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=visible_nodes,
            node_color=colors_visible,
            node_size=2000,
            edgecolors="#444444",
            linewidths=2,
        )

    # Edges: style light/dashed if any endpoint is invisible
    edges_visible = []
    edges_faded = []
    for u, v in g.edges():
        if u in visible_nodes and v in visible_nodes:
            edges_visible.append((u, v))
        else:
            edges_faded.append((u, v))

    if edges_faded:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edges_faded,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            width=2.0,
            edge_color="#AAAAAA",
            style="dashed",
            connectionstyle="arc3,rad=0.06",
        )

    if edges_visible:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edges_visible,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22,
            width=2.6,
            edge_color="#555555",
            connectionstyle="arc3,rad=0.06",
        )

    # Labels
    if labels_invisible:
        nx.draw_networkx_labels(
            g,
            pos,
            labels=labels_invisible,
            font_family="AppleGothic",
            font_size=10,
            font_weight="normal",
            font_color="#777777",
        )

    if labels_visible:
        nx.draw_networkx_labels(
            g,
            pos,
            labels=labels_visible,
            font_family="AppleGothic",
            font_size=11,
            font_weight="bold",
            font_color="#111111",
        )

    # Edge labels: draw for visible-visible in dark, faded for others
    edge_labels_dark: Dict[Tuple[str, str], str] = {}
    edge_labels_faded: Dict[Tuple[str, str], str] = {}
    for u, v, a in g.edges(data=True):
        ctx = a.get("context", "")
        if not ctx:
            continue
        if u in visible_nodes and v in visible_nodes:
            edge_labels_dark[(u, v)] = ctx
        else:
            edge_labels_faded[(u, v)] = ctx

    if edge_labels_faded:
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=edge_labels_faded,
            font_size=8.5,
            font_family="AppleGothic",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8,
            ),
            label_pos=0.55,
            font_color="#777777",
        )

    if edge_labels_dark:
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=edge_labels_dark,
            font_size=9,
            font_family="AppleGothic",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
            label_pos=0.55,
            font_color="#111111",
        )

    # Legend
    handles = [
        Patch(facecolor=col, edgecolor="#444444", label=stage)
        for stage, col in color_map.items()
    ]
    plt.legend(
        handles=handles,
        title="Stage",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"✅ 그래프 시각화 완료: {save_path}") 