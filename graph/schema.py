from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeDef:
    name: str
    stage: str = ""
    visible: bool = True
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeDef:
    source: str
    target: str
    context: Optional[str] = None


@dataclass
class GraphDef:
    nodes: Dict[str, NodeDef]
    edges: List[EdgeDef]
    
    
@dataclass
class CycleDetectionResult:
    success: bool
    order: List[str]
    cyclic_nodes: List[str]
    start_nodes: List[str] = field(default_factory=list)
    end_nodes: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_nodes: List[str] = field(default_factory=list)
    end_nodes: List[str] = field(default_factory=list)
    isolated_nodes: List[str] = field(default_factory=list)
    unreachable_nodes: List[str] = field(default_factory=list) 