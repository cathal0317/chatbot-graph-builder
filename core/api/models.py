from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SlotItem(BaseModel):
    name: str
    value: Any
    confidence: Optional[float] = None
    source: Optional[str] = None


class ContextSummary(BaseModel):
    last_intent: Optional[str] = None
    last_stage: Optional[str] = None
    last_confidence: Optional[float] = None
    node_turns: Optional[int] = None
    total_turns: Optional[int] = None
    visited_nodes: Optional[List[str]] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class NodeState(BaseModel):
    current: Optional[str] = None
    previous: Optional[str] = None
    stage: Optional[str] = None
    successors: List[str] = Field(default_factory=list)
    start_node: Optional[str] = None


class SessionInfo(BaseModel):
    id: str
    is_complete: bool
    turn_count: int
    started_at: datetime
    last_updated: datetime


class Message(BaseModel):
    text: str


class APIData(BaseModel):
    message: Message
    session: SessionInfo
    node: NodeState
    slots: List[SlotItem]
    context: ContextSummary


class APIResponse(BaseModel):
    data: APIData 