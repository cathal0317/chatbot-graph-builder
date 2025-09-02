from __future__ import annotations

from typing import List, Optional

from .models import APIResponse, APIData, Message, SessionInfo, NodeState, ContextSummary, SlotItem
from ..models_simplified import DialogueState


def build_api_response(
    dialogue_state: DialogueState,
    response_text: str,
    start_node: Optional[str],
    node_stage: Optional[str],
    successors: Optional[List[str]],
) -> APIResponse:
    slots_list = []
    for k, v in dialogue_state.slots.items():
        slots_list.append(SlotItem(name=k, value=v.value, confidence=v.confidence, source=v.source))

    context_summary = ContextSummary(
        last_intent=dialogue_state.context.get('last_intent'),
        last_stage=dialogue_state.context.get('last_stage'),
        last_confidence=dialogue_state.context.get('last_confidence'),
        node_turns=dialogue_state.context.get('node_turns'),
        total_turns=dialogue_state.context.get('total_turns'),
        visited_nodes=dialogue_state.context.get('visited_nodes'),
        raw=dialogue_state.context
    )

    node_state = NodeState(
        current=dialogue_state.current_node,
        previous=dialogue_state.previous_node,
        stage=node_stage,
        successors=successors or [],
        start_node=start_node,
    )

    session_info = SessionInfo(
        id=dialogue_state.session_id,
        is_complete=dialogue_state.is_complete,
        turn_count=dialogue_state.turn_count,
        started_at=dialogue_state.started_at,
        last_updated=dialogue_state.last_updated
    )

    return APIResponse(
        data=APIData(
            message=Message(text=response_text or ""),
            session=session_info,
            node=node_state,
            slots=slots_list,
            context=context_summary
        )
    ) 