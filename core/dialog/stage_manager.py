from __future__ import annotations

from enum import Enum
from typing import Dict, Any, List, Optional, Set
import re
from ..runtime.graph_info import GraphInfo


class DialogueStage(Enum):
    GREETINGS = "greetings"          # welcome, 환영, greeting
    SLOT_FILLING = "slot_filling"    # collect_, input, 수집, verification, select
    CONFIRMATION = "confirmation"    # confirm, 확인, 최종 확인
    COMPLETION = "completion"        # completion_, process_, approved, rejected
    GOODBYE = "goodbye"               # 대화 종료 단계
    GENERAL_CHAT = "general_chat"    # 기타 일반 대화


class StageBasedNodeManager:
    """Stage-based classifier and router that works on top of GraphInfo.

    - Classifies nodes into DialogueStage groups using existing JSON fields
    - Provides stage transition rules and node selection per stage
    - Designed to be non-invasive and compatible with current DSTManager flow
    """

    STAGE_TRANSITIONS: Dict[DialogueStage, List[DialogueStage]] = {
        DialogueStage.GREETINGS: [DialogueStage.SLOT_FILLING, DialogueStage.GENERAL_CHAT],
        DialogueStage.SLOT_FILLING: [DialogueStage.SLOT_FILLING, DialogueStage.CONFIRMATION, DialogueStage.GENERAL_CHAT],
        DialogueStage.CONFIRMATION: [DialogueStage.SLOT_FILLING, DialogueStage.COMPLETION, DialogueStage.GENERAL_CHAT],
        DialogueStage.COMPLETION: [],
        DialogueStage.GENERAL_CHAT: [DialogueStage.SLOT_FILLING, DialogueStage.CONFIRMATION, DialogueStage.COMPLETION],
        DialogueStage.GOODBYE: [],
    }

    def __init__(self, graph_info: GraphInfo):
        self.graph_info: GraphInfo = graph_info
        self.nodes_info: Dict[str, Any] = graph_info.nodes_info
        self.node_to_stage: Dict[str, DialogueStage] = {}
        self.stage_groups: Dict[DialogueStage, List[str]] = self._classify_all_nodes()

    # ------------------------------
    # Public APIs
    # ------------------------------
    def classify_all_nodes(self) -> Dict[DialogueStage, List[str]]:
        return self.stage_groups

    def get_node_stage(self, node_id: str) -> DialogueStage:
        return self.node_to_stage.get(node_id, DialogueStage.GENERAL_CHAT)

    def determine_next_stage(self, current_stage: DialogueStage, intent_context: Dict[str, Any]) -> DialogueStage:
        """Decide next stage using context captured in DST (LLM NLU results in dialogue_state.context)."""
        intent = str(intent_context.get("last_intent") or intent_context.get("intent") or "").lower()
        all_filled = bool(intent_context.get("last_all_slots_filled", False))
        user_msg = str(intent_context.get("last_user_message") or "").lower()

        if current_stage == DialogueStage.GREETINGS:
            return DialogueStage.SLOT_FILLING
        if current_stage == DialogueStage.SLOT_FILLING:
            if all_filled or (intent.startswith("confirm") or "최종" in intent or "동의" in intent):
                return DialogueStage.CONFIRMATION
            return DialogueStage.SLOT_FILLING
        if current_stage == DialogueStage.CONFIRMATION:
            yes_tokens = ["yes", "y", "네", "확인", "맞아", "그래", "ok", "확정", "동의"]
            no_tokens = ["no", "n", "아니오", "아니", "취소", "거부", "반대"]
            if any(tok in user_msg for tok in yes_tokens) or intent in ("confirm", "yes"):
                return DialogueStage.COMPLETION
            if any(tok in user_msg for tok in no_tokens) or intent in ("cancel", "deny", "no"):
                return DialogueStage.SLOT_FILLING
            return DialogueStage.CONFIRMATION
        if current_stage == DialogueStage.GENERAL_CHAT:
            if all_filled:
                return DialogueStage.CONFIRMATION
            return DialogueStage.SLOT_FILLING
        return DialogueStage.COMPLETION if current_stage == DialogueStage.COMPLETION else DialogueStage.GENERAL_CHAT

    def select_node_from_stage(self, stage: DialogueStage, dialogue_state: Any, graph: Any = None) -> Optional[str]:
        candidates = self.stage_groups.get(stage, [])
        if not candidates:
            return None

        current_node = getattr(dialogue_state, "current_node", None)
        visited: List[str] = list(dialogue_state.context.get("visited_nodes", [])) if hasattr(dialogue_state, "context") else []
        visited_set: Set[str] = set(visited)

        g = graph if graph is not None else self.graph_info.graph
        if g is not None and current_node is not None and current_node in g:
            try:
                successors = list(g.successors(current_node))
                for nxt in successors:
                    if nxt in candidates:
                        return nxt
            except Exception:
                pass
        for nid in candidates:
            if nid not in visited_set:
                return nid
        return candidates[0]

    def get_stage_transitions(self) -> Dict[DialogueStage, List[DialogueStage]]:
        return self.STAGE_TRANSITIONS

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _classify_all_nodes(self) -> Dict[DialogueStage, List[str]]:
        groups: Dict[DialogueStage, List[str]] = {
            DialogueStage.GREETINGS: [],
            DialogueStage.SLOT_FILLING: [],
            DialogueStage.CONFIRMATION: [],
            DialogueStage.COMPLETION: [],
            DialogueStage.GENERAL_CHAT: [],
        }
        for node_id, cfg in self.nodes_info.items():
            stage = self._classify_node_to_stage(node_id, cfg)
            self.node_to_stage[node_id] = stage
            groups[stage].append(node_id)
        return groups

    def _classify_node_to_stage(self, node_id: str, node_info: Dict[str, Any]) -> DialogueStage:
        params = node_info.get("params", {}) or {}
        if isinstance(params, dict):
            req = params.get("required_slots")
            if isinstance(req, list) and len(req) > 0:
                return DialogueStage.SLOT_FILLING

        text_parts: List[str] = []
        for key in ("name", "description", "ko_name"):
            val = node_info.get(key)
            if isinstance(val, str) and val:
                text_parts.append(val.lower())
        responses = node_info.get("responses", {}) or {}
        if isinstance(responses, dict):
            for k, v in responses.items():
                if isinstance(k, str):
                    text_parts.append(k.lower())
                if isinstance(v, str):
                    text_parts.append(v.lower())
        actions = node_info.get("actions", []) or []
        if isinstance(actions, list):
            for act in actions:
                if isinstance(act, str):
                    text_parts.append(act.lower())
        for nxt in node_info.get("next_nodes", []) or []:
            if isinstance(nxt, str):
                text_parts.append(nxt.lower())
            elif isinstance(nxt, dict):
                if isinstance(nxt.get("name"), str):
                    text_parts.append(nxt["name"].lower())
                if isinstance(nxt.get("context"), str):
                    text_parts.append(nxt["context"].lower())

        hay = "\n".join(text_parts)

        greet_patterns = [r"\bwelcome\b", r"\bgreet", "환영", "인사", "소개", "start"]
        slot_patterns = [r"\bcollect\b", r"\binput\b", r"\bselect\b", "수집", "입력", "선택", "verification", "신청", "정보"]
        confirm_patterns = [r"\bconfirm\b", "최종", "동의", "terms", "약관", "final_confirm"]
        completion_patterns = [r"\bcompletion\b", r"\bcomplete\b", r"\bprocess\b", "승인", "거절", "완료"]

        if self._any_match(hay, greet_patterns) or node_id.startswith("welcome"):
            return DialogueStage.GREETINGS
        if self._any_match(hay, completion_patterns) or node_id.startswith("completion_") or node_id.startswith("process_"):
            return DialogueStage.COMPLETION
        if node_id.startswith("final_") or node_id.startswith("confirm") or self._any_match(hay, confirm_patterns):
            return DialogueStage.CONFIRMATION
        if self._any_match(hay, slot_patterns):
            return DialogueStage.SLOT_FILLING
        return DialogueStage.GENERAL_CHAT

    @staticmethod
    def _any_match(text: str, patterns: List[str]) -> bool:
        for p in patterns:
            try:
                if p.startswith("\\b") or "[" in p or "(" in p:
                    if re.search(p, text):
                        return True
                elif p in text:
                    return True
            except Exception:
                if p in text:
                    return True
        return False 