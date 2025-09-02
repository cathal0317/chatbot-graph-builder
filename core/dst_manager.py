from typing import Dict, Any
import logging

from .models_simplified import DialogueState, create_dialogue_state
from .executors.factory import executor_factory
from .condition_eval import ConditionEvaluator
from .openai_client import OpenAIClient
from storage.context_store import ContextStore
from .runtime.graph_info import GraphInfo
from .dialog.stage_manager import StageBasedNodeManager, DialogueStage
from .api import build_api_response


logger = logging.getLogger(__name__)

class DSTManager:
    """Dialogue State Tracking Manager (graph-driven)"""
    
    def __init__(self, graph_info: GraphInfo, use_redis: bool = False, start_node: str = None):
        self.runtime = graph_info
        self.nodes_info = graph_info.nodes_info  
        self.executor_factory = executor_factory
        self.condition_evaluator = ConditionEvaluator()
        self.openai_client = OpenAIClient()
        self.context_store = ContextStore(use_redis=use_redis)
        self.stage_manager = StageBasedNodeManager(self.runtime)

        # Start node: prefer graph-reported starts, else first node
        self.start_node = start_node or (graph_info.start_nodes[0] if graph_info.start_nodes else next(iter(graph_info.graph.nodes()), None))
        logger.info(f"DST Manager initialized. Start node: {self.start_node}")

    def start_session(self, session_id: str = None) -> str:
        dialogue_state = DialogueState(session_id=session_id)
        dialogue_state.update_node(self.start_node)
        # Initialize per-node and total turn counters
        dialogue_state.context['node_turns'] = 1
        dialogue_state.context['total_turns'] = 0
        self.context_store.save_state(dialogue_state.session_id, dialogue_state)
        logger.info(f"세션 시작: {dialogue_state.session_id}")
        return dialogue_state.session_id

    def process_turn(self, session_id: str, user_message: str) -> Dict[str, Any]:
        try:
            logger.debug(f"사용자 입력: '{user_message}' (세션: {session_id})")
            
            dialogue_state = self.context_store.load_state(session_id) or DialogueState(session_id=session_id)
            if not dialogue_state.current_node:
                dialogue_state.update_node(self.start_node)
            
            if not isinstance(dialogue_state, DialogueState):
                dialogue_state = create_dialogue_state(session_id)
                dialogue_state.update_node(self.start_node)
            dialogue_state.increment_turn()
            # Update total turns in context
            dialogue_state.context['total_turns'] = dialogue_state.turn_count
            
            current_node = dialogue_state.current_node
            if not current_node or current_node not in self.nodes_info:
                logger.error(f"Invalid current node: {current_node}")
                return self._create_error_response("시스템 오류가 발생했습니다.")
            node_info = self.nodes_info[current_node]

            # Auto-run NLU to capture intent/entities/stage
            self._auto_extract_intent(user_message, node_info, dialogue_state)
            # Log LLM-detected stage and context snapshot
            try:
                detected_stage = dialogue_state.context.get('last_stage')
                logger.debug(f"[Stage] LLM detected stage: {detected_stage} (node={current_node})")
            except Exception:
                pass

            # Select executor: prefer LLM-provided stage, then node config, then fallback
            llm_stage = str(dialogue_state.context.get('last_stage') or '').strip().lower()
            node_stage = llm_stage if llm_stage else str(node_info.get('stage', 'default'))
            if not node_stage or node_stage == 'default':
                try:
                    node_stage = self.stage_manager.get_node_stage(current_node).value
                except Exception:
                    node_stage = 'default'
            logger.debug(f"[Stage] Executor stage selected: {node_stage}")
            executor = self.executor_factory.get(node_stage)
            execution_result = executor.execute(node_info, dialogue_state, user_message, self.openai_client)

            result_dict = execution_result.model_dump() if hasattr(execution_result, 'model_dump') else execution_result
            self._update_dialogue_state(dialogue_state, result_dict)

            # Pre-compute stage intent for routing hints (LLM-first)
            try:
                current_stage = self.stage_manager.get_node_stage(current_node)
                llm_stage_str = str(dialogue_state.context.get('last_stage') or '').strip().lower()
                desired_next_stage = DialogueStage(llm_stage_str) if llm_stage_str else None
                if desired_next_stage is None:
                    desired_next_stage = self.stage_manager.determine_next_stage(current_stage, dialogue_state.context)
            except Exception:
                current_stage = None
                desired_next_stage = None
            # Log routing hints
            try:
                logger.debug(
                    f"[Stage] Routing hints: current={getattr(current_stage, 'value', None)}, "
                    f"llm={dialogue_state.context.get('last_stage')}, "
                    f"desired={getattr(desired_next_stage, 'value', None)}"
                )
            except Exception:
                pass

            # Determine next node:
            # 1) Use executor-provided next_node if present
            # 2) Else use graph successors (prefer successors matching LLM stage, then rule-based desired_next_stage)
            # 3) Else use stage-based transition to pick a node from next stage
            next_node = result_dict.get('next_node')
            if next_node == "STAY_CURRENT":
                next_node = current_node
            
            # Guard: do not advance on off-topic/general chat
            last_intent = dialogue_state.context.get('last_intent')
            last_stage = dialogue_state.context.get('last_stage')
            if str(last_intent).lower() == 'off_topic' or str(last_stage).lower() == 'general_chat':
                next_node = current_node
            
            if not next_node:
                succ = list(self.runtime.graph.successors(current_node))
                if succ:
                    # Prefer successors whose explicit node 'stage' matches LLM last_stage
                    last_stage_str = str(last_stage).lower() if last_stage is not None else ''
                    if last_stage_str:
                        meta_matched = [n for n in succ if str(self.nodes_info.get(n, {}).get('stage') or '').lower() == last_stage_str]
                        if meta_matched:
                            next_node = meta_matched[0]
                    if not next_node and desired_next_stage is not None:
                        # Fallback: prefer successors whose classified stage matches desired_next_stage
                        stage_matched = [n for n in succ if self.stage_manager.get_node_stage(n) == desired_next_stage]
                        next_node = stage_matched[0] if stage_matched else succ[0]
                    if not next_node:
                        next_node = succ[0]
                else:
                    next_node = None
            
            if not next_node:
                # Stage-based fallback path (no explicit successor)
                if current_stage is None:
                    current_stage = self.stage_manager.get_node_stage(current_node)
                if desired_next_stage is None:
                    desired_next_stage = self.stage_manager.determine_next_stage(current_stage, dialogue_state.context)
                picked = self.stage_manager.select_node_from_stage(desired_next_stage, dialogue_state)
                next_node = picked or current_node

            # Update per-node turn counter before potential node change
            try:
                if not next_node or next_node == current_node:
                    dialogue_state.context['node_turns'] = int(dialogue_state.context.get('node_turns', 1)) + 1
                else:
                    dialogue_state.context['node_turns'] = 1
            except Exception:
                dialogue_state.context['node_turns'] = 1

            if next_node and next_node != current_node and next_node in self.nodes_info:
                # Track visited nodes for better routing heuristics
                visited = dialogue_state.context.get('visited_nodes', [])
                if isinstance(visited, list):
                    if current_node not in visited:
                        visited.append(current_node)
                    dialogue_state.context['visited_nodes'] = visited
                dialogue_state.update_node(next_node)
                logger.debug(f"노드 전환: '{current_node}' → '{next_node}'")

            self.context_store.save_state(session_id, dialogue_state)
            
            # Build structured API response via builder
            try:
                successors = list(self.runtime.graph.successors(dialogue_state.current_node))
            except Exception:
                successors = []
            api_resp = build_api_response(
                dialogue_state=dialogue_state,
                response_text=result_dict.get('response', ''),
                start_node=self.start_node,
                node_stage=node_stage,
                successors=successors,
            )

            return {
                'response': result_dict.get('response', '응답을 생성할 수 없습니다.'),
                'session_id': session_id,
                'current_node': dialogue_state.current_node,
                'turn_count': dialogue_state.turn_count,
                'session_complete': dialogue_state.is_complete,
                'slots': dialogue_state.get_filled_slots(),
                'context': dialogue_state.context,
                'data': api_resp.model_dump()
            }
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            return self._create_error_response("처리 중 오류가 발생했습니다.")

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        try:
            dialogue_state = self.context_store.load_state(session_id)
            if not dialogue_state:
                return self._create_error_response("세션을 찾을 수 없습니다.")
            try:
                successors = list(self.runtime.graph.successors(dialogue_state.current_node))
            except Exception:
                successors = []
            try:
                node_stage = self.stage_manager.get_node_stage(dialogue_state.current_node).value
            except Exception:
                node_stage = 'default'
            api_resp = build_api_response(
                dialogue_state=dialogue_state,
                response_text='',
                start_node=self.start_node,
                node_stage=node_stage,
                successors=successors,
            )
            return {
                'response': '',
                'session_id': session_id,
                'current_node': dialogue_state.current_node,
                'turn_count': dialogue_state.turn_count,
                'session_complete': dialogue_state.is_complete,
                'slots': dialogue_state.get_filled_slots(),
                'context': dialogue_state.context,
                'data': api_resp.model_dump()
            }
        except Exception as e:
            logger.error(f"Error loading session info: {e}")
            return self._create_error_response("세션 정보를 불러오지 못했습니다.")

    def _auto_extract_intent(self, user_message: str, node_config: Dict[str, Any], dialogue_state: DialogueState):
        try:
            intent_data = self.openai_client.extract_intent_entities(user_message=user_message, node_config=node_config, context=dialogue_state.context)
            dialogue_state.context.update({
                'last_intent': intent_data.get('intent'),
                'last_entities': intent_data.get('entities', {}),
                'last_confidence': intent_data.get('confidence', 0.0),
                'last_stage': intent_data.get('stage'),
                'last_missing_slots': intent_data.get('missing_slots', []),
                'last_all_slots_filled': intent_data.get('all_slots_filled', False),
            })
        except Exception as e:
            logger.debug(f"Auto intent extraction failed: {e}")

    def _update_dialogue_state(self, dialogue_state: DialogueState, result: Dict[str, Any]):
        if 'context_updates' in result:
            dialogue_state.context.update(result['context_updates'])
        if 'slot_updates' in result:
            for slot_name, slot_data in result['slot_updates'].items():
                if isinstance(slot_data, dict) and 'value' in slot_data:
                    dialogue_state.set_slot(slot_name, slot_data['value'], slot_data.get('confidence', 1.0), slot_data.get('source', 'executor'))
                else:
                    dialogue_state.set_slot(slot_name, slot_data)

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        return {
            'response': message,
            'session_id': None,
            'current_node': None,
            'turn_count': 0,
            'session_complete': False,
            'slots': {},
            'context': {},
            'error': True
        } 