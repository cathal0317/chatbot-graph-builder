from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime

from .models_simplified import DialogueState, create_dialogue_state
from .unified_executor import executor_factory
from .condition_eval import ConditionEvaluator
from .openai_client import OpenAIClient
from .stage_detector import OptimizedStageManager
from storage.context_store import ContextStore

# 그래프 유효성 검증 
from graph.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

class DSTManager:
    """Dialogue State Tracking Manager"""
    
    def __init__(self, nodes_config: Dict[str, Any], 
                 use_redis: bool = False,
                 start_node: str = None,
                 enable_llm_stage_detection: bool = True):
        self.nodes_config = nodes_config
        self.executor_factory = executor_factory  
        self.condition_evaluator = ConditionEvaluator()
        self.openai_client = OpenAIClient()
        self.context_store = ContextStore(use_redis=use_redis)
        
        # Optimized Stage detection system 
        self.stage_manager = OptimizedStageManager(
            enable_detection=enable_llm_stage_detection
        )
        
        # Optional graph system for enhanced features
        self.graph_builder = self._try_initialize_graph(nodes_config)
        
        # 시작 노드 찾기
        self.start_node = start_node or self._find_start_node()
        
        logger.info(f"DST Manager initialized with {len(nodes_config)} nodes")
        logger.info(f"Start node: {self.start_node}")
        logger.info(f"Stage detection: {'enabled' if enable_llm_stage_detection else 'disabled'}")
        
        # Pre-warm stage cache for better performance
        if enable_llm_stage_detection:
            self.stage_manager.warm_cache(nodes_config)
    
    def _get_node_stage(self, node_config: Dict[str, Any]) -> str:
        """Rule-based stage detection with caching"""
        try:
            stage = self.stage_manager.get_node_stage(node_config)
            return stage
        except Exception as e:
            logger.warning(f"Stage detection failed, using fallback: {e}")
            return node_config.get('stage', 'default')
    
    def _try_initialize_graph(self, nodes_config: Dict[str, Any]) -> Optional[GraphBuilder]:
        """Try to initialize graph system (non-critical)"""
        try:
            graph_builder = GraphBuilder()
            graph_builder.nodes_info = nodes_config
            if graph_builder.build_graph():
                logger.debug("Graph system available for enhanced features")
                return graph_builder
        except Exception as e:
            logger.debug(f"Graph system not available: {e}")
        return None
    
    def _find_start_node(self) -> str:
        """Find starting node"""
        # Try graph builder
        if self.graph_builder:
            try:
                cycle_result = self.graph_builder.detect_cycles()
                if cycle_result.get('start_nodes'):
                    return cycle_result['start_nodes'][0]
            except Exception:
                pass
        
        # Fallback to simple logic
        for name, config in self.nodes_config.items():
            stage = config.get('stage', '').lower()
            if stage in ['initial', 'greeting', 'start']:
                return name
            if not config.get('prev_nodes'):
                return name
        
        # Last resort: first node
        return list(self.nodes_config.keys())[0] if self.nodes_config else None
    
    def start_session(self, session_id: str = None) -> str:
        """Start new session"""
        dialogue_state = DialogueState(session_id=session_id)
        dialogue_state.update_node(self.start_node)
        
        self.context_store.save_state(dialogue_state.session_id, dialogue_state)
        logger.info(f"세션 시작: {dialogue_state.session_id}")
        logger.debug(f"시작 노드: '{self.start_node}'")
        return dialogue_state.session_id
    
    def process_turn(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process a dialogue turn"""
        try:
            logger.debug(f"사용자 입력: '{user_message}' (세션: {session_id})")
            
            # Load or create state
            dialogue_state = self.context_store.load_state(session_id)
            if not dialogue_state:
                dialogue_state = DialogueState(session_id=session_id)
                dialogue_state.update_node(self.start_node)
            
            # Ensure we have a valid DialogueState (already should be)
            if not isinstance(dialogue_state, DialogueState):
                # Fallback for legacy states - create new one
                dialogue_state = create_dialogue_state(session_id)
                dialogue_state.update_node(self.start_node)
            dialogue_state.increment_turn()
            
            current_node = dialogue_state.current_node
            if not current_node or current_node not in self.nodes_config:
                logger.error(f"Invalid current node: {current_node}")
                return self._create_error_response("시스템 오류가 발생했습니다.")
            
            node_config = self.nodes_config[current_node]
            
            # Auto-run NLU for context
            self._auto_extract_intent(user_message, node_config, dialogue_state)
            
            # Execute node with enhanced stage detection
            stage = self._get_node_stage(node_config)
            
            # 스테이지 정보 디버그 로그
            logger.debug(f"노드 '{current_node}' 실행 - 감지된 스테이지: '{stage}'")
            
            executor = self.executor_factory.get(stage)
            logger.debug(f"사용된 Executor: {executor.__class__.__name__}")
            
            execution_result = executor.execute(
                node_config, dialogue_state, user_message, self.openai_client
            )
            
            # Handle result (both dict and ExecutionResult formats)
            if hasattr(execution_result, 'model_dump'):  # Pydantic model
                result_dict = execution_result.model_dump()
            else:
                result_dict = execution_result
            
            # Update dialogue state
            self._update_dialogue_state(dialogue_state, result_dict)
            
            # Determine next node
            next_node = result_dict.get('next_node')
            
            # Handle special control values
            if next_node == "STAY_CURRENT":
                logger.debug(f"Executor가 현재 노드 유지 요청: '{current_node}'")
                next_node = current_node
            elif not next_node:
                next_node = self._determine_next_node(node_config, dialogue_state)
            
            # Update node if needed
            if next_node and next_node != current_node and next_node in self.nodes_config:
                dialogue_state.update_node(next_node)
                logger.debug(f"노드 전환: '{current_node}' → '{next_node}'")
            elif next_node == current_node:
                logger.debug(f"현재 노드 유지: '{current_node}'")
            elif not next_node:
                logger.debug(f"다음 노드 없음, '{current_node}'에서 대기")
            
            # Check completion conditions
            if (node_config.get('stage') in ['final', 'end', 'completed'] or 
                result_dict.get('context_updates', {}).get('session_ended')):
                dialogue_state.set_complete(True)
                end_reason = result_dict.get('context_updates', {}).get('end_reason', 'normal_completion')
                logger.info(f"세션 완료: {session_id} (이유: {end_reason})")
            
            # Save state
            self.context_store.save_state(session_id, dialogue_state)
            
            # Create response
            return {
                'response': result_dict.get('response', '응답을 생성할 수 없습니다.'),
                'session_id': session_id,
                'current_node': dialogue_state.current_node,
                'turn_count': dialogue_state.turn_count,
                'session_complete': dialogue_state.is_complete,
                'slots': dialogue_state.get_filled_slots(),
                'context': dialogue_state.context
            }
            
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            return self._create_error_response("처리 중 오류가 발생했습니다.")
    
    def _auto_extract_intent(self, user_message: str, node_config: Dict[str, Any], 
                           dialogue_state: DialogueState):
        """Auto-extract intent for context (non-critical)"""
        try:
            intent_data = self.openai_client.extract_intent_entities(
                user_message=user_message, node_config=node_config, 
                context=dialogue_state.context
            )
            dialogue_state.context.update({
                'last_intent': intent_data.get('intent'),
                'last_entities': intent_data.get('entities', {}),
                'last_confidence': intent_data.get('confidence', 0.0)
            })
        except Exception as e:
            logger.debug(f"Auto intent extraction failed: {e}")
    
    def _update_dialogue_state(self, dialogue_state: DialogueState, result: Dict[str, Any]):
        """Update dialogue state from execution result"""
        # Update context
        if 'context_updates' in result:
            dialogue_state.context.update(result['context_updates'])
        
        # Update slots
        if 'slot_updates' in result:
            for slot_name, slot_data in result['slot_updates'].items():
                if isinstance(slot_data, dict) and 'value' in slot_data:
                    dialogue_state.set_slot(
                        slot_name, 
                        slot_data['value'], 
                        slot_data.get('confidence', 1.0),
                        slot_data.get('source', 'executor')
                    )
                else:
                    dialogue_state.set_slot(slot_name, slot_data)
    
    def _determine_next_node(self, node_config: Dict[str, Any], dialogue_state: DialogueState) -> Optional[str]:
        """Determine next node using conditions or graph"""
        conditions = node_config.get('conditions', {})
        if conditions:
            eval_context = {
                'slots': dialogue_state.get_filled_slots(),
                'context': dialogue_state.context,
                'turn_count': dialogue_state.turn_count
            }
            result = self.condition_evaluator.evaluate_conditions(conditions, eval_context)
            if result:
                return result
        
        # Try graph-based transitions
        return self._get_next_from_graph(dialogue_state.current_node, dialogue_state)
    
    def _get_next_from_graph(self, current_node: str, dialogue_state: DialogueState = None) -> Optional[str]:
        """Get next node from graph structure"""
        if self.graph_builder:
            try:
                successors = self.graph_builder.get_successors(current_node)
                return successors[0] if successors else None
            except Exception:
                pass
        
        # Manual fallback with context matching
        for name, config in self.nodes_config.items():
            prev_nodes = config.get('prev_nodes', [])
            for prev in prev_nodes:
                if isinstance(prev, dict):
                    prev_name = prev['name']
                    required_context = prev.get('context')
                    
                    # Check if previous node matches
                    if prev_name == current_node:
                        # Check context condition if specified
                        if required_context and dialogue_state:
                            context_value = dialogue_state.context.get(required_context)
                            if context_value:  # Context exists and is truthy
                                return name
                        elif not required_context:  # No context required
                            return name
                else:
                    prev_name = str(prev)
                    if prev_name == current_node:
                        return name
        return None
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
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
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        dialogue_state = self.context_store.load_state(session_id)
        if not dialogue_state:
            return None
        
        # DialogueState should already be properly typed
        
        return {
            'session_id': session_id,
            'current_node': dialogue_state.current_node,
            'turn_count': dialogue_state.turn_count,
            'started_at': dialogue_state.started_at.isoformat(),
            'last_updated': dialogue_state.last_updated.isoformat(),
            'is_complete': dialogue_state.is_complete,
            'slots': dialogue_state.get_filled_slots(),
            'context': dialogue_state.context
        }
    
    def reset_session(self, session_id: str) -> bool:
        """Reset session"""
        try:
            dialogue_state = DialogueState(session_id=session_id)
            dialogue_state.update_node(self.start_node)
            self.context_store.save_state(session_id, dialogue_state)
            logger.info(f"Reset session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting session: {e}")
            return False
    
    def get_stage_detection_result(self, node_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed stage detection result with confidence and reasoning"""
        if node_name not in self.nodes_config:
            return None
        
        node_config = self.nodes_config[node_name]
        result = self.stage_manager.get_detection_result(node_config)
        
        if result:
            return {
                'node_name': node_name,
                'detected_stage': result.stage.value,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'matched_rules': [r.description for r in result.matched_rules]
            }
        return None
    
    def validate_stage_transition(self, from_node: str, to_node: str) -> bool:
        """Validate if stage transition between nodes is logically valid"""
        from_stage = self._get_node_stage(self.nodes_config.get(from_node, {}))
        to_stage = self._get_node_stage(self.nodes_config.get(to_node, {}))
        
        return self.stage_manager.validate_stage_flow(from_stage, to_stage)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics including stage detection metrics"""
        stage_stats = self.stage_manager.get_performance_stats()
        
        return {
            'stage_detection': stage_stats,
            'total_nodes': len(self.nodes_config),
            'start_node': self.start_node,
            'graph_enabled': self.graph_builder is not None
        }
    
    def analyze_chatbot_flow(self) -> Dict[str, Any]:
        """Analyze the entire chatbot flow and stage distribution"""
        stage_distribution = {}
        node_analysis = {}
        
        for node_name, node_config in self.nodes_config.items():
            stage = self._get_node_stage(node_config)
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            
            detection_result = self.get_stage_detection_result(node_name)
            if detection_result:
                node_analysis[node_name] = {
                    'stage': stage,
                    'confidence': detection_result['confidence'],
                    'reasoning': detection_result['reasoning']
                }
        
        return {
            'stage_distribution': stage_distribution,
            'node_analysis': node_analysis,
            'total_nodes': len(self.nodes_config),
            'unique_stages': len(stage_distribution)
        } 