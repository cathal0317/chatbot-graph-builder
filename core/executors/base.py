from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import logging

from ..models_simplified import DialogueState, NodeConfig, create_node_config, create_dialogue_state

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Base class for all executors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, 
                node_config: Union[Dict[str, Any], NodeConfig], 
                dialogue_state: DialogueState, 
                user_message: str,
                openai_client: Any) -> Dict[str, Any]:
        """Execute node logic and return response dict"""
        pass
    
    def prepare_inputs(self, node_config, dialogue_state):
        """Ensure inputs are properly formatted"""
        if isinstance(node_config, dict):
            node_config = create_node_config(node_config.get('name', 'unknown'), node_config)
        if not isinstance(dialogue_state, DialogueState):
            dialogue_state = create_dialogue_state(getattr(dialogue_state, 'session_id', None))
        return node_config, dialogue_state
    
    def get_response_template(self, node_config: Union[Dict, NodeConfig], key: str = "default") -> str:
        if isinstance(node_config, NodeConfig):
            responses = node_config.responses
        else:
            responses = node_config.get("responses", {})
        return responses.get(key, responses.get("default", ""))
    
    def get_node_param(self, node_config: Union[Dict, NodeConfig], key: str, default: Any = None) -> Any:
        if isinstance(node_config, NodeConfig):
            params = getattr(node_config, 'params', {}) or {}
        else:
            params = (node_config.get('params') or {}) if isinstance(node_config, dict) else {}
        return params.get(key, default)
    
    def get_current_node_turns(self, dialogue_state: DialogueState) -> int:
        try:
            return int(dialogue_state.context.get('node_turns', 1))
        except Exception:
            return 1
    
    def format_response(self, template: str, dialogue_state: DialogueState, 
                       extra_context: Dict[str, Any] = None) -> str:
        try:
            context = {
                **dialogue_state.get_filled_slots(),
                **dialogue_state.context,
                'turn_count': dialogue_state.turn_count,
                'session_id': dialogue_state.session_id,
                **(extra_context or {})
            }
            return template.format(**context)
        except (KeyError, Exception) as e:
            self.logger.warning(f"Template formatting failed: {e}")
            return template
    
    def generate_natural_response(self, node_config: Union[Dict, NodeConfig], 
                                dialogue_state: DialogueState, user_message: str,
                                openai_client: Any, intent_data: Dict[str, Any] = None,
                                fallback_template: str = None) -> str:
        try:
            context = {
                **dialogue_state.get_filled_slots(),
                **dialogue_state.context,
                'turn_count': dialogue_state.turn_count,
                'user_message': user_message,
                'node_purpose': node_config.description if hasattr(node_config, 'description') else node_config.get('description', ''),
                'current_node': dialogue_state.current_node
            }
            response = openai_client.generate_response(
                context=context,
                node_config=node_config.model_dump() if hasattr(node_config, 'model_dump') else node_config,
                intent_data=intent_data or {}
            )
            return response
        except Exception as e:
            self.logger.warning(f"LLM response generation failed: {e}")
            if fallback_template:
                return self.format_response(fallback_template, dialogue_state)
            else:
                return "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다. 다시 말씀해 주세요."

    def get_max_turns(self, node_config: Union[Dict, NodeConfig], default: int) -> int:
        raw_value = self.get_node_param(node_config, 'max_turn', None)
        if raw_value is None:
            raw_value = self.get_node_param(node_config, 'max_turns', None)
        if raw_value is None:
            return int(default)
        try:
            return int(raw_value)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid max_turn value: {raw_value}. Using default {default}.")
            return int(default) 

    def check_turn_limit(self, dialogue_state: DialogueState, max_turns: int = 10) -> bool:
        return dialogue_state.turn_count >= max_turns

    def handle_off_topic_input(self, user_message: str, node_config: Union[Dict, NodeConfig], 
                             dialogue_state: DialogueState, openai_client: Any) -> Dict[str, Any]:
        try:
            context = {
                **dialogue_state.get_filled_slots(),
                **dialogue_state.context,
                'user_message': user_message,
                'node_purpose': node_config.description if hasattr(node_config, 'description') else node_config.get('description', ''),
                'turn_count': dialogue_state.turn_count,
                'guidance_needed': True
            }
            response = openai_client.generate_response(
                context=context,
                node_config=node_config.model_dump() if hasattr(node_config, 'model_dump') else node_config,
                intent_data={'intent': 'off_topic', 'confidence': 0.8}
            )
            return {
                'response': response,
                'next_node': 'STAY_CURRENT',
                'context_updates': {
                    'off_topic_count': dialogue_state.context.get('off_topic_count', 0) + 1,
                    'last_off_topic_message': user_message
                }
            }
        except Exception as e:
            self.logger.warning(f"Off-topic handling failed: {e}")
            return {
                'response': (response if 'response' in locals() else '') + " 죄송합니다. 현재 진행 중인 단계에 맞는 정보를 제공해 주세요.",
                'next_node': 'STAY_CURRENT'
            }

   