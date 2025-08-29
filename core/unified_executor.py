"""
Simple Executor System - clean and straightforward
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from importlib import import_module
import logging

from .models_simplified import DialogueState, NodeConfig, ExecutionResult, create_node_config, create_dialogue_state
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)


# ============================================================================
# Base Executor
# ============================================================================

class BaseExecutor(ABC):
    """Base class for all executors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, 
                node_config: Union[Dict[str, Any], NodeConfig], 
                dialogue_state: DialogueState, 
                user_message: str,
                openai_client: OpenAIClient) -> Dict[str, Any]:
        """Execute node logic and return response dict"""
        pass
    
    def prepare_inputs(self, node_config, dialogue_state):
        """Ensure inputs are properly formatted"""
        # Convert dict to NodeConfig if needed
        if isinstance(node_config, dict):
            node_config = create_node_config(node_config.get('name', 'unknown'), node_config)
        
        # Ensure DialogueState (should already be correct type)
        if not isinstance(dialogue_state, DialogueState):
            dialogue_state = create_dialogue_state(getattr(dialogue_state, 'session_id', None))
        
        return node_config, dialogue_state
    
    def get_response_template(self, node_config: Union[Dict, NodeConfig], key: str = "default") -> str:
        """Get response template from node config"""
        if isinstance(node_config, NodeConfig):
            responses = node_config.responses
        else:
            responses = node_config.get("responses", {})
        return responses.get(key, responses.get("default", ""))
    
    def format_response(self, template: str, dialogue_state: DialogueState, 
                       extra_context: Dict[str, Any] = None) -> str:
        """Format response template with dialogue context"""
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
                                openai_client, intent_data: Dict[str, Any] = None,
                                fallback_template: str = None) -> str:
        """Generate natural response using LLM with fallback to template"""
        try:
            # Prepare context for LLM
            context = {
                **dialogue_state.get_filled_slots(),
                **dialogue_state.context,
                'turn_count': dialogue_state.turn_count,
                'user_message': user_message,
                'node_purpose': node_config.description if hasattr(node_config, 'description') else node_config.get('description', ''),
                'current_node': dialogue_state.current_node
            }
            
            # Use LLM to generate natural response
            response = openai_client.generate_response(
                context=context,
                node_config=node_config.model_dump() if hasattr(node_config, 'model_dump') else node_config,
                intent_data=intent_data or {}
            )
            
            return response
            
        except Exception as e:
            self.logger.warning(f"LLM response generation failed: {e}")
            
            # Fallback to template or default message
            if fallback_template:
                return self.format_response(fallback_template, dialogue_state)
            else:
                return "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다. 다시 말씀해 주세요."
    
    def check_turn_limit(self, dialogue_state: DialogueState, max_turns: int = 10) -> bool:
        """Check if turn limit is exceeded"""
        return dialogue_state.turn_count >= max_turns
    
    def handle_off_topic_input(self, user_message: str, node_config: Union[Dict, NodeConfig], 
                             dialogue_state: DialogueState, openai_client) -> Dict[str, Any]:
        """Handle when user input is off-topic for current node"""
        try:
            # Generate contextual response for off-topic input
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
                'response': "죄송합니다. 현재 진행 중인 단계에 맞는 정보를 제공해 주세요.",
                'next_node': 'STAY_CURRENT'
            }


# ============================================================================
# Core Executors
# ============================================================================

class GreetingExecutor(BaseExecutor):
    """Handles greeting and initial interactions"""
    
    def execute(self, node_config, dialogue_state, user_message, openai_client):
        node_config, dialogue_state = self.prepare_inputs(node_config, dialogue_state)
        
        # Check turn limit
        if self.check_turn_limit(dialogue_state, max_turns=5):
            return {
                'response': "안녕히 가세요. 또 다른 도움이 필요하시면 언제든 말씀해 주세요!",
                'next_node': None,
                'context_updates': {'session_ended': True, 'end_reason': 'turn_limit'}
            }
        
        # Get template as fallback
        response_key = "returning_user" if dialogue_state.turn_count > 1 else "default"
        fallback_template = self.get_response_template(node_config, response_key)
        
        # Generate natural response using LLM
        response = self.generate_natural_response(
            node_config=node_config,
            dialogue_state=dialogue_state,
            user_message=user_message,
            openai_client=openai_client,
            intent_data={'intent': 'greeting', 'confidence': 0.9},
            fallback_template=fallback_template
        )
        
        return {
            'response': response,
            'context_updates': {
                'start': True,
                'greeted': True,
                'last_user_message': user_message
            }
        }


class SlotFillingExecutor(BaseExecutor):
    """Handles slot filling and information collection"""
    
    def execute(self, node_config, dialogue_state, user_message, openai_client):
        node_config, dialogue_state = self.prepare_inputs(node_config, dialogue_state)
        
        # Check turn limit - slot filling can take longer
        if self.check_turn_limit(dialogue_state, max_turns=15):
            return {
                'response': "정보 수집에 시간이 많이 걸리고 있습니다. 처음부터 다시 시작하거나 도움을 요청하시겠어요?",
                'next_node': None,
                'context_updates': {'session_ended': True, 'end_reason': 'turn_limit_slot_filling'}
            }
        
        # Get required slots
        required_slots = node_config.params.get('required_slots', [])
        
        # Extract entities using OpenAI
        slot_updates = {}
        intent_data = {}
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config.model_dump(),
                context=dialogue_state.context
            )
            entities = intent_data.get('entities', {})
            
            # Update slots with extracted entities
            for entity_name, entity_value in entities.items():
                if entity_name in required_slots and entity_value:
                    slot_updates[entity_name] = {
                        'value': entity_value,
                        'confidence': intent_data.get('confidence', 0.8),
                        'source': 'nlu'
                    }
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
        
        # Check missing slots
        missing_slots = []
        for slot in required_slots:
            if not dialogue_state.has_slot(slot) and slot not in slot_updates:
                missing_slots.append(slot)
        
        # Check if input seems off-topic
        if intent_data.get('confidence', 1.0) < 0.5 and not slot_updates:
            off_topic_count = dialogue_state.context.get('off_topic_count', 0)
            if off_topic_count >= 3:
                return {
                    'response': "대화가 잘 진행되지 않고 있습니다. 처음부터 다시 시작하시겠어요?",
                    'next_node': None,
                    'context_updates': {'session_ended': True, 'end_reason': 'too_many_off_topic'}
                }
            else:
                return self.handle_off_topic_input(user_message, node_config, dialogue_state, openai_client)
        
        # Generate natural response
        if missing_slots:
            # Still need more information
            fallback_template = self.get_response_template(node_config, "initial")
            
            response = self.generate_natural_response(
                node_config=node_config,
                dialogue_state=dialogue_state,
                user_message=user_message,
                openai_client=openai_client,
                intent_data={**intent_data, 'missing_slots': missing_slots},
                fallback_template=fallback_template
            )
            next_node = "STAY_CURRENT"  # Stay in current node
        else:
            # All slots filled
            fallback_template = self.get_response_template(node_config, "confirmation")
            
            response = self.generate_natural_response(
                node_config=node_config,
                dialogue_state=dialogue_state,
                user_message=user_message,
                openai_client=openai_client,
                intent_data={**intent_data, 'all_slots_filled': True},
                fallback_template=fallback_template
            )
            next_node = None  # Allow transition
        
        return {
            'response': response,
            'slot_updates': slot_updates,
            'next_node': next_node
        }


class DefaultExecutor(BaseExecutor):
    """Default executor for general cases"""
    
    def execute(self, node_config, dialogue_state, user_message, openai_client):
        node_config, dialogue_state = self.prepare_inputs(node_config, dialogue_state)
        
        # Check turn limit
        if self.check_turn_limit(dialogue_state, max_turns=8):
            return {
                'response': "대화가 길어지고 있습니다. 필요한 정보를 정리해서 다시 시작하시겠어요?",
                'next_node': None,
                'context_updates': {'session_ended': True, 'end_reason': 'turn_limit_default'}
            }
        
        # Extract intent for better response generation
        intent_data = {}
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config.model_dump(),
                context=dialogue_state.context
            )
        except Exception as e:
            self.logger.warning(f"Intent extraction failed: {e}")
        
        # Get template as fallback
        fallback_template = self.get_response_template(node_config)
        
        # Generate natural response using LLM
        response = self.generate_natural_response(
            node_config=node_config,
            dialogue_state=dialogue_state,
            user_message=user_message,
            openai_client=openai_client,
            intent_data=intent_data,
            fallback_template=fallback_template
        )
        
        return {
            'response': response,
            'context_updates': {
                'last_user_message': user_message,
                'last_intent': intent_data.get('intent'),
                'last_confidence': intent_data.get('confidence', 0.0)
            }
        }


# ============================================================================
# Legacy Executor Wrapper
# ============================================================================

class LegacyExecutorWrapper(BaseExecutor):
    """Wraps old-style executors to work with new interface"""
    
    def __init__(self, legacy_executor):
        super().__init__()
        self.legacy_executor = legacy_executor
    
    def execute(self, node_config, dialogue_state, user_message, openai_client):
        node_config, dialogue_state = self.prepare_inputs(node_config, dialogue_state)
        
        # Convert to legacy format
        legacy_config = self._to_legacy_config(node_config)
        legacy_state = self._to_legacy_state(dialogue_state)
        
        try:
            result = self.legacy_executor.execute(
                legacy_config, legacy_state, user_message, openai_client
            )
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {'response': str(result)}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Legacy executor failed: {e}")
            return {
                'response': "죄송합니다. 처리 중 오류가 발생했습니다.",
                'error': True
            }
    
    def _to_legacy_config(self, node_config: NodeConfig) -> Dict[str, Any]:
        """Convert NodeConfig to legacy dict format"""
        return {
            'name': node_config.name,
            'stage': str(node_config.stage),
            'description': node_config.description,
            'ko_name': node_config.ko_name,
            'visible': node_config.visible,
            'prev_nodes': node_config.prev_nodes,
            'conditions': node_config.conditions,
            'responses': node_config.responses,
            'actions': node_config.actions,
            'params': node_config.params
        }
    
    def _to_legacy_state(self, dialogue_state: DialogueState):
        """Convert DialogueState to legacy format"""
        class LegacyState:
            def __init__(self, state: DialogueState):
                self.session_id = state.session_id
                self.current_node = state.current_node
                self.previous_node = state.previous_node
                self.turn_count = state.turn_count
                self.error_count = state.error_count
                self.started_at = state.started_at
                self.last_updated = state.last_updated
                self.is_complete = state.is_complete
                self.context = state.context
                
                # Convert slots to legacy format
                self.slots = {}
                for name, slot_value in state.slots.items():
                    self.slots[name] = {
                        'value': slot_value.value,
                        'confidence': slot_value.confidence,
                        'updated_at': slot_value.updated_at.isoformat()
                    }
            
            def get_slot(self, name):
                slot_data = self.slots.get(name)
                return slot_data['value'] if slot_data else None
            
            def has_slot(self, name):
                return name in self.slots and self.slots[name]['value'] is not None
            
            def get_filled_slots(self):
                return {name: data['value'] for name, data in self.slots.items() 
                       if data['value'] is not None}
        
        return LegacyState(dialogue_state)


# ============================================================================
# Executor Factory
# ============================================================================

class ExecutorFactory:
    """Simple factory for creating executors"""
    
    def __init__(self):
        # Built-in executors
        self.executors = {
            'initial': GreetingExecutor,
            'greeting': GreetingExecutor,
            'slot_filling': SlotFillingExecutor,
            'default': DefaultExecutor,
        }
        
        # Legacy executor registry
        self.legacy_registry = {
            'validation': 'executors.validation:ValidationExecutor',
            'confirmation': 'executors.confirmation:ConfirmationExecutor',
            'final': 'executors.final:FinalExecutor',
            'error': 'executors.error:ErrorExecutor',
            'nlu': 'executors.nlu:NLUExecutor',
            'api_call': 'executors.api_call:APIExecutor',
            'processing': 'executors.processing:ProcessingExecutor',
        }
        
        self._cache = {}
    
    def get(self, stage: str) -> BaseExecutor:
        """Get executor for stage"""
        if stage not in self._cache:
            # Check built-in executors first
            if stage in self.executors:
                self._cache[stage] = self.executors[stage]()
            
            # Try legacy executors
            elif stage in self.legacy_registry:
                try:
                    executor_path = self.legacy_registry[stage]
                    module_path, class_name = executor_path.split(":")
                    module = import_module(module_path)
                    executor_class = getattr(module, class_name)
                    legacy_executor = executor_class()
                    self._cache[stage] = LegacyExecutorWrapper(legacy_executor)
                except Exception as e:
                    self.logger.warning(f"Failed to load legacy executor for {stage}: {e}")
                    self._cache[stage] = DefaultExecutor()
            
            # Fallback to default
            else:
                self._cache[stage] = DefaultExecutor()
        
        return self._cache[stage]
    
    def register(self, stage: str, executor_class: type):
        """Register a custom executor"""
        self.executors[stage] = executor_class
        self._cache.pop(stage, None)


# ============================================================================
# Global Factory Instance
# ============================================================================

# Create the global factory instance
executor_factory = ExecutorFactory()

# Backward compatibility alias
unified_factory = executor_factory 