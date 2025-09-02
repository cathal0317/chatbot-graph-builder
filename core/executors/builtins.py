from __future__ import annotations

from typing import Dict, Any
import logging

from .base import BaseExecutor

logger = logging.getLogger(__name__)

"""각 스테이지 별 룰 기반 실행"""

class GreetingExecutor(BaseExecutor):
    
    def execute(self, node_config, dialogue_state, user_message, openai_client):
        node_config, dialogue_state = self.prepare_inputs(node_config, dialogue_state)
        
        # Per-node max turns (default 5)
        max_turns = self.get_max_turns(node_config, 5)
        if self.get_current_node_turns(dialogue_state) >= max_turns:
            return {
                'response': "안녕히 가세요. 또 다른 도움이 필요하시면 언제든 말씀해 주세요!",
                'next_node': None,
                'context_updates': {'session_ended': True, 'end_reason': 'node_turn_limit'}
            }
        
        response_key = "returning_user" if dialogue_state.turn_count > 1 else "default"
        fallback_template = self.get_response_template(node_config, response_key)
        
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

        # 노드 턴 횟수가 max_turns를 넘어가면 출력
        max_turns = self.get_max_turns(node_config, 8)
        if self.get_current_node_turns(dialogue_state) >= max_turns:
            return {
                'response': "정보 수집에 시간이 많이 걸리고 있습니다. 처음부터 다시 시작하거나 도움을 요청하시겠어요?",
                'next_node': None,
                'context_updates': {'session_ended': True, 'end_reason': 'node_turn_limit'}
            }
        
        required_slots = node_config.params.get('required_slots', [])
        
        slot_updates: Dict[str, Any] = {}
        intent_data: Dict[str, Any] = {}
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config.model_dump(),
                context=dialogue_state.context
            )
            entities = intent_data.get('entities', {})
            
            # 필수 슬롯 
            for entity_name, entity_value in entities.items():
                if entity_name in required_slots and entity_value:
                    slot_updates[entity_name] = {
                        'value': entity_value,
                        'confidence': intent_data.get('confidence', 0.8),
                        'source': 'nlu'
                    }
            
            # 엔티티 기반 보편 슬롯 업데이트 (NLU 주도)
            for entity_name, entity_value in entities.items():
                if not entity_value:
                    continue
                if entity_name in slot_updates:
                    continue
                slot_updates[entity_name] = {
                    'value': entity_value,
                    'confidence': intent_data.get('confidence', 0.8),
                    'source': 'nlu_refilled' if dialogue_state.has_slot(entity_name) else 'nlu_auto'
                }
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        missing_slots = []
        for slot in required_slots:
            if not dialogue_state.has_slot(slot) and slot not in slot_updates:
                missing_slots.append(slot)
        
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
        
        if missing_slots:
            fallback_template = self.get_response_template(node_config, "initial")
            response = self.generate_natural_response(
                node_config=node_config,
                dialogue_state=dialogue_state,
                user_message=user_message,
                openai_client=openai_client,
                intent_data={**intent_data, 'missing_slots': missing_slots},
                fallback_template=fallback_template
            )
            next_node = "STAY_CURRENT"
        else:
            fallback_template = self.get_response_template(node_config, "confirmation")
            response = self.generate_natural_response(
                node_config=node_config,
                dialogue_state=dialogue_state,
                user_message=user_message,
                openai_client=openai_client,
                intent_data={**intent_data, 'all_slots_filled': True},
                fallback_template=fallback_template
            )
            next_node = None
        
        return {
            'response': response,
            'slot_updates': slot_updates,
            'next_node': next_node
        }


class DefaultExecutor(BaseExecutor):
    """Default executor for general cases"""
    
    def execute(self, node_config, dialogue_state, user_message, openai_client):
        node_config, dialogue_state = self.prepare_inputs(node_config, dialogue_state)
        
        # Per-node max turns (default 8)
        max_turns = self.get_max_turns(node_config, 8)
        if self.get_current_node_turns(dialogue_state) >= max_turns:
            return {
                'response': "대화가 길어지고 있습니다. 필요한 정보를 정리해서 다시 시작하시겠어요?",
                'next_node': None,
                'context_updates': {'session_ended': True, 'end_reason': 'node_turn_limit'}
            }
        
        intent_data: Dict[str, Any] = {}
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config.model_dump(),
                context=dialogue_state.context
            )
        except Exception as e:
            logger.warning(f"Intent extraction failed: {e}")
        
        fallback_template = self.get_response_template(node_config)
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