from typing import Dict, Any, List
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class SlotFillingExecutor(BaseExecutor):
    """Executor for slot filling and data collection nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute slot filling logic"""
        
        logger.debug(f"Executing slot filling node: {node_config.get('name')}")
        
        # Get required slots from node parameters
        node_params = node_config.get('params', {})
        required_slots = node_params.get('required_slots', [])
        slot_name = node_params.get('slot_name')  # For single slot collection
        
        # If this is a single slot collection node
        if slot_name:
            return self._handle_single_slot(
                slot_name, node_config, dialogue_state, user_message, openai_client
            )
        
        # Multi-slot collection
        if required_slots:
            return self._handle_multi_slot(
                required_slots, node_config, dialogue_state, user_message, openai_client
            )
        
        # Extract entities from user message using NLU
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config,
                context=dialogue_state.context
            )
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            intent_data = {'entities': {}}
        
        # Update slots with extracted entities
        slot_updates = {}
        entities = intent_data.get('entities', {})
        for entity_name, entity_value in entities.items():
            if entity_value:
                slot_updates[entity_name] = {
                    'value': entity_value,
                    'confidence': intent_data.get('confidence', 1.0)
                }
        
        # Generate response
        response = self._generate_slot_response(
            node_config, dialogue_state, slot_updates, openai_client
        )
        
        return {
            'response': response,
            'next_node': None,
            'context_updates': {
                'last_user_message': user_message,
                'slot_collection_attempt': dialogue_state.context.get('slot_collection_attempt', 0) + 1
            },
            'slot_updates': slot_updates,
            'intent_data': intent_data
        }
    
    def _handle_single_slot(self, slot_name: str, node_config: Dict[str, Any],
                           dialogue_state: 'DialogueState', user_message: str,
                           openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Handle collection of a single slot"""
        
        # Check if slot is already filled
        if dialogue_state.has_slot(slot_name):
            confidence = dialogue_state.get_slot_confidence(slot_name)
            if confidence >= 0.8:  # High confidence, slot already filled
                response_template = self._get_response_template(node_config, "already_filled")
                if not response_template:
                    response_template = "이미 {slot_name}을(를) 입력받았습니다."
                
                context = {
                    'slot_name': slot_name,
                    'slot_value': dialogue_state.get_slot(slot_name),
                    **dialogue_state.get_filled_slots()
                }
                response = self._format_response(response_template, context)
                
                return {
                    'response': response,
                    'next_node': None,
                    'context_updates': {'last_user_message': user_message},
                    'slot_updates': {}
                }
        
        # Extract slot value from user message
        slot_value = self._extract_slot_value(
            slot_name, user_message, node_config, dialogue_state, openai_client
        )
        
        slot_updates = {}
        if slot_value:
            slot_updates[slot_name] = slot_value
        
        # Generate appropriate response
        if slot_value:
            # Slot value extracted successfully
            response_template = self._get_response_template(node_config, "confirmation")
            if not response_template:
                response_template = "{slot_name}이(가) {slot_value}로 확인되었습니다."
            
            context = {
                'slot_name': slot_name,
                'slot_value': slot_value['value'],
                **dialogue_state.get_filled_slots()
            }
        else:
            # Need to ask for the slot
            response_template = self._get_response_template(node_config, "initial")
            if not response_template:
                response_template = self._get_response_template(node_config, "retry")
            if not response_template:
                response_template = f"{slot_name}을(를) 입력해 주세요."
            
            context = {
                'slot_name': slot_name,
                **dialogue_state.get_filled_slots()
            }
        
        response = self._format_response(response_template, context)
        
        return {
            'response': response,
            'next_node': None,
            'context_updates': {'last_user_message': user_message},
            'slot_updates': slot_updates
        }
    
    def _handle_multi_slot(self, required_slots: List[str], node_config: Dict[str, Any],
                          dialogue_state: 'DialogueState', user_message: str,
                          openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Handle collection of multiple slots"""
        
        # Check which slots are missing
        missing_slots = dialogue_state.get_missing_slots(required_slots)
        
        if not missing_slots:
            # All slots are filled
            response_template = self._get_response_template(node_config, "completion")
            if not response_template:
                response_template = "모든 정보가 입력되었습니다."
            
            context = {
                **dialogue_state.get_filled_slots(),
                'total_slots': len(required_slots),
                'filled_slots': len(required_slots)
            }
            response = self._format_response(response_template, context)
            
            return {
                'response': response,
                'next_node': None,
                'context_updates': {
                    'last_user_message': user_message,
                    'form_complete': True
                },
                'slot_updates': {}
            }
        
        # Extract entities from user message
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config,
                context=dialogue_state.context
            )
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            intent_data = {'entities': {}}
        
        # Update slots with extracted entities
        slot_updates = {}
        entities = intent_data.get('entities', {})
        
        for slot_name in missing_slots:
            if slot_name in entities and entities[slot_name]:
                slot_updates[slot_name] = {
                    'value': entities[slot_name],
                    'confidence': intent_data.get('confidence', 1.0)
                }
        
        # Generate response asking for next missing slot
        remaining_missing = [slot for slot in missing_slots if slot not in slot_updates]
        
        if remaining_missing:
            next_slot = remaining_missing[0]
            response_template = self._get_response_template(node_config, "field_prompt")
            if not response_template:
                response_template = "{field_name}을(를) 입력해 주세요."
            
            context = {
                'field_name': next_slot,
                'progress': ((len(required_slots) - len(remaining_missing)) / len(required_slots)) * 100,
                'current': len(required_slots) - len(remaining_missing) + 1,
                'total': len(required_slots),
                **dialogue_state.get_filled_slots()
            }
        else:
            # Progress update
            response_template = self._get_response_template(node_config, "progress")
            if not response_template:
                response_template = "정보가 업데이트되었습니다."
            
            context = {
                'progress': ((len(required_slots) - len(missing_slots) + len(slot_updates)) / len(required_slots)) * 100,
                'current': len(required_slots) - len(missing_slots) + len(slot_updates),
                'total': len(required_slots),
                **dialogue_state.get_filled_slots()
            }
        
        response = self._format_response(response_template, context)
        
        return {
            'response': response,
            'next_node': None,
            'context_updates': {'last_user_message': user_message},
            'slot_updates': slot_updates
        }
    
    def _extract_slot_value(self, slot_name: str, user_message: str,
                           node_config: Dict[str, Any], dialogue_state: 'DialogueState',
                           openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Extract slot value from user message"""
        
        try:
            # Use OpenAI to extract the specific slot value
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config={
                    **node_config,
                    'target_slot': slot_name,
                    'slot_extraction_mode': True
                },
                context=dialogue_state.context
            )
            
            entities = intent_data.get('entities', {})
            if slot_name in entities and entities[slot_name]:
                return {
                    'value': entities[slot_name],
                    'confidence': intent_data.get('confidence', 1.0)
                }
        
        except Exception as e:
            logger.error(f"Slot extraction failed for {slot_name}: {e}")
        
        return None
    
    def _generate_slot_response(self, node_config: Dict[str, Any],
                               dialogue_state: 'DialogueState',
                               slot_updates: Dict[str, Any],
                               openai_client: 'OpenAIClient') -> str:
        """Generate appropriate response for slot filling"""
        
        if slot_updates:
            # Slots were updated
            response_template = self._get_response_template(node_config, "success")
            if not response_template:
                response_template = "정보가 확인되었습니다."
        else:
            # No slots were extracted
            response_template = self._get_response_template(node_config, "retry")
            if not response_template:
                response_template = self._get_response_template(node_config, "clarification")
            if not response_template:
                response_template = "정보를 다시 입력해 주세요."
        
        context = {
            'slot_updates': slot_updates,
            **dialogue_state.get_filled_slots(),
            **dialogue_state.context
        }
        
        if response_template:
            return self._format_response(response_template, context)
        else:
            # Generate response using OpenAI
            return openai_client.generate_response(
                context=context,
                node_config=node_config
            ) 