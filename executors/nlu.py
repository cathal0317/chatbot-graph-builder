from typing import Dict, Any
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class NLUExecutor(BaseExecutor):
    """Executor for NLU (Natural Language Understanding) nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute NLU logic"""
        
        logger.debug(f"Executing NLU node: {node_config.get('name')}")
        
        # Extract intent and entities using OpenAI
        try:
            intent_data = openai_client.extract_intent_entities(
                user_message=user_message,
                node_config=node_config,
                context=dialogue_state.context
            )
            
            logger.debug(f"NLU extracted: {intent_data}")
            
        except Exception as e:
            logger.error(f"NLU extraction failed: {e}")
            intent_data = {
                'intent': 'unknown',
                'entities': {},
                'confidence': 0.0
            }
        
        # Determine response based on confidence
        confidence = intent_data.get('confidence', 0.0)
        intent = intent_data.get('intent', 'unknown')
        
        if confidence >= 0.8:
            response_key = "understood"
        elif confidence >= 0.5:
            response_key = "processing"
        else:
            response_key = "unclear"
        
        # Get response template
        response_template = self._get_response_template(node_config, response_key)
        if not response_template:
            response_template = self._get_response_template(node_config, "default")
        
        # Prepare context for response formatting
        context = {
            'user_message': user_message,
            'intent': intent,
            'confidence': confidence,
            'entities': intent_data.get('entities', {}),
            **dialogue_state.get_filled_slots(),
            **dialogue_state.context
        }
        
        # Generate response
        if response_template:
            response = self._format_response(response_template, context)
        else:
            # Use OpenAI to generate appropriate response
            response = openai_client.generate_response(
                context=context,
                node_config=node_config,
                intent_data=intent_data
            )
        
        # Update context with NLU results
        context_updates = {
            'last_intent': intent,
            'last_confidence': confidence,
            'last_entities': intent_data.get('entities', {}),
            'last_user_message': user_message
        }
        
        # Extract entities as potential slot updates
        slot_updates = {}
        entities = intent_data.get('entities', {})
        for entity_name, entity_value in entities.items():
            if entity_value:  # Only update if entity has a value
                slot_updates[entity_name] = {
                    'value': entity_value,
                    'confidence': confidence
                }
        
        return {
            'response': response,
            'next_node': None,  # Let condition evaluator decide based on intent
            'context_updates': context_updates,
            'slot_updates': slot_updates,
            'intent_data': intent_data  # Pass intent data for condition evaluation
        } 