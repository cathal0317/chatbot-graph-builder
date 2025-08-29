from typing import Dict, Any
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class GreetingExecutor(BaseExecutor):
    """Executor for greeting and initial nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute greeting logic"""
        
        logger.debug(f"Executing greeting node: {node_config.get('name')}")
        
        # Determine if this is a returning user
        is_returning = dialogue_state.turn_count > 1
        
        # Choose appropriate response template
        if is_returning:
            response_template = self._get_response_template(node_config, "returning_user")
            if not response_template:
                response_template = self._get_response_template(node_config, "default")
        else:
            response_template = self._get_response_template(node_config, "default")
        
        # Prepare context for response formatting
        context = {
            'user_message': user_message,
            'turn_count': dialogue_state.turn_count,
            'is_returning': is_returning,
            **dialogue_state.get_filled_slots(),
            **dialogue_state.context
        }
        
        # Generate response
        if response_template:
            response = self._format_response(response_template, context)
        else:
            # Use OpenAI to generate a greeting
            system_context = {
                **context,
                'node_description': node_config.get('description', 'Greeting node'),
                'stage': 'greeting'
            }
            response = openai_client.generate_response(
                context=system_context,
                node_config=node_config
            )
        
        # Update context with greeting information
        context_updates = {
            'greeted': True,
            'greeting_time': dialogue_state.last_updated.isoformat(),
            'last_user_message': user_message
        }
        
        return {
            'response': response,
            'next_node': None,  # Let condition evaluator decide next node
            'context_updates': context_updates,
            'slot_updates': {}
        } 