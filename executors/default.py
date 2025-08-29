from typing import Dict, Any
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class DefaultExecutor(BaseExecutor):
    """Default executor for unknown or unhandled stages"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute default node logic"""
        
        logger.info(f"Using default executor for node: {node_config.get('name')}")
        
        # Get response template
        response_template = self._get_response_template(node_config, "default")
        
        # Prepare context for response formatting
        context = {
            'user_message': user_message,
            'node_name': node_config.get('name'),
            'stage': node_config.get('stage'),
            **dialogue_state.get_filled_slots(),
            **dialogue_state.context
        }
        
        # Generate response
        if response_template:
            response = self._format_response(response_template, context)
        else:
            # Use OpenAI to generate a response if no template
            response = openai_client.generate_response(
                context=context,
                node_config=node_config
            )
        
        return {
            'response': response,
            'next_node': None,  # Let condition evaluator decide
            'context_updates': {
                'last_user_message': user_message,
                'last_node': node_config.get('name')
            },
            'slot_updates': {}
        } 