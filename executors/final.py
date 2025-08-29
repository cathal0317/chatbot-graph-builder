from typing import Dict, Any
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class FinalExecutor(BaseExecutor):
    """Executor for final and completion nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute final logic"""
        
        logger.debug(f"Executing final node: {node_config.get('name')}")
        
        # Get the appropriate response template
        stage = node_config.get('stage', 'final')
        node_name = node_config.get('name', '')
        
        # Determine response key based on node type
        response_key = "default"
        
        if 'complete' in node_name.lower() or 'success' in node_name.lower():
            response_key = "success"
        elif 'cancel' in node_name.lower():
            response_key = "cancelled"
        elif 'error' in node_name.lower() or 'fail' in node_name.lower():
            response_key = "error"
        
        response_template = self._get_response_template(node_config, response_key)
        if not response_template:
            response_template = self._get_response_template(node_config, "default")
        
        # Prepare context for response formatting
        context = {
            'user_message': user_message,
            **dialogue_state.get_filled_slots(),
            **dialogue_state.context
        }
        
        # Generate response
        if response_template:
            response = self._format_response(response_template, context)
        else:
            # Use OpenAI to generate a completion response
            response = openai_client.generate_response(
                context=context,
                node_config=node_config
            )
        
        # Mark session as complete
        context_updates = {
            'session_completed': True,
            'completion_time': dialogue_state.last_updated.isoformat(),
            'final_node': node_config.get('name'),
            'last_user_message': user_message
        }
        
        # Add completion summary
        if dialogue_state.get_filled_slots():
            context_updates['completion_summary'] = dialogue_state.get_filled_slots()
        
        return {
            'response': response,
            'next_node': None,  # No next node for final states
            'context_updates': context_updates,
            'slot_updates': {},
            'session_complete': True
        }


class ErrorExecutor(BaseExecutor):
    """Executor for error handling nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute error handling logic"""
        
        logger.debug(f"Executing error node: {node_config.get('name')}")
        
        # Increment error count
        dialogue_state.increment_error()
        
        # Check error count to determine response
        error_count = dialogue_state.error_count
        
        if error_count == 1:
            response_key = "first_error"
        elif error_count == 2:
            response_key = "second_error"
        elif error_count >= 3:
            response_key = "repeated_error"
        else:
            response_key = "default"
        
        response_template = self._get_response_template(node_config, response_key)
        if not response_template:
            response_template = self._get_response_template(node_config, "error")
        if not response_template:
            response_template = self._get_response_template(node_config, "default")
        
        # Prepare context for response
        context = {
            'user_message': user_message,
            'error_count': error_count,
            **dialogue_state.get_filled_slots(),
            **dialogue_state.context
        }
        
        # Generate response
        if response_template:
            response = self._format_response(response_template, context)
        else:
            # Use OpenAI to generate error response
            response = openai_client.generate_response(
                context={
                    **context,
                    'is_error_situation': True,
                    'error_type': node_config.get('description', 'Unknown error')
                },
                node_config=node_config
            )
        
        # Update context
        context_updates = {
            'last_error_node': node_config.get('name'),
            'last_error_time': dialogue_state.last_updated.isoformat(),
            'last_user_message': user_message
        }
        
        # If too many errors, suggest ending session
        session_complete = False
        if error_count >= 5:
            context_updates['too_many_errors'] = True
            session_complete = True
        
        return {
            'response': response,
            'next_node': None,
            'context_updates': context_updates,
            'slot_updates': {},
            'session_complete': session_complete
        } 