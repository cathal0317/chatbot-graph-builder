from typing import Dict, Any
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class ConfirmationExecutor(BaseExecutor):
    """Executor for confirmation nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute confirmation logic"""
        
        logger.debug(f"Executing confirmation node: {node_config.get('name')}")
        
        # Check if this is the first time showing confirmation
        confirmation_shown = dialogue_state.context.get('confirmation_shown', False)
        
        if not confirmation_shown:
            # First time - show confirmation request
            response_template = self._get_response_template(node_config, "confirmation_request")
            if not response_template:
                response_template = "다음 내용을 확인해 주세요:\n{summary}\n\n확인하시려면 '네', 취소하시려면 '아니오'를 입력해 주세요."
            
            # Build summary from collected slots
            slots = dialogue_state.get_filled_slots()
            summary_parts = []
            for slot_name, slot_value in slots.items():
                summary_parts.append(f"{slot_name}: {slot_value}")
            summary = "\n".join(summary_parts)
            
            context = {
                'summary': summary,
                **slots
            }
            
            response = self._format_response(response_template, context)
            
            return {
                'response': response,
                'next_node': None,
                'context_updates': {
                    'confirmation_shown': True,
                    'last_user_message': user_message
                },
                'slot_updates': {}
            }
        
        # Process user's confirmation response
        user_response = user_message.lower().strip()
        
        # Determine user intent from response
        if self._is_confirmation(user_response):
            condition_result = "user_confirmed"
            response_template = self._get_response_template(node_config, "confirmed")
            if not response_template:
                response_template = "확인되었습니다. 처리를 진행하겠습니다."
            
        elif self._is_cancellation(user_response):
            condition_result = "user_cancelled"
            response_template = self._get_response_template(node_config, "cancelled")
            if not response_template:
                response_template = "취소되었습니다."
                
        elif self._is_modification_request(user_response):
            condition_result = "modify_requested"
            response_template = self._get_response_template(node_config, "modify")
            if not response_template:
                response_template = "수정을 위해 다시 진행하겠습니다."
                
        else:
            # Unclear response - ask again
            condition_result = None
            response_template = self._get_response_template(node_config, "options")
            if not response_template:
                response_template = "확인하시려면 '네' 또는 '확인', 취소하시려면 '아니오' 또는 '취소'를 입력해 주세요."
        
        # Prepare context for response
        context = {
            'user_response': user_response,
            **dialogue_state.get_filled_slots()
        }
        
        response = self._format_response(response_template, context)
        
        # Update context
        context_updates = {
            'last_user_message': user_message,
            'confirmation_response': user_response
        }
        
        if condition_result:
            context_updates['confirmation_result'] = condition_result
        
        return {
            'response': response,
            'next_node': None,
            'context_updates': context_updates,
            'slot_updates': {},
            'confirmation_result': condition_result
        }
    
    def _is_confirmation(self, user_response: str) -> bool:
        """Check if user response indicates confirmation"""
        confirmation_words = [
            '네', '예', '확인', '맞습니다', '맞아요', '좋습니다', '좋아요', 
            'yes', 'y', 'ok', 'okay', '응', 'ㅇㅇ', 'ㅇ', '1'
        ]
        return any(word in user_response for word in confirmation_words)
    
    def _is_cancellation(self, user_response: str) -> bool:
        """Check if user response indicates cancellation"""
        cancellation_words = [
            '아니오', '아니', '취소', '취소해주세요', '안해요', '그만', 
            'no', 'n', 'cancel', 'stop', '아니요', 'ㄴ', '0'
        ]
        return any(word in user_response for word in cancellation_words)
    
    def _is_modification_request(self, user_response: str) -> bool:
        """Check if user response indicates modification request"""
        modification_words = [
            '수정', '바꿔', '바꿔주세요', '변경', '고치고', '다시', 
            'modify', 'change', 'edit', '틀렸어요', '틀렸습니다'
        ]
        return any(word in user_response for word in modification_words) 