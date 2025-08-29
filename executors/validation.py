from typing import Dict, Any
import logging
from core.unified_executor import BaseExecutor

logger = logging.getLogger(__name__)

class ValidationExecutor(BaseExecutor):
    """Executor for validation nodes"""
    
    def execute(self, node_config: Dict[str, Any], 
                dialogue_state: 'DialogueState', 
                user_message: str,
                openai_client: 'OpenAIClient') -> Dict[str, Any]:
        """Execute validation logic"""
        
        logger.debug(f"Executing validation node: {node_config.get('name')}")
        
        # Get validation parameters
        node_params = node_config.get('params', {})
        slots_to_validate = node_params.get('validate_slots', [])
        validation_rules = node_params.get('validation_rules', {})
        
        validation_results = {}
        all_valid = True
        
        # Validate each slot
        for slot_name in slots_to_validate:
            if dialogue_state.has_slot(slot_name):
                slot_value = dialogue_state.get_slot(slot_name)
                slot_rules = validation_rules.get(slot_name, {})
                
                validation_result = self._validate_slot_value(
                    slot_name, slot_value, slot_rules
                )
                validation_results[slot_name] = validation_result
                
                if not validation_result['valid']:
                    all_valid = False
            else:
                validation_results[slot_name] = {
                    'valid': False,
                    'error': f'{slot_name} is required but not provided'
                }
                all_valid = False
        
        # Generate response based on validation results
        if all_valid:
            response_template = self._get_response_template(node_config, "valid")
            if not response_template:
                response_template = "입력이 확인되었습니다."
            next_condition = "valid_input"
        else:
            response_template = self._get_response_template(node_config, "invalid")
            if not response_template:
                response_template = "입력 형식을 다시 확인해 주세요."
            next_condition = "invalid_format"
        
        # Prepare context for response
        context = {
            'validation_results': validation_results,
            'user_message': user_message,
            **dialogue_state.get_filled_slots()
        }
        
        response = self._format_response(response_template, context)
        
        # Update context with validation results
        context_updates = {
            'last_validation_results': validation_results,
            'last_validation_success': all_valid,
            'last_user_message': user_message
        }
        
        return {
            'response': response,
            'next_node': None,
            'context_updates': context_updates,
            'slot_updates': {},
            'validation_result': next_condition
        }
    
    def _validate_slot_value(self, slot_name: str, slot_value: Any, 
                           validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single slot value"""
        
        result = {
            'valid': True,
            'errors': []
        }
        
        # Type validation
        expected_type = validation_rules.get('type', 'string')
        if not self._check_type(slot_value, expected_type):
            result['valid'] = False
            result['errors'].append(f'Expected {expected_type}, got {type(slot_value).__name__}')
        
        # Required validation
        if validation_rules.get('required', False) and not slot_value:
            result['valid'] = False
            result['errors'].append(f'{slot_name} is required')
        
        # Length validation for strings
        if expected_type == 'string' and slot_value:
            min_length = validation_rules.get('min_length')
            max_length = validation_rules.get('max_length')
            
            if min_length and len(str(slot_value)) < min_length:
                result['valid'] = False
                result['errors'].append(f'Minimum length is {min_length}')
            
            if max_length and len(str(slot_value)) > max_length:
                result['valid'] = False
                result['errors'].append(f'Maximum length is {max_length}')
        
        # Range validation for numbers
        if expected_type in ['number', 'integer', 'float'] and slot_value is not None:
            try:
                num_value = float(slot_value)
                min_value = validation_rules.get('min_value')
                max_value = validation_rules.get('max_value')
                
                if min_value is not None and num_value < min_value:
                    result['valid'] = False
                    result['errors'].append(f'Minimum value is {min_value}')
                
                if max_value is not None and num_value > max_value:
                    result['valid'] = False
                    result['errors'].append(f'Maximum value is {max_value}')
            except (ValueError, TypeError):
                result['valid'] = False
                result['errors'].append('Invalid number format')
        
        # Pattern validation
        pattern = validation_rules.get('pattern')
        if pattern and slot_value:
            import re
            try:
                if not re.match(pattern, str(slot_value)):
                    result['valid'] = False
                    result['errors'].append('Does not match required pattern')
            except re.error:
                logger.warning(f'Invalid regex pattern: {pattern}')
        
        # Allowed values validation
        allowed_values = validation_rules.get('allowed_values')
        if allowed_values and slot_value not in allowed_values:
            result['valid'] = False
            result['errors'].append(f'Must be one of: {", ".join(map(str, allowed_values))}')
        
        return result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        
        if value is None:
            return True  # None is valid for optional fields
        
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        elif expected_type == 'float':
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == 'number':
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == 'boolean':
            return isinstance(value, bool) or str(value).lower() in ['true', 'false', '1', '0']
        elif expected_type == 'list':
            return isinstance(value, list)
        elif expected_type == 'dict':
            return isinstance(value, dict)
        
        return True  # Default to valid for unknown types 