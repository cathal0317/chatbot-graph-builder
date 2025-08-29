import json
import os
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SlotLoader:
    """Loads and validates slot configurations from JSON files"""
    
    def __init__(self):
        self.slot_types = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'float': float,
            'boolean': bool,
            'date': str,  # Will be validated separately
            'time': str,  # Will be validated separately
            'datetime': str,  # Will be validated separately
            'email': str,  # Will be validated separately
            'phone': str,  # Will be validated separately
            'url': str,  # Will be validated separately
            'list': list,
            'dict': dict
        }
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load slot configurations from JSON file"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Slot configuration file not found: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError("Slot configuration must be a dictionary")
            
            # Validate and process slots
            processed_slots = {}
            for slot_name, slot_config in data.items():
                processed_slot = self._validate_and_process_slot(slot_name, slot_config)
                processed_slots[slot_name] = processed_slot
            
            logger.info(f"Loaded {len(processed_slots)} slots from {file_path}")
            return processed_slots
            
        except Exception as e:
            logger.error(f"Failed to load slots from {file_path}: {e}")
            raise
    
    def _validate_and_process_slot(self, slot_name: str, slot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process a single slot configuration"""
        if not isinstance(slot_config, dict):
            raise ValueError(f"Slot '{slot_name}' configuration must be a dictionary")
        
        processed_config = slot_config.copy()
        
        # Validate slot type
        slot_type = processed_config.get('type', 'string')
        if slot_type not in self.slot_types:
            raise ValueError(f"Slot '{slot_name}' has invalid type: {slot_type}")
        
        # Set default values
        processed_config.setdefault('required', False)
        processed_config.setdefault('description', f"Slot for {slot_name}")
        processed_config.setdefault('prompt', f"Please provide {slot_name}")
        processed_config.setdefault('validation', {})
        
        # Process validation rules
        validation = processed_config.get('validation', {})
        processed_config['validation'] = self._process_validation_rules(slot_name, validation, slot_type)
        
        # Process examples
        examples = processed_config.get('examples', [])
        if not isinstance(examples, list):
            logger.warning(f"Slot '{slot_name}' examples should be a list, converting")
            processed_config['examples'] = [examples] if examples else []
        
        return processed_config
    
    def _process_validation_rules(self, slot_name: str, validation: Dict[str, Any], slot_type: str) -> Dict[str, Any]:
        """Process validation rules for a slot"""
        processed_validation = validation.copy()
        
        # Add type-specific validation
        if slot_type == 'string':
            if 'min_length' in processed_validation:
                processed_validation['min_length'] = int(processed_validation['min_length'])
            if 'max_length' in processed_validation:
                processed_validation['max_length'] = int(processed_validation['max_length'])
        
        elif slot_type in ['number', 'integer', 'float']:
            if 'min_value' in processed_validation:
                processed_validation['min_value'] = float(processed_validation['min_value'])
            if 'max_value' in processed_validation:
                processed_validation['max_value'] = float(processed_validation['max_value'])
        
        elif slot_type == 'list':
            if 'min_items' in processed_validation:
                processed_validation['min_items'] = int(processed_validation['min_items'])
            if 'max_items' in processed_validation:
                processed_validation['max_items'] = int(processed_validation['max_items'])
        
        # Validate pattern if present
        if 'pattern' in processed_validation:
            try:
                import re
                re.compile(processed_validation['pattern'])
            except re.error as e:
                raise ValueError(f"Slot '{slot_name}' has invalid regex pattern: {e}")
        
        return processed_validation
    
    def validate_slot_value(self, slot_name: str, value: Any, slot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a slot value against its configuration"""
        result = {
            'valid': False,
            'value': value,
            'errors': [],
            'normalized_value': value
        }
        
        slot_type = slot_config.get('type', 'string')
        validation = slot_config.get('validation', {})
        
        # Check if value is required
        if slot_config.get('required', False) and (value is None or value == ''):
            result['errors'].append(f"{slot_name} is required")
            return result
        
        # If value is empty and not required, it's valid
        if value is None or value == '':
            result['valid'] = True
            return result
        
        # Type validation and normalization
        try:
            normalized_value = self._normalize_value(value, slot_type)
            result['normalized_value'] = normalized_value
        except ValueError as e:
            result['errors'].append(f"Invalid {slot_type}: {e}")
            return result
        
        # Specific validations
        validation_errors = self._validate_specific_rules(
            slot_name, normalized_value, slot_type, validation
        )
        result['errors'].extend(validation_errors)
        
        result['valid'] = len(result['errors']) == 0
        return result
    
    def _normalize_value(self, value: Any, slot_type: str) -> Any:
        """Normalize value to the expected type"""
        if slot_type == 'string':
            return str(value).strip()
        
        elif slot_type == 'integer':
            if isinstance(value, str):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            else:
                raise ValueError(f"Cannot convert {type(value)} to integer")
        
        elif slot_type == 'float':
            if isinstance(value, str):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                raise ValueError(f"Cannot convert {type(value)} to float")
        
        elif slot_type == 'number':
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return float(value)
            elif isinstance(value, (int, float)):
                return value
            else:
                raise ValueError(f"Cannot convert {type(value)} to number")
        
        elif slot_type == 'boolean':
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ['true', 'yes', '1', 'on', 'enabled']
            else:
                raise ValueError(f"Cannot convert {type(value)} to boolean")
        
        elif slot_type in ['date', 'time', 'datetime']:
            # For now, just return as string - could add datetime parsing
            return str(value).strip()
        
        elif slot_type in ['email', 'phone', 'url']:
            return str(value).strip()
        
        elif slot_type == 'list':
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                # Try to parse as JSON list or split by comma
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return [item.strip() for item in value.split(',')]
            else:
                raise ValueError(f"Cannot convert {type(value)} to list")
        
        elif slot_type == 'dict':
            if isinstance(value, dict):
                return value
            elif isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON for dict type")
            else:
                raise ValueError(f"Cannot convert {type(value)} to dict")
        
        return value
    
    def _validate_specific_rules(self, slot_name: str, value: Any, 
                                slot_type: str, validation: Dict[str, Any]) -> List[str]:
        """Validate specific rules for a slot value"""
        errors = []
        
        # Pattern validation
        if 'pattern' in validation:
            import re
            if not re.match(validation['pattern'], str(value)):
                errors.append(f"{slot_name} does not match required pattern")
        
        # String validations
        if slot_type == 'string':
            if 'min_length' in validation and len(str(value)) < validation['min_length']:
                errors.append(f"{slot_name} must be at least {validation['min_length']} characters")
            if 'max_length' in validation and len(str(value)) > validation['max_length']:
                errors.append(f"{slot_name} must be no more than {validation['max_length']} characters")
        
        # Number validations
        elif slot_type in ['number', 'integer', 'float']:
            if 'min_value' in validation and value < validation['min_value']:
                errors.append(f"{slot_name} must be at least {validation['min_value']}")
            if 'max_value' in validation and value > validation['max_value']:
                errors.append(f"{slot_name} must be no more than {validation['max_value']}")
        
        # List validations
        elif slot_type == 'list':
            if 'min_items' in validation and len(value) < validation['min_items']:
                errors.append(f"{slot_name} must have at least {validation['min_items']} items")
            if 'max_items' in validation and len(value) > validation['max_items']:
                errors.append(f"{slot_name} must have no more than {validation['max_items']} items")
        
        # Allowed values validation
        if 'allowed_values' in validation:
            allowed = validation['allowed_values']
            if value not in allowed:
                errors.append(f"{slot_name} must be one of: {', '.join(map(str, allowed))}")
        
        # Email validation
        if slot_type == 'email':
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, str(value)):
                errors.append(f"{slot_name} must be a valid email address")
        
        # Phone validation
        elif slot_type == 'phone':
            import re
            # Simple Korean phone number pattern
            phone_pattern = r'^(\+82-?|0)([0-9]{1,4}-?[0-9]{3,4}-?[0-9]{4})$'
            clean_phone = re.sub(r'[\s\-()]', '', str(value))
            if not re.match(r'^(\+82|0)[0-9]{9,11}$', clean_phone):
                errors.append(f"{slot_name} must be a valid phone number")
        
        # URL validation
        elif slot_type == 'url':
            import re
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, str(value)):
                errors.append(f"{slot_name} must be a valid URL")
        
        return errors


def load_slots(file_path: str) -> Dict[str, Any]:
    """Convenience function to load slots from file"""
    loader = SlotLoader()
    return loader.load_from_file(file_path) 