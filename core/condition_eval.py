from typing import Dict, Any, List, Optional
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConditionEvaluator:
    #정의해 놓은 conditions 체킹 연산자
    
    def __init__(self):
        self.operators = {
            'equals': self._equals,
            'not_equals': self._not_equals,
            'greater_than': self._greater_than,
            'less_than': self._less_than,
            'contains': self._contains,
            'not_contains': self._not_contains,
            'regex_match': self._regex_match,
            'exists': self._exists,
            'not_exists': self._not_exists,
            'in_list': self._in_list,
            'not_in_list': self._not_in_list,
            'length_equals': self._length_equals,
            'length_greater': self._length_greater,
            'length_less': self._length_less,
            'is_empty': self._is_empty,
            'is_not_empty': self._is_not_empty
        }
    
    def evaluate_conditions(self, conditions: Dict[str, Any], 
                          context: Dict[str, Any], 
                          intent_data: Dict[str, Any] = None) -> str:
        """
        Evaluate node conditions and return the next node name
        
        Args:
            conditions: Dictionary of condition_name -> condition_rules or next_node
            context: Current dialogue context including slots
            intent_data: Intent and entities extracted from user input
            
        Returns:
            Next node name or None if no conditions match
        """
        if not conditions:
            return None
            
        # 추출한 데이터 모음
        eval_context = {
            **context,
            'intent': intent_data.get('intent') if intent_data else None,
            'entities': intent_data.get('entities', {}) if intent_data else {},
            'confidence': intent_data.get('confidence') if intent_data else 0.0
        }
        
        # 각 조건 체킹
        for condition_name, condition_rule in conditions.items():
            try:
                if self._evaluate_single_condition(condition_rule, eval_context):
                    logger.debug(f"Condition '{condition_name}' matched")
                    # 조건 체킹 결과가 참인 경우 다음 노드 반환
                    if isinstance(condition_rule, str):
                        return condition_rule
                    else:
                        return condition_name
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition_name}': {e}")
                continue
        
        # 조건 체킹 결과가 거짓인 경우 None 반환
        logger.debug("No conditions matched")
        return None
        
    def _evaluate_single_condition(self, condition_rule: Any, context: Dict[str, Any]) -> bool:
        """단일 조건 체킹"""
        
        # Simple string condition 
        if isinstance(condition_rule, str):
            return True
        
        # Dictionary condition with operators
        if isinstance(condition_rule, dict):
            return self._evaluate_dict_condition(condition_rule, context)
        
        # List condition (OR logic)
        if isinstance(condition_rule, list):
            return any(self._evaluate_single_condition(rule, context) for rule in condition_rule)
        
        # Boolean condition
        if isinstance(condition_rule, bool):
            return condition_rule
            
        return False
    
    def _evaluate_dict_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """딕셔너리 형태 조건 체킹"""
        
        # 특별한 조건 타입1ㅂ
        if 'and' in condition:
            return all(self._evaluate_single_condition(rule, context) 
                      for rule in condition['and'])
        
        if 'or' in condition:
            return any(self._evaluate_single_condition(rule, context) 
                      for rule in condition['or'])
        
        if 'not' in condition:
            return not self._evaluate_single_condition(condition['not'], context)
        
        # Field-based condition
        field = condition.get('field')
        operator = condition.get('operator', 'equals')
        value = condition.get('value')
        
        if not field:
            return False
        
        field_value = self._get_nested_value(context, field)
        
        if operator not in self.operators:
            logger.warning(f"Unknown operator: {operator}")
            return False
        
        return self.operators[operator](field_value, value)
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """데이터 조회"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    # 사용 가능 연산자들
    def _equals(self, field_value: Any, target_value: Any) -> bool:
        return field_value == target_value
    
    def _not_equals(self, field_value: Any, target_value: Any) -> bool:
        return field_value != target_value
    
    def _greater_than(self, field_value: Any, target_value: Any) -> bool:
        try:
            return float(field_value) > float(target_value)
        except (ValueError, TypeError):
            return False
    
    def _less_than(self, field_value: Any, target_value: Any) -> bool:
        try:
            return float(field_value) < float(target_value)
        except (ValueError, TypeError):
            return False
    
    def _contains(self, field_value: Any, target_value: Any) -> bool:
        if field_value is None:
            return False
        return str(target_value) in str(field_value)
    
    def _not_contains(self, field_value: Any, target_value: Any) -> bool:
        return not self._contains(field_value, target_value)
    
    def _regex_match(self, field_value: Any, pattern: str) -> bool:
        if field_value is None:
            return False
        try:
            return bool(re.search(pattern, str(field_value)))
        except re.error:
            return False
    
    def _exists(self, field_value: Any, _: Any = None) -> bool:
        return field_value is not None
    
    def _not_exists(self, field_value: Any, _: Any = None) -> bool:
        return field_value is None
    
    def _in_list(self, field_value: Any, target_list: List[Any]) -> bool:
        if not isinstance(target_list, list):
            return False
        return field_value in target_list
    
    def _not_in_list(self, field_value: Any, target_list: List[Any]) -> bool:
        return not self._in_list(field_value, target_list)
    
    def _length_equals(self, field_value: Any, target_length: int) -> bool:
        if field_value is None:
            return False
        try:
            return len(field_value) == int(target_length)
        except (TypeError, ValueError):
            return False
    
    def _length_greater(self, field_value: Any, target_length: int) -> bool:
        if field_value is None:
            return False
        try:
            return len(field_value) > int(target_length)
        except (TypeError, ValueError):
            return False
    
    def _length_less(self, field_value: Any, target_length: int) -> bool:
        if field_value is None:
            return False
        try:
            return len(field_value) < int(target_length)
        except (TypeError, ValueError):
            return False
    
    def _is_empty(self, field_value: Any, _: Any = None) -> bool:
        if field_value is None:
            return True
        if isinstance(field_value, (str, list, dict)):
            return len(field_value) == 0
        return False
    
    def _is_not_empty(self, field_value: Any, _: Any = None) -> bool:
        return not self._is_empty(field_value) 