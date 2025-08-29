"""
Robust Stage Detection System - Pattern-based with clear rules and efficient processing
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import logging
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .models_simplified import NodeStage

logger = logging.getLogger(__name__)


@dataclass
class StageRule:
    """A single rule for stage detection"""
    pattern_type: str  # 'keyword', 'regex', 'structure', 'semantic'
    pattern: Any
    weight: int
    target_stage: NodeStage
    description: str


@dataclass
class StageDetectionResult:
    """Result of stage detection with confidence and reasoning"""
    stage: NodeStage
    confidence: float
    matched_rules: List[StageRule]
    reasoning: str


class StageRuleEngine:
    """Efficient rule-based stage detection engine"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.stage_priorities = self._define_stage_priorities()
    
    def _initialize_rules(self) -> List[StageRule]:
        """Initialize all stage detection rules with clear patterns"""
        rules = []
        
        # === GREETING/INITIAL STAGE RULES ===
        rules.extend([
            StageRule('structure', 'no_prev_nodes', 10, NodeStage.GREETING, 
                     "Entry point node (no previous nodes)"),
            StageRule('keyword', ['welcome', 'greeting', 'hello', '안녕', '환영'], 8, NodeStage.GREETING,
                     "Greeting keywords in name/responses"),
            StageRule('keyword', ['start', 'begin', '시작'], 6, NodeStage.GREETING,
                     "Start/begin keywords"),
        ])
        
        # === SLOT FILLING/DATA COLLECTION RULES ===
        rules.extend([
            StageRule('structure', 'has_required_slots', 12, NodeStage.SLOT_FILLING,
                     "Node has required_slots parameter"),
            StageRule('structure', 'has_slot_validation', 10, NodeStage.SLOT_FILLING,
                     "Node has slot validation rules"),
            StageRule('keyword', ['collect', 'input', 'enter', '입력', '수집'], 8, NodeStage.SLOT_FILLING,
                     "Data collection keywords"),
            StageRule('semantic', 'asks_for_information', 6, NodeStage.SLOT_FILLING,
                     "Responses ask for user information"),
        ])
        
        # === VALIDATION STAGE RULES ===
        rules.extend([
            StageRule('keyword', ['validate', 'verify', 'check', '검증', '확인'], 10, NodeStage.VALIDATION,
                     "Validation keywords"),
            StageRule('structure', 'has_validation_actions', 12, NodeStage.VALIDATION,
                     "Node has validation actions"),
            StageRule('regex', r'validate_\w+|check_\w+', 8, NodeStage.VALIDATION,
                     "Validation function patterns"),
        ])
        
        # === CONFIRMATION STAGE RULES ===
        rules.extend([
            StageRule('keyword', ['confirm', 'proceed', 'continue', '계속', '진행'], 10, NodeStage.CONFIRMATION,
                     "Confirmation keywords"),
            StageRule('semantic', 'asks_yes_no', 8, NodeStage.CONFIRMATION,
                     "Responses ask for yes/no confirmation"),
            StageRule('structure', 'before_processing', 6, NodeStage.CONFIRMATION,
                     "Node before processing nodes"),
        ])
        
        # === PROCESSING STAGE RULES ===
        rules.extend([
            StageRule('keyword', ['process', 'execute', 'submit', '처리', '실행'], 12, NodeStage.PROCESSING,
                     "Processing keywords"),
            StageRule('structure', 'has_api_actions', 10, NodeStage.PROCESSING,
                     "Node has API/external actions"),
            StageRule('keyword', ['api', 'service', 'external'], 8, NodeStage.PROCESSING,
                     "External system integration"),
        ])
        
        # === COMPLETION/FINAL STAGE RULES ===
        rules.extend([
            StageRule('structure', 'no_next_nodes', 10, NodeStage.FINAL,
                     "Terminal node (no next nodes)"),
            StageRule('keyword', ['complete', 'finish', 'done', '완료', '종료'], 12, NodeStage.FINAL,
                     "Completion keywords"),
            StageRule('keyword', ['success', 'approved', '성공', '승인'], 8, NodeStage.COMPLETION,
                     "Success completion"),
        ])
        
        # === ERROR HANDLING RULES ===
        rules.extend([
            StageRule('keyword', ['error', 'fail', 'reject', '오류', '실패'], 10, NodeStage.ERROR,
                     "Error keywords"),
            StageRule('structure', 'error_handling_node', 8, NodeStage.ERROR_HANDLING,
                     "Dedicated error handling"),
        ])
        
        return rules
    
    def _define_stage_priorities(self) -> Dict[NodeStage, int]:
        """Define priority order for conflicting stages"""
        return {
            NodeStage.ERROR: 10,
            NodeStage.PROCESSING: 9,
            NodeStage.VALIDATION: 8,
            NodeStage.CONFIRMATION: 7,
            NodeStage.SLOT_FILLING: 6,
            NodeStage.COMPLETION: 5,
            NodeStage.FINAL: 4,
            NodeStage.GREETING: 3,
            NodeStage.DEFAULT: 1
        }
    
    def detect_stage(self, node_config: Dict[str, Any]) -> StageDetectionResult:
        """Detect stage using rule engine with confidence scoring"""
        node_name = node_config.get('name', '')
        
        # Collect matching rules and scores
        matched_rules = []
        stage_scores = {}
        
        for rule in self.rules:
            if self._rule_matches(rule, node_config):
                matched_rules.append(rule)
                stage = rule.target_stage
                stage_scores[stage] = stage_scores.get(stage, 0) + rule.weight
        
        if not stage_scores:
            return StageDetectionResult(
                stage=NodeStage.DEFAULT,
                confidence=0.1,
                matched_rules=[],
                reasoning="No matching rules found"
            )
        
        # Apply priority-based resolution for ties
        best_stage = self._resolve_stage_conflicts(stage_scores)
        max_score = stage_scores[best_stage]
        
        # Calculate confidence (0.0 to 1.0)
        confidence = min(max_score / 20.0, 1.0)  # Max possible score ~20
        
        # Generate reasoning
        reasoning = self._generate_reasoning(matched_rules, best_stage, max_score)
        
        logger.debug(f"Stage detection for '{node_name}': {best_stage.value} (confidence: {confidence:.2f})")
        
        return StageDetectionResult(
            stage=best_stage,
            confidence=confidence,
            matched_rules=[r for r in matched_rules if r.target_stage == best_stage],
            reasoning=reasoning
        )
    
    def _rule_matches(self, rule: StageRule, node_config: Dict[str, Any]) -> bool:
        """Check if a rule matches the node configuration"""
        
        if rule.pattern_type == 'keyword':
            return self._match_keywords(rule.pattern, node_config)
        elif rule.pattern_type == 'regex':
            return self._match_regex(rule.pattern, node_config)
        elif rule.pattern_type == 'structure':
            return self._match_structure(rule.pattern, node_config)
        elif rule.pattern_type == 'semantic':
            return self._match_semantic(rule.pattern, node_config)
        
        return False
    
    def _match_keywords(self, keywords: List[str], node_config: Dict[str, Any]) -> bool:
        """Match keywords in node name, description, responses"""
        search_text = ' '.join([
            node_config.get('name', ''),
            node_config.get('description', ''),
            str(node_config.get('responses', {})),
            ' '.join(node_config.get('actions', []))
        ]).lower()
        
        return any(keyword.lower() in search_text for keyword in keywords)
    
    def _match_regex(self, pattern: str, node_config: Dict[str, Any]) -> bool:
        """Match regex patterns in node content"""
        search_text = ' '.join([
            node_config.get('name', ''),
            ' '.join(node_config.get('actions', []))
        ])
        
        return bool(re.search(pattern, search_text, re.IGNORECASE))
    
    def _match_structure(self, pattern: str, node_config: Dict[str, Any]) -> bool:
        """Match structural patterns"""
        
        if pattern == 'no_prev_nodes':
            return len(node_config.get('prev_nodes', [])) == 0
        
        elif pattern == 'no_next_nodes':
            return len(node_config.get('next_nodes', [])) == 0
        
        elif pattern == 'has_required_slots':
            return bool(node_config.get('params', {}).get('required_slots'))
        
        elif pattern == 'has_slot_validation':
            return bool(node_config.get('params', {}).get('validation_rules'))
        
        elif pattern == 'has_validation_actions':
            actions = node_config.get('actions', [])
            return any('validate' in action.lower() for action in actions)
        
        elif pattern == 'has_api_actions':
            actions = node_config.get('actions', [])
            return any(api_keyword in action.lower() 
                      for action in actions 
                      for api_keyword in ['api', 'request', 'call', 'http'])
        
        elif pattern == 'before_processing':
            next_nodes = node_config.get('next_nodes', [])
            return any('process' in node.lower() for node in next_nodes)
        
        elif pattern == 'error_handling_node':
            node_name = node_config.get('name', '').lower()
            return 'error' in node_name or 'exception' in node_name
        
        return False
    
    def _match_semantic(self, pattern: str, node_config: Dict[str, Any]) -> bool:
        """Match semantic patterns in responses"""
        responses = str(node_config.get('responses', {})).lower()
        
        if pattern == 'asks_for_information':
            question_indicators = ['?', '입력', '알려주', 'enter', 'provide']
            return any(indicator in responses for indicator in question_indicators)
        
        elif pattern == 'asks_yes_no':
            yn_indicators = ['예/아니오', 'y/n', '맞나요', 'correct', 'proceed']
            return any(indicator in responses for indicator in yn_indicators)
        
        return False
    
    def _resolve_stage_conflicts(self, stage_scores: Dict[NodeStage, int]) -> NodeStage:
        """Resolve conflicts when multiple stages have similar scores"""
        
        # Find stages with max score
        max_score = max(stage_scores.values())
        tied_stages = [stage for stage, score in stage_scores.items() if score == max_score]
        
        if len(tied_stages) == 1:
            return tied_stages[0]
        
        # Use priority to break ties
        return max(tied_stages, key=lambda s: self.stage_priorities.get(s, 0))
    
    def _generate_reasoning(self, matched_rules: List[StageRule], 
                          chosen_stage: NodeStage, score: int) -> str:
        """Generate human-readable reasoning for the decision"""
        
        relevant_rules = [r for r in matched_rules if r.target_stage == chosen_stage]
        
        if not relevant_rules:
            return "Default assignment due to no matching rules"
        
        reasons = [rule.description for rule in relevant_rules[:3]]  # Top 3 reasons
        
        return f"Stage '{chosen_stage.value}' chosen (score: {score}) because: " + \
               "; ".join(reasons)


class OptimizedStageManager:
    """High-performance stage manager with intelligent caching"""
    
    def __init__(self, enable_detection: bool = True):
        self.enable_detection = enable_detection
        self.rule_engine = StageRuleEngine() if enable_detection else None
        
        # Multi-level caching
        self.stage_cache = {}  # node_name -> stage_str
        self.result_cache = {}  # node_name -> StageDetectionResult
        self.pattern_cache = {}  # pattern_hash -> bool
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_node_stage(self, node_config: Dict[str, Any]) -> str:
        """Get stage for node with optimized caching"""
        
        node_name = node_config.get('name')
        if not node_name:
            return NodeStage.DEFAULT.value
        
        # Check cache first
        if node_name in self.stage_cache:
            self.cache_hits += 1
            return self.stage_cache[node_name]
        
        self.cache_misses += 1
        
        # Detect stage
        if self.rule_engine:
            result = self.rule_engine.detect_stage(node_config)
            stage_str = result.stage.value
            
            # Cache both stage and full result
            self.stage_cache[node_name] = stage_str
            self.result_cache[node_name] = result
        else:
            stage_str = NodeStage.DEFAULT.value
            self.stage_cache[node_name] = stage_str
        
        logger.debug(f"Stage detected: '{node_name}' → '{stage_str}'")
        return stage_str
    
    def get_detection_result(self, node_config: Dict[str, Any]) -> Optional[StageDetectionResult]:
        """Get detailed detection result with confidence and reasoning"""
        
        node_name = node_config.get('name')
        if not node_name or not self.rule_engine:
            return None
        
        # Check cache
        if node_name in self.result_cache:
            return self.result_cache[node_name]
        
        # Trigger detection (will cache result)
        self.get_node_stage(node_config)
        return self.result_cache.get(node_name)
    
    def validate_stage_flow(self, current_stage: str, next_stage: str) -> bool:
        """Validate if stage transition is logically valid"""
        
        valid_transitions = {
            NodeStage.GREETING.value: [NodeStage.SLOT_FILLING.value, NodeStage.INTENT_DETECTION.value],
            NodeStage.SLOT_FILLING.value: [NodeStage.VALIDATION.value, NodeStage.SLOT_FILLING.value],
            NodeStage.VALIDATION.value: [NodeStage.CONFIRMATION.value, NodeStage.SLOT_FILLING.value, NodeStage.ERROR.value],
            NodeStage.CONFIRMATION.value: [NodeStage.PROCESSING.value, NodeStage.SLOT_FILLING.value],
            NodeStage.PROCESSING.value: [NodeStage.COMPLETION.value, NodeStage.FINAL.value, NodeStage.ERROR.value],
            NodeStage.COMPLETION.value: [NodeStage.FINAL.value],
            NodeStage.ERROR.value: [NodeStage.SLOT_FILLING.value, NodeStage.FINAL.value]
        }
        
        allowed_next = valid_transitions.get(current_stage, [next_stage])  # Allow if not defined
        return next_stage in allowed_next
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get caching and performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cached_nodes': len(self.stage_cache),
            'unique_stages': len(set(self.stage_cache.values()))
        }
    
    def clear_cache(self):
        """Clear all caches and reset metrics"""
        self.stage_cache.clear()
        self.result_cache.clear()
        self.pattern_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def warm_cache(self, nodes_config: Dict[str, Any]):
        """Pre-warm cache with all nodes for better performance"""
        logger.info(f"Warming stage cache for {len(nodes_config)} nodes...")
        
        for node_name, node_config in nodes_config.items():
            self.get_node_stage(node_config)
        
        logger.info(f"Cache warmed: {len(self.stage_cache)} stages cached")


# Backward compatibility alias
StageManager = OptimizedStageManager 