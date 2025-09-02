"""
Fallback-only Stage Manager
- Keeps public API compatible with previous implementation
- No heavy rule detection; returns node_config's declared stage (or 'default')
"""
from typing import Dict, Any, Optional
import logging

from .models_simplified import NodeStage

logger = logging.getLogger(__name__)

class OptimizedStageManager:
    """Minimal fallback stage manager (API-compatible).
    Prefers explicit stage set on node_config; otherwise returns 'default'.
    """

    def __init__(self, enable_detection: bool = True):
        # Flag kept for compatibility; not used in fallback mode
        self.enable_detection = enable_detection

    # Cache and stats kept for compatibility
    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'cache_hits': 0,
            'cache_misses': 0,
            'hit_rate': "0.00%",
            'cached_nodes': 0,
            'unique_stages': 0
        }

    def clear_cache(self):
        # No-op in fallback mode
        pass

    def warm_cache(self, nodes_config: Dict[str, Any]):
        # No-op in fallback mode
        pass

    def get_node_stage(self, node_config: Dict[str, Any]) -> str:
        """Return node_config's stage (string) or 'default' if missing.
        Maps common aliases via NodeStage.from_string for consistency.
        """
        try:
            raw = (node_config.get('stage') or 'default')
            if isinstance(raw, str):
                stage_enum = NodeStage.from_string(raw.lower())
            else:
                # If enum already provided
                stage_enum = raw if isinstance(raw, NodeStage) else NodeStage.DEFAULT
            return stage_enum.value
        except Exception as e:
            logger.debug(f"Fallback stage resolution error: {e}")
            return NodeStage.DEFAULT.value

    def get_detection_result(self, node_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """No rule-based details in fallback mode."""
        return None

    def validate_stage_flow(self, current_stage: str, next_stage: str) -> bool:
        """Always allow transitions in fallback mode (keep simple)."""
        return True

# Backward compatibility alias
StageManager = OptimizedStageManager 