from __future__ import annotations
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Annotated
from uuid import uuid4
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


# ============================================================================
# Base Configuration
# ============================================================================

class BaseConfig(BaseModel):
    """Base configuration with common settings"""
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )


# ============================================================================
# Node Stage Definitions
# ============================================================================

class NodeStage(str, Enum):
    """Node stages for chatbot flow control"""
    # 핵심 stages
    INITIAL = "initial"
    GREETING = "greeting"
    SLOT_FILLING = "slot_filling"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"
    PROCESSING = "processing"
    FINAL = "final"
    ERROR = "error"
    DEFAULT = "default"
    
    # Extended stages
    GENERAL_CHAT = "general_chat"
    INTENT_DETECTION = "intent_detection"
    DATA_COLLECTION = "data_collection"
    API_INTEGRATION = "api_integration"
    DECISION = "decision"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"
    
    @classmethod
    def from_string(cls, stage_str: str) -> 'NodeStage':
        """Convert string to NodeStage with fallback mapping"""
        try:
            return cls(stage_str)
        except ValueError:
            # Alias mapping for common variations
            aliases = {
                "start": cls.INITIAL,
                "nlu": cls.INTENT_DETECTION,
                "form_filling": cls.SLOT_FILLING,
                "input_collection": cls.DATA_COLLECTION,
                "selection": cls.DATA_COLLECTION,
                "api_call": cls.API_INTEGRATION,
                "integration": cls.API_INTEGRATION,
                "end": cls.FINAL,
                "flow_start": cls.GREETING,
                "context_management": cls.PROCESSING,
                "session_management": cls.PROCESSING,
            }
            return aliases.get(stage_str, cls.DEFAULT)


# ============================================================================
# Slot Management
# ============================================================================

class SlotValue(BaseConfig):
    """Slot value with metadata"""
    value: Any
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    source: str = "user"
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def is_empty(self) -> bool:
        """Check if slot value is empty"""
        return self.value is None or (isinstance(self.value, str) and not self.value.strip())


# ============================================================================
# Dialogue State
# ============================================================================

class DialogueState(BaseConfig):
    """Main dialogue state container"""
    # Session info
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    current_node: Optional[str] = None
    previous_node: Optional[str] = None
    
    # Data containers
    slots: Dict[str, SlotValue] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Counters
    turn_count: Annotated[int, Field(ge=0)] = 0
    error_count: Annotated[int, Field(ge=0)] = 0
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # Status
    is_complete: bool = False
    
    # ========================================
    # Node Management
    # ========================================
    
    def update_node(self, node_name: str, context_data: Dict[str, Any] = None):
        """Update current node and optional context"""
        self.previous_node = self.current_node
        self.current_node = node_name
        self.last_updated = datetime.now()
        
        if context_data:
            self.context.update(context_data)
    
    # ========================================
    # Slot Management
    # ========================================
    
    def set_slot(self, slot_name: str, value: Any, confidence: float = 1.0, source: str = "user"):
        """Set a slot value with metadata"""
        self.slots[slot_name] = SlotValue(
            value=value,
            confidence=confidence,
            source=source
        )
        self.last_updated = datetime.now()
    
    def get_slot(self, slot_name: str) -> Any:
        """Get slot value"""
        slot_obj = self.slots.get(slot_name)
        return slot_obj.value if slot_obj else None
    
    def get_slot_confidence(self, slot_name: str) -> float:
        """Get slot confidence score"""
        slot_obj = self.slots.get(slot_name)
        return slot_obj.confidence if slot_obj else 0.0
    
    def has_slot(self, slot_name: str) -> bool:
        """Check if slot exists and has non-empty value"""
        slot_obj = self.slots.get(slot_name)
        return slot_obj is not None and not slot_obj.is_empty()
    
    def clear_slot(self, slot_name: str):
        """Remove a specific slot"""
        if slot_name in self.slots:
            del self.slots[slot_name]
            self.last_updated = datetime.now()
    
    def get_filled_slots(self) -> Dict[str, Any]:
        """Get all non-empty slots as simple dict"""
        return {
            name: slot.value 
            for name, slot in self.slots.items() 
            if not slot.is_empty()
        }
    
    # ========================================
    # Session Management
    # ========================================
    
    def increment_turn(self):
        """Increment turn counter"""
        self.turn_count += 1
        self.last_updated = datetime.now()
    
    def set_complete(self, complete: bool = True):
        """Mark session as complete or incomplete"""
        self.is_complete = complete
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        return self.model_dump()


# ============================================================================
# Node Configuration
# ============================================================================

class NodeConfig(BaseConfig):
    """Node configuration model"""
    # Required fields
    name: str
    stage: Union[str, NodeStage] = NodeStage.DEFAULT
    description: str
    
    # Optional metadata
    ko_name: Optional[str] = None
    visible: bool = True
    
    # Flow control
    prev_nodes: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Content
    responses: Dict[str, str] = Field(default_factory=dict)
    actions: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('stage', mode='before')
    @classmethod
    def validate_stage(cls, v):
        """Convert string stages to NodeStage enum"""
        if isinstance(v, str):
            return NodeStage.from_string(v)
        return v
    
    @property
    def has_conditions(self) -> bool:
        """Check if node has transition conditions"""
        return bool(self.conditions)


# ============================================================================
# Execution Result
# ============================================================================

class ExecutionResult(BaseConfig):
    """Result of executor operation"""
    # Basic result
    success: bool = True
    response: str
    next_node: Optional[str] = None
    
    # State updates
    context_updates: Dict[str, Any] = Field(default_factory=dict)
    slot_updates: Dict[str, Any] = Field(default_factory=dict)
    
    # Additional data
    intent_data: Optional[Dict[str, Any]] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Utility Functions
# ============================================================================

def create_node_config(name: str, config_dict: Dict[str, Any]) -> NodeConfig:
    """Create NodeConfig from dictionary with validation"""
    config = config_dict.copy()
    config['name'] = name
    
    # Set defaults for required fields
    config.setdefault('description', f"Node: {name}")
    config.setdefault('stage', 'default')
    
    return NodeConfig(**config)


def create_dialogue_state(session_id: Optional[str] = None) -> DialogueState:
    """Create new DialogueState with optional session ID"""
    return DialogueState(session_id=session_id or str(uuid4())) 