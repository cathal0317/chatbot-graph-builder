from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime, timedelta
from core.models_simplified import DialogueState

logger = logging.getLogger(__name__)

class ContextStore:
    """Context store for managing dialogue states"""
    
    def __init__(self, use_redis: bool = False, redis_host: str = 'localhost', 
                 redis_port: int = 6379, redis_db: int = 0, 
                 session_ttl: int = 3600):  # 1 hour TTL
        self.use_redis = use_redis
        self.session_ttl = session_ttl
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    db=redis_db,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis for context storage")
            except ImportError:
                logger.warning("Redis not available, falling back to in-memory storage")
                self.use_redis = False
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to in-memory storage")
                self.use_redis = False
        
        # In-memory storage as fallback
        if not self.use_redis:
            self.memory_store = {}
            logger.info("Using in-memory storage for context")
    
    def save_state(self, session_id: str, dialogue_state: DialogueState) -> bool:
        """Save dialogue state"""
        try:
            state_data = dialogue_state.to_dict()
            
            if self.use_redis and self.redis_client:
                # Save to Redis with TTL
                key = f"session:{session_id}"
                self.redis_client.setex(
                    key, 
                    self.session_ttl, 
                    json.dumps(state_data, ensure_ascii=False)
                )
            else:
                # Save to memory with expiration check
                self.memory_store[session_id] = {
                    'data': state_data,
                    'expires_at': datetime.now() + timedelta(seconds=self.session_ttl)
                }
            
            logger.debug(f"Saved state for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state for session {session_id}: {e}")
            return False
    
    def load_state(self, session_id: str) -> Optional[DialogueState]:
        """Load dialogue state"""
        try:
            if self.use_redis and self.redis_client:
                # Load from Redis
                key = f"session:{session_id}"
                data = self.redis_client.get(key)
                if data:
                    state_data = json.loads(data)
                    return DialogueState.model_validate(state_data)
            else:
                # Load from memory with expiration check
                session_data = self.memory_store.get(session_id)
                if session_data:
                    if datetime.now() < session_data['expires_at']:
                        return DialogueState.model_validate(session_data['data'])
                    else:
                        # Session expired, remove it
                        del self.memory_store[session_id]
                        logger.debug(f"Session {session_id} expired and removed")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load state for session {session_id}: {e}")
            return None
    
    def delete_state(self, session_id: str) -> bool:
        """Delete dialogue state"""
        try:
            if self.use_redis and self.redis_client:
                # Delete from Redis
                key = f"session:{session_id}"
                self.redis_client.delete(key)
            else:
                # Delete from memory
                self.memory_store.pop(session_id, None)
            
            logger.debug(f"Deleted state for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete state for session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[str]:
        """List all active sessions"""
        try:
            if self.use_redis and self.redis_client:
                # Get all session keys from Redis
                keys = self.redis_client.keys("session:*")
                return [key.replace("session:", "") for key in keys]
            else:
                # Get all sessions from memory (with expiration check)
                current_time = datetime.now()
                active_sessions = []
                expired_sessions = []
                
                for session_id, session_data in self.memory_store.items():
                    if current_time < session_data['expires_at']:
                        active_sessions.append(session_id)
                    else:
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    del self.memory_store[session_id]
                
                return active_sessions
                
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def cleanup_expired(self) -> int:
        """Clean up expired sessions (mainly for memory store)"""
        if self.use_redis:
            # Redis handles TTL automatically
            return 0
        
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self.memory_store.items():
                if current_time >= session_data['expires_at']:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.memory_store[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information without loading full state"""
        try:
            if self.use_redis and self.redis_client:
                key = f"session:{session_id}"
                ttl = self.redis_client.ttl(key)
                if ttl > 0:
                    return {
                        'session_id': session_id,
                        'ttl_seconds': ttl,
                        'expires_at': datetime.now() + timedelta(seconds=ttl)
                    }
            else:
                session_data = self.memory_store.get(session_id)
                if session_data:
                    if datetime.now() < session_data['expires_at']:
                        return {
                            'session_id': session_id,
                            'expires_at': session_data['expires_at'],
                            'ttl_seconds': int((session_data['expires_at'] - datetime.now()).total_seconds())
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session info for {session_id}: {e}")
            return None 