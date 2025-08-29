from openai import AzureOpenAI
import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Azure OpenAI thin wrapper (chat-completion only)"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-06-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        
        # Validate required environment variables
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        if not os.getenv("AZURE_OPENAI_KEY"):
            raise ValueError("AZURE_OPENAI_KEY environment variable is required")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Basic chat completion"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.3),
                max_tokens=kwargs.get("max_tokens", 256),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise

    def chat_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat completion with JSON response format for NLU"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 512),
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content}")
            raise
        except Exception as e:
            logger.error(f"JSON chat completion failed: {e}")
            raise

    def extract_intent_entities(self, user_message: str, node_config: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract intent and entities using NLU"""
        system_prompt = """당신의 목적은 사용자와 전화 통화를 하는 상황 속에서 사용자 발화의 의도를 찾아내 정답을 맞추는 것입니다. 사용자 발화 속 숨은 의도나 완곡한 표현을 심사숙고하여 정확히 추론해야 합니다.
Given the user message and node configuration, extract the user's intent and any relevant entities.

Return a JSON object with the following structure:
{
    "intent": "detected_intent_name",
    "entities": {
        "entity_name": "entity_value"
    },
    "confidence": 0.95
}

If the intent is unclear, set confidence below 0.7."""
        
        user_prompt = f"""
User message: {user_message}

Node context: {json.dumps(node_config, ensure_ascii=False, indent=2)}

Conversation context: {json.dumps(context or {}, ensure_ascii=False, indent=2)}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.chat_json(messages, temperature=0.1)

    def generate_response(self, context: Dict[str, Any], node_config: Dict[str, Any], 
                         intent_data: Dict[str, Any] = None) -> str:
        """Generate natural language response using NLG"""
        
        # Enhanced system prompt for more natural conversation
        system_prompt = """당신은 도움이 되고 자연스러운 한국어 챗봇 어시스턴트입니다.

다음 지침을 따라 응답하세요:
1. 자연스럽고 대화적인 한국어로 응답하세요
2. 사용자의 입력에 적절히 반응하며 공감을 표현하세요
3. 현재 진행 중인 단계(노드)의 목적에 맞게 대화를 이어가세요
4. 필요한 정보가 있으면 자연스럽게 요청하세요
5. 사용자가 주제에서 벗어나면 친절하게 안내하세요
6. 짧고 명확하되 따뜻한 톤을 유지하세요"""
        
        # Build comprehensive user prompt
        user_message = context.get('user_message', '')
        node_purpose = context.get('node_purpose', '')
        turn_count = context.get('turn_count', 0)
        current_slots = {k: v for k, v in context.items() if k not in ['user_message', 'node_purpose', 'turn_count', 'current_node', 'guidance_needed']}
        
        # Special handling for different scenarios
        scenario_info = ""
        if intent_data:
            if intent_data.get('intent') == 'off_topic':
                scenario_info = "\n주의: 사용자가 현재 단계와 관련 없는 말을 했습니다. 친절하게 현재 단계로 안내해주세요."
            elif 'missing_slots' in intent_data:
                missing = intent_data['missing_slots']
                scenario_info = f"\n아직 필요한 정보: {', '.join(missing)}. 자연스럽게 이 정보를 요청해주세요."
            elif intent_data.get('all_slots_filled'):
                scenario_info = "\n모든 필요한 정보가 수집되었습니다. 확인 및 다음 단계 안내를 해주세요."
        
        user_prompt = f"""
사용자 입력: "{user_message}"

현재 상황:
- 진행 중인 단계: {node_purpose}
- 대화 턴: {turn_count}번째
- 수집된 정보: {json.dumps(current_slots, ensure_ascii=False)}
- 의도/엔티티: {json.dumps(intent_data or {}, ensure_ascii=False)}

노드 설정: {json.dumps(node_config, ensure_ascii=False, indent=2)}
{scenario_info}

위 정보를 바탕으로 사용자에게 자연스럽고 도움이 되는 한국어 응답을 생성해주세요.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.chat(messages, temperature=0.1) 