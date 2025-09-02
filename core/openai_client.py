from openai import AzureOpenAI
import os
import json
import logging
from typing import List, Dict, Any
from .nlu.prompts import Prompts 

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-06-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        
        # 환경 변수 설정
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
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content}")
            raise
        except Exception as e:
            logger.error(f"JSON chat completion failed: {e}")
            raise

    def extract_intent_entities(self, user_message: str, node_config: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract intent, entities, and stage in ONE pass using strict JSON schema."""
        
        # Use centralized prompts
        system_prompt = Prompts.extract_intent_entities_prompt
        user_prompt = Prompts.build_nlu_user_prompt(user_message, node_config, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        data: Dict[str, Any] = self.chat_json(messages, temperature=0.1)

        # Safe post-processing: ensure required keys exist and types are sane
        if not isinstance(data, dict):
            raise ValueError("NLU response is not a JSON object")

        data.setdefault("intent", "unknown")
        data.setdefault("stage", "fallback")
        data.setdefault("entities", {})
        data.setdefault("missing_slots", [])
        data.setdefault("all_slots_filled", False)
        data.setdefault("confidence", 0.5)

        # Type coercion / validation
        if not isinstance(data["entities"], dict):
            data["entities"] = {}
        if not isinstance(data["missing_slots"], list):
            data["missing_slots"] = []
        if not isinstance(data["all_slots_filled"], bool):
            data["all_slots_filled"] = (len(data["missing_slots"]) == 0)
        try:
            data["confidence"] = float(data["confidence"])
        except Exception:
            data["confidence"] = 0.5

        # Normalize stage names to internal set
        stage_raw = str(data.get("stage", "")).lower().strip()
        stage_map = {
            "start": "greetings",
            "initial": "greetings",
            "greeting": "greetings",
            "info_collection": "slot_filling",
            "slot_filling": "slot_filling",
            "confirmation": "confirmation",
            "final": "completion",
            "completion": "completion",
            "general": "general_chat",
            "general_chat": "general_chat",
            "fallback": "general_chat",
        }
        if stage_raw in stage_map:
            data["stage"] = stage_map[stage_raw]
        elif stage_raw:
            data["stage"] = stage_raw
        else:
            data["stage"] = "general_chat"

        # Off-topic heuristic: if intent looks unrelated and no entities/slots updated
        intent_name = str(data.get("intent", "")).lower()
        if intent_name in ("off_topic", "chitchat", "small_talk", "weather_inquiry"):
            data["intent"] = "off_topic"
            data["stage"] = "general_chat"

        return data

    def generate_response(self, context: Dict[str, Any], node_config: Dict[str, Any], 
                         intent_data: Dict[str, Any] = None, user_input_system_prompt: str = None) -> str:
        """Generate natural language response using NLG"""
        
        # Use centralized prompts and fix string concatenation bug
        system_prompt = Prompts.generate_response_system_prompt
        if user_input_system_prompt:
            system_prompt += "\n" + user_input_system_prompt
        
        # Use centralized user prompt builder
        user_prompt = Prompts.build_nlg_user_prompt(context, node_config, intent_data)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.chat(messages, temperature=0.1) 