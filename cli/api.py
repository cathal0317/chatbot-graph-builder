from __future__ import annotations

import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from core.runtime.graph_info import load_and_validate
from core.dst_manager import DSTManager
from core.api import APIResponse
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.getenv("CHATBOT_CONFIG", "config/card_issuance_chatbot.json")
USE_REDIS = bool(int(os.getenv("USE_REDIS", "0")))

# Initialize runtime
graph_info = load_and_validate(CONFIG_PATH)
dst = DSTManager(graph_info=graph_info, use_redis=USE_REDIS)

app = FastAPI(title="Chatbot Graph Builder API")


class StartSessionReq(BaseModel):
    session_id: Optional[str] = None


class StartSessionRes(BaseModel):
    session_id: str


class SendMessageReq(BaseModel):
    session_id: str
    message: str


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/sessions", response_model=StartSessionRes)
def start_session(body: StartSessionReq) -> StartSessionRes:
    sid = dst.start_session(session_id=body.session_id)
    return StartSessionRes(session_id=sid)


@app.post("/sessions/{session_id}/messages", response_model=APIResponse)
def send_message(session_id: str, body: SendMessageReq) -> Dict[str, Any]:
    if body.session_id and body.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id mismatch")
    result = dst.process_turn(session_id=session_id, user_message=body.message)
    if result.get('error'):
        raise HTTPException(status_code=500, detail=result['response'])
    return result.get('data') or {}


@app.get("/sessions/{session_id}", response_model=APIResponse)
def get_session(session_id: str) -> Dict[str, Any]:
    result = dst.get_session_info(session_id)
    if result.get('error'):
        raise HTTPException(status_code=404, detail=result['response'])
    return result.get('data') or {}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True} 