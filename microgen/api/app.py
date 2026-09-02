"""FastAPI server module implementing OpenAI-compatible HTTP endpoints and SSE streaming."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


from microgen.scheduler.queue import Request


# --- OpenAI API Schemas ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "microgen-model"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "microgen-model"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    stream: bool = False


def create_app(
    scheduler: Any = None,
    tokenizer: Any = None,
) -> FastAPI:
    """Create and configure FastAPI web application instance."""
    app = FastAPI(title="MicroGen Inference Server", version="0.1.0")

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "microgen-model",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "microgen",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        if scheduler is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Inference engine backend not initialized.")

        # Format messages into single prompt text
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages]) + "\nassistant:"
        token_ids = tokenizer.encode(prompt)

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        req = Request(
            request_id=req_id,
            prompt=prompt,
            prompt_ids=token_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        scheduler.add_request(req)

        if not request.stream:
            # Synchronous wait for completion
            while not req.is_finished:
                await asyncio.sleep(0.01)

            generated_text = tokenizer.decode(req.generated_token_ids)
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": generated_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": len(req.generated_token_ids),
                    "total_tokens": len(token_ids) + len(req.generated_token_ids),
                },
            }
        else:

            async def event_generator():
                last_idx = 0
                while not req.is_finished or last_idx < len(req.generated_token_ids):
                    if last_idx < len(req.generated_token_ids):
                        new_tokens = req.generated_token_ids[last_idx:]
                        last_idx = len(req.generated_token_ids)
                        text_chunk = tokenizer.decode(new_tokens)
                        chunk_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text_chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"
                    await asyncio.sleep(0.01)

                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> Any:
        if scheduler is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Inference engine backend not initialized.")

        prompt_str = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        token_ids = tokenizer.encode(prompt_str)

        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        req = Request(
            request_id=req_id,
            prompt=prompt_str,
            prompt_ids=token_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        scheduler.add_request(req)

        if not request.stream:
            while not req.is_finished:
                await asyncio.sleep(0.01)

            generated_text = tokenizer.decode(req.generated_token_ids)
            return {
                "id": req_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": generated_text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": len(req.generated_token_ids),
                    "total_tokens": len(token_ids) + len(req.generated_token_ids),
                },
            }
        else:

            async def event_generator():
                last_idx = 0
                while not req.is_finished or last_idx < len(req.generated_token_ids):
                    if last_idx < len(req.generated_token_ids):
                        new_tokens = req.generated_token_ids[last_idx:]
                        last_idx = len(req.generated_token_ids)
                        text_chunk = tokenizer.decode(new_tokens)
                        chunk_payload = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": text_chunk,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"
                    await asyncio.sleep(0.01)

                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
