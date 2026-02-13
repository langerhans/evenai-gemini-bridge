import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import google.genai as genai
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_FILE = Path(__file__).parent / "config.yaml"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.md"

# Module-level state, initialized during lifespan
client: genai.Client | None = None
gemini_model: str = ""
bearer_token: str = ""
server_port: int = 8000


def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml."""
    if not CONFIG_FILE.exists():
        logger.error("Configuration file not found: %s", CONFIG_FILE)
        logger.error("Please copy config.example.yaml to config.yaml and configure it.")
        raise RuntimeError("config.yaml not found")

    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application state on startup."""
    global client, gemini_model, bearer_token, server_port

    config = load_config()
    gemini_model = config.get("gemini_model", "gemini-2.0-flash-exp")
    bearer_token = config.get("token", "")
    server_port = config.get("server", {}).get("port", 8000)

    api_key = config.get("gemini_api_key")
    if not api_key or api_key == "your-api-key-here":
        logger.warning("GEMINI_API_KEY not configured properly. API calls will fail.")
    else:
        client = genai.Client(api_key=api_key)
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)
        logger.info("Gemini client initialized with model: %s", gemini_model)

    yield

    client = None


app = FastAPI(title="EvenAI Gemini Bridge", version="0.1.0", lifespan=lifespan)

# Track last request for deduplication
_last_request: dict[str, Any] = {}

# Duplicate requests arriving within this window are silently discarded.
# The device sometimes fires the same request twice in quick succession.
DEDUP_WINDOW_S = 0.5


def read_system_prompt() -> str:
    """Read the system prompt from the markdown file."""
    try:
        return SYSTEM_PROMPT_FILE.read_text().strip()
    except FileNotFoundError:
        logger.warning("System prompt file not found: %s", SYSTEM_PROMPT_FILE)
        return "You are a helpful AI assistant."


def convert_messages_to_gemini(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI format messages to Gemini format."""
    gemini_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system":
            # System messages are handled separately in the new API
            continue
        elif role == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": content}]})
    
    return gemini_messages


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """
    OpenAI-compatible chat completions endpoint.
    Forwards requests to Google Gemini API.
    """
    request_time = time.monotonic()

    # Verify bearer token
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header[7:] != bearer_token:
        return JSONResponse(
            status_code=200,
            content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "error",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Authorization failed. Invalid or missing token.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    body = await request.json()
    
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    try:
        # Read system prompt
        system_prompt = read_system_prompt()
        
        # Extract messages from request
        messages = body.get("messages", [])
        
        # Log the last user message
        last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user_message:
            content = last_user_message.get("content")
            logger.info("User: %s", content)
            
            # Discard duplicate: same message arriving within the dedup window
            last = _last_request.get("content")
            last_time = _last_request.get("time", 0.0)
            delta = request_time - last_time
            if content == last and delta < DEDUP_WINDOW_S:
                logger.warning("Duplicate request discarded (%.2fs apart)", delta)
                return JSONResponse(
                    status_code=200,
                    content={
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": body.get("model", gemini_model),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            _last_request["content"] = content
            _last_request["time"] = request_time
        
        # Convert to Gemini format
        gemini_contents = convert_messages_to_gemini(messages)
        
        # Generate response using new API
        response = await client.aio.models.generate_content(
            model=gemini_model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        
        # Extract response text
        response_text = getattr(response, "text", "") or ""
        
        logger.info("Model: %s", response_text)
        
        # Convert response to OpenAI format
        return JSONResponse(
            status_code=200,
            content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", gemini_model),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    
    except Exception:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def main() -> None:
    """Run the server."""
    config = load_config()
    port = config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
