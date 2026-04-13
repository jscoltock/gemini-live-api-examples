import asyncio
import base64
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive
from tools import TOOL_DECLARATIONS, TOOL_MAPPING, set_notification_channel, task_manager, AGENTS
import agent_config
from ollama_tools import list_tools as list_ollama_tools
from chat_providers import stream_chat, list_models as list_chat_models

# Load environment variables
load_dotenv()

# Configure logging — force=True to override any prior basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("server.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)
# Suppress the firehose of raw API responses
logging.getLogger("gemini_live").setLevel(logging.INFO)
logging.getLogger("tools").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration — supports multiple API keys for fallback
# Set GEMINI_API_KEY_2 (or GEMINI_API_KEYS=key1,key2) in .env
_raw_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not _raw_keys:
    _single = os.getenv("GEMINI_API_KEY", "")
    if _single:
        _raw_keys = [_single]
_extra = os.getenv("GEMINI_API_KEY_2", "")
if _extra and _extra not in _raw_keys:
    _raw_keys.append(_extra)
if not _raw_keys:
    raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY or GEMINI_API_KEYS in .env")
GEMINI_API_KEYS = _raw_keys
logger.info(f"Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")

# --- Session usage tracker ---
# Accumulates token counts per WebSocket session, resets on new connection.
_session_usage = {
    "turns": 0,
    "prompt_tokens": 0,
    "response_tokens": 0,
    "total_tokens": 0,
    "cached_tokens": 0,
    "thoughts_tokens": 0,
    "model": MODEL,
}
# Pricing per million tokens (Gemini 3.1 Flash Live preview rates)
_INPUT_PRICE_PER_M = 0.30   # $/M input tokens
_OUTPUT_PRICE_PER_M = 2.50  # $/M output tokens


def _reset_session_usage():
    global _session_usage
    _session_usage = {
        "turns": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "thoughts_tokens": 0,
        "model": MODEL,
    }


def _accumulate_usage(usage_data: dict):
    """Called from WS handler with the latest usage snapshot from Gemini."""
    global _session_usage
    # Don't increment turns here — usage events fire per-chunk, not per-turn.
    # Turns are tracked client-side via turn_complete events.
    # Gemini Live usage_metadata is cumulative per-session,
    # so we take the latest value (not add to running total).
    _session_usage["prompt_tokens"] = usage_data.get("prompt_token_count", _session_usage["prompt_tokens"])
    _session_usage["response_tokens"] = usage_data.get("response_token_count", _session_usage["response_tokens"])
    _session_usage["total_tokens"] = usage_data.get("total_token_count", _session_usage["total_tokens"])
    _session_usage["cached_tokens"] = usage_data.get("cached_content_token_count", _session_usage["cached_tokens"])
    _session_usage["thoughts_tokens"] = usage_data.get("thoughts_token_count", _session_usage["thoughts_tokens"])


# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/api/tasks")
async def get_tasks():
    """Return current task list for the agent panel."""
    return task_manager.list_tasks()


@app.get("/api/agents")
async def get_agents():
    """Return agent configs (name, backend, model, timeout) for the UI."""
    return agent_config.list_agents()


@app.post("/api/agents/reload")
async def reload_agents():
    """Hot-reload agents from agents.yaml without server restart."""
    names = agent_config.reload_agents()
    return {"agents": names, "count": len(names)}


@app.get("/api/agents/{name}")
async def get_agent(name: str):
    """Return a single agent's full config."""
    agent = agent_config.get_agent(name)
    if not agent:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"Agent '{name}' not found"}, status_code=404)
    return agent


@app.post("/api/agents")
async def create_agent(data: dict):
    """Create a new agent."""
    from fastapi.responses import JSONResponse
    agent, errors = agent_config.create_agent(data)
    if errors:
        return JSONResponse({"errors": errors}, status_code=400)
    agent_config.reload_agents()
    return agent


@app.put("/api/agents/{name}")
async def update_agent(name: str, data: dict):
    """Update an existing agent."""
    from fastapi.responses import JSONResponse
    agent, errors = agent_config.update_agent(name, data)
    if errors:
        return JSONResponse({"errors": errors}, status_code=400)
    agent_config.reload_agents()
    return agent


@app.delete("/api/agents/{name}")
async def delete_agent(name: str):
    """Delete an agent."""
    from fastapi.responses import JSONResponse
    success, message = agent_config.delete_agent(name)
    if not success:
        return JSONResponse({"error": message}, status_code=404)
    agent_config.reload_agents()
    return {"message": message}


@app.get("/api/ollama-tools")
async def get_ollama_tools():
    """Return all registered Ollama tool names (for the UI agent editor)."""
    return {"tools": list_ollama_tools()}


@app.get("/api/usage")
async def get_usage():
    """Return accumulated token usage and estimated cost for this session."""
    prompt = _session_usage["prompt_tokens"]
    response = _session_usage["response_tokens"]
    # Subtract cached tokens from prompt to avoid double-counting
    billable_input = max(0, prompt - _session_usage["cached_tokens"])
    est_cost = (billable_input * _INPUT_PRICE_PER_M + response * _OUTPUT_PRICE_PER_M) / 1_000_000
    return {
        **_session_usage,
        "billable_input_tokens": billable_input,
        "estimated_cost_usd": round(est_cost, 4),
        "input_price_per_m": _INPUT_PRICE_PER_M,
        "output_price_per_m": _OUTPUT_PRICE_PER_M,
    }


@app.post("/api/usage/reset")
async def reset_usage():
    """Reset the session usage counters."""
    _reset_session_usage()
    return {"message": "Usage counters reset"}


@app.get("/api/gemini-config")
async def get_gemini_config():
    """Return the Gemini Live session config (model, voice, system_prompt)."""
    return agent_config.get_gemini_session()


@app.put("/api/gemini-config")
async def update_gemini_config(data: dict):
    """Update the Gemini Live session config (system_prompt, tools, voice, model).
    Changes take effect on the next WebSocket connection."""
    from fastapi.responses import JSONResponse
    try:
        updated = agent_config.update_gemini_session(data)
        # Rebuild Gemini tool declarations so next WS connection picks up changes
        import tools
        tools.TOOL_DECLARATIONS, tools.TOOL_MAPPING = tools.build_gemini_tools()
        return updated
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/models")
async def get_models():
    """Return available models (Live and non-Live) for the UI selector."""
    return {"models": list_chat_models()}


@app.post("/api/chat")
async def chat_endpoint(data: dict):
    """SSE streaming chat endpoint for non-Live models."""
    model_id = data.get("model", "")
    messages = data.get("messages", [])
    if not model_id or not messages:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "model and messages are required"}, status_code=400)

    async def event_stream():
        try:
            async for chunk in stream_chat(model_id, messages):
                # SSE format: data: {...}\n\n
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Gemini Live."""
    await websocket.accept()

    logger.info("WebSocket connection accepted")
    _reset_session_usage()

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()
    notification_queue = asyncio.Queue()
    file_input_queue = asyncio.Queue()

    async def audio_output_callback(data):
        await websocket.send_bytes(data)

    async def audio_interrupt_callback():
        pass

    async def receive_from_client():
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                        elif isinstance(payload, dict) and payload.get("type") == "file":
                            file_data = base64.b64decode(payload["data"])
                            await file_input_queue.put({
                                "data": file_data,
                                "mime_type": payload.get("mime_type", "application/octet-stream"),
                                "file_name": payload.get("file_name", "unknown"),
                            })
                            logger.info(f"Received file from client: {payload.get('file_name')} ({len(file_data)} bytes)")
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")

    receive_task = asyncio.create_task(receive_from_client())

    # Register notification channel so background tasks can notify the session
    set_notification_channel(asyncio.get_running_loop(), notification_queue)

    async def run_session():
        """Try connecting with each API key in order, fallback on failure."""
        last_error = None
        for attempt, api_key in enumerate(GEMINI_API_KEYS):
            key_label = f"key {attempt+1}/{len(GEMINI_API_KEYS)}"
            try:
                client = GeminiLive(
                    api_key=api_key,
                    model=MODEL,
                    input_sample_rate=16000,
                    tools=TOOL_DECLARATIONS,
                    tool_mapping=TOOL_MAPPING,
                )
                logger.info(f"Connecting with {key_label}...")
                async for event in client.start_session(
                    audio_input_queue=audio_input_queue,
                    video_input_queue=video_input_queue,
                    text_input_queue=text_input_queue,
                    audio_output_callback=audio_output_callback,
                    audio_interrupt_callback=audio_interrupt_callback,
                    notification_queue=notification_queue,
                    file_input_queue=file_input_queue,
                ):
                    if event:
                        # Accumulate usage stats before forwarding to client
                        if event.get("type") == "usage":
                            _accumulate_usage(event["usage"])
                        await websocket.send_json(event)
                return  # session ended normally
            except Exception as e:
                last_error = e
                logger.warning(f"{key_label} failed: {type(e).__name__}: {e}")
                if attempt < len(GEMINI_API_KEYS) - 1:
                    logger.info(f"Falling back to next key...")
                    await websocket.send_json({
                        "type": "gemini",
                        "text": f"[API key {attempt+1} failed, switching to fallback key...]"
                    })
                else:
                    raise

    try:
        await run_session()
    except Exception as e:
        import traceback
        logger.error(f"Error in Gemini session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
