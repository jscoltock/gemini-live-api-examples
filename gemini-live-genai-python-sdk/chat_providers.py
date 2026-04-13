"""
Chat providers for non-Live models (Ollama and ZAI/OpenAI-compatible).
Each provider supports tool calling with the same agents/tools as Gemini Live.
Streams text chunks as SSE events.
"""
import asyncio
import json
import logging
import os
import traceback

import httpx

logger = logging.getLogger(__name__)

# --- Provider config ---

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_zai_key():
    return os.getenv("ZAI_API_KEY", "")


def _get_zai_base():
    return os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")


# --- Models ---
# Loaded dynamically from agents.yaml chat_models section.
# The hardcoded dict is only a fallback for when yaml is unavailable.

_AVAILABLE_MODELS_FALLBACK = {
    "glm-5.1": {
        "label": "GLM-5.1",
        "backend": "zai",
        "model": "glm-5.1",
        "voice": False,
        "interruptible": False,
    },
    "qwen3.5:9b-64K": {
        "label": "Qwen 3.5 9B (Ollama)",
        "backend": "ollama",
        "model": "qwen3.5:9b-64K",
        "voice": False,
        "interruptible": False,
    },
}


def _get_available_models() -> dict:
    """Load models from agents.yaml, fallback to hardcoded."""
    try:
        import agent_config
        models = {}
        for m in agent_config.list_chat_models():
            mid = m["id"]
            models[mid] = {
                "label": m.get("label", mid),
                "backend": m.get("backend", ""),
                "model": m.get("model", mid),
                "voice": m.get("voice", False),
                "interruptible": m.get("interruptible", False),
                "system_prompt": m.get("system_prompt", ""),
            }
        return models if models else _AVAILABLE_MODELS_FALLBACK
    except Exception:
        return _AVAILABLE_MODELS_FALLBACK


# --- Tool schemas (OpenAI-compatible format) ---

def _get_tool_schemas():
    """Build OpenAI-compatible tool schemas from tools.py declarations."""
    # Import here to avoid circular imports and ensure .env is loaded first
    from tools import TOOL_MAPPING
    import agent_config

    schemas = []
    for name in TOOL_MAPPING:
        schemas.append(_TOOL_SCHEMAS.get(name))
    return [s for s in schemas if s]


def _get_system_prompt(model_id: str = None) -> str:
    """Get the system prompt for a specific model from agents.yaml."""
    try:
        import agent_config
        if model_id:
            cfg = agent_config.get_chat_model(model_id)
            if cfg and cfg.get("system_prompt"):
                return cfg["system_prompt"]
        # Fallback to gemini_session system prompt
        cfg = agent_config.get_gemini_session()
        return cfg.get("system_prompt", "")
    except Exception:
        return ""


# Static tool schemas — same definitions as in tools.py but in OpenAI format
_TOOL_SCHEMAS = {
    "run_bash": {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash shell command and return the output. Use for quick one-liners: checking files, system info, etc. 30s timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run",
                    }
                },
                "required": ["command"],
            },
        }
    },
    "ask_agent": {
        "type": "function",
        "function": {
            "name": "ask_agent",
            "description": (
                "Send a task to a specialist agent. Runs in the background, returns a task ID immediately. "
                "You will receive a notification when the agent finishes.\n\n"
                "If the primary config fails, fallbacks are tried automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Agent name to use",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt or instruction to send to the agent. Include all relevant context from the conversation since the agent has no memory of previous turns.",
                    },
                },
                "required": ["agent", "prompt"],
            },
        }
    },
    "list_tasks": {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List running and recent background tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "description": "Optional filter: running, completed, failed, timed_out, cancelled",
                    }
                },
                "required": [],
            },
        }
    },
    "cancel_task": {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": "Cancel a running background task by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to cancel",
                    }
                },
                "required": ["task_id"],
            },
        }
    },
}

MAX_TOOL_ITERATIONS = 10


# --- Tool execution ---

def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name with the given args.

    For ask_agent, uses the synchronous version that blocks until
    the agent finishes and returns the actual result — since GLM/Qwen
    have no notification channel to push results back later.
    """
    try:
        if name == "ask_agent":
            from tools import ask_agent_sync
            return ask_agent_sync(args.get("agent", ""), args.get("prompt", ""))

        from tools import TOOL_MAPPING
        func = TOOL_MAPPING.get(name)
        if not func:
            return f"Error: unknown tool '{name}'"
        result = func(**args)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Non-streaming calls for tool loop ---

async def _ollama_chat_once(messages: list[dict], model: str, tools: list[dict] = None) -> dict:
    """Single non-streaming call to Ollama /api/chat. Returns the full response."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": tools or [],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _zai_chat_once(messages: list[dict], model: str, tools: list[dict] = None) -> dict:
    """Single non-streaming call to ZAI chat completions. Returns the full response."""
    headers = {
        "Authorization": f"Bearer {_get_zai_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    base_url = _get_zai_base()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# --- Streaming generators ---


async def stream_ollama(messages: list[dict], model: str, tools: list[dict] = None, trace_callback=None):
    """Stream from local Ollama /api/chat with tool support.
    trace_callback: optional callable(dict) for observability events."""
    # Tool loop: keep calling until no more tool_calls or max iterations
    if tools:
        for iteration in range(MAX_TOOL_ITERATIONS):
            if trace_callback:
                trace_callback({"type": "thinking", "data": {"content": f"Calling {model} (iteration {iteration + 1})..."}})

            data = await _ollama_chat_once(messages, model, tools)
            msg = data.get("message", {})
            messages.append(msg)

            # Check for thinking content alongside tool calls
            thinking = msg.get("content", "")

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                # Final answer — stream it
                if thinking:
                    if trace_callback:
                        trace_callback({"type": "response", "data": {"content": thinking[:2000]}})
                    yield thinking
                return

            # Model produced tool calls — emit thinking if present
            if thinking and trace_callback:
                trace_callback({"type": "thinking", "data": {"content": thinking[:2000]}})

            # Execute tool calls
            for call in tool_calls:
                func_info = call.get("function", {})
                name = func_info.get("name", "")
                args = func_info.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if trace_callback:
                    trace_callback({"type": "tool_call", "data": {"name": name, "args": args}})

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda n=name, a=args: _execute_tool(n, a))

                if trace_callback:
                    trace_callback({"type": "tool_result", "data": {"name": name, "result": str(result)[:2000]}})

                logger.info(f"Ollama tool call: {name}({json.dumps(args)[:100]}) -> {str(result)[:100]}")
                yield f"\n[Tool: {name}]\n{result}\n\n"
                messages.append({"role": "tool", "name": name, "content": str(result)})

        if trace_callback:
            trace_callback({"type": "error", "data": {"message": "Max tool iterations reached"}})
        yield "\n[Max tool iterations reached]"

    else:
        # No tools — simple streaming
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done"):
                            return
                    except json.JSONDecodeError:
                        continue


async def stream_zai(messages: list[dict], model: str, tools: list[dict] = None, trace_callback=None):
    """Stream from ZAI (OpenAI-compatible) with tool support.
    trace_callback: optional callable(dict) for observability events."""
    if tools:
        for iteration in range(MAX_TOOL_ITERATIONS):
            if trace_callback:
                trace_callback({"type": "thinking", "data": {"content": f"Calling {model} (iteration {iteration + 1})..."}})

            data = await _zai_chat_once(messages, model, tools)
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            messages.append(msg)

            thinking = msg.get("content", "")

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                if thinking:
                    if trace_callback:
                        trace_callback({"type": "response", "data": {"content": thinking[:2000]}})
                    yield thinking
                return

            if thinking and trace_callback:
                trace_callback({"type": "thinking", "data": {"content": thinking[:2000]}})

            # Execute tool calls
            for call in tool_calls:
                func_info = call.get("function", {})
                name = func_info.get("name", "")
                args = func_info.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if trace_callback:
                    trace_callback({"type": "tool_call", "data": {"name": name, "args": args}})

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda n=name, a=args: _execute_tool(n, a))

                if trace_callback:
                    trace_callback({"type": "tool_result", "data": {"name": name, "result": str(result)[:2000]}})

                logger.info(f"ZAI tool call: {name}({json.dumps(args)[:100]}) -> {str(result)[:100]}")
                yield f"\n[Tool: {name}]\n{result}\n\n"
                # OpenAI format tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": str(result),
                })

        if trace_callback:
            trace_callback({"type": "error", "data": {"message": "Max tool iterations reached"}})
        yield "\n[Max tool iterations reached]"

    else:
        # No tools — simple streaming
        headers = {
            "Authorization": f"Bearer {_get_zai_key()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        base_url = _get_zai_base()
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


async def stream_chat(model_id: str, messages: list[dict], use_tools: bool = True, trace_callback=None):
    """Route to the correct provider with tools and system prompt."""
    AVAILABLE_MODELS = _get_available_models()
    cfg = AVAILABLE_MODELS.get(model_id)
    if not cfg:
        raise ValueError(f"Unknown model: {model_id}")

    backend = cfg["backend"]
    model_name = cfg.get("model", model_id)

    # Inject model-specific system prompt if not already present
    system_prompt = _get_system_prompt(model_id)
    if system_prompt and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": system_prompt}] + messages

    # Get tool schemas
    tools = _get_tool_schemas() if use_tools else None

    if backend == "ollama":
        async for chunk in stream_ollama(messages, model_name, tools, trace_callback=trace_callback):
            yield chunk
    elif backend == "zai":
        async for chunk in stream_zai(messages, model_name, tools, trace_callback=trace_callback):
            yield chunk
    else:
        raise ValueError(f"Non-streamable backend: {backend}")
