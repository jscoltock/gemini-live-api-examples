import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import yaml
from google.genai import types

from task_manager import TaskManager
from ollama_tools import get_schemas, get_funcs, list_tools as list_ollama_tools

logger = logging.getLogger(__name__)

# --- Agent config ---
AGENTS_CONFIG_PATH = Path(__file__).parent / "agents.yaml"


def _load_agents() -> dict:
    """Load agent definitions from agents.yaml."""
    with open(AGENTS_CONFIG_PATH) as f:
        return yaml.safe_load(f)["agents"]


AGENTS = _load_agents()


def _get_attempts(agent_config: dict) -> list[dict]:
    """
    Build the ordered list of configs to try: primary first, then fallbacks.
    Each config has backend, model, timeout only.
    system_prompt is NOT included — it lives at agent level only.
    """
    attempts = []

    # Primary config
    primary = {k: v for k, v in agent_config.items()
               if k in ("backend", "model", "timeout")}
    attempts.append(primary)

    # Fallbacks
    for fb in agent_config.get("fallbacks", []):
        attempts.append({k: v for k, v in fb.items() if k in ("backend", "model", "timeout")})

    return attempts


def _build_command(config: dict, system_prompt: str, prompt: str) -> str:
    """Build the shell command for an agent based on its backend type."""
    backend = config["backend"].lower()
    model = config.get("model", "")

    # Combine system prompt + user prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    # Escape for shell
    escaped_prompt = shlex.quote(full_prompt)

    if backend == "ollama":
        # Use Ollama HTTP API directly
        payload = json.dumps({"model": model, "prompt": full_prompt, "stream": False})
        return (
            f"curl -s http://localhost:11434/api/generate "
            f"-d {shlex.quote(payload)} "
            f"| python3 -c \"import sys,json; print(json.load(sys.stdin)['response'])\""
        )

    elif backend == "claude-code":
        cmd = f"claude --model {shlex.quote(model)}"
        cmd += " --dangerously-skip-permissions"
        cmd += f" -p {escaped_prompt}"
        return cmd

    else:
        raise ValueError(f"Unknown backend: {backend}")


# --- Shared state for notification callback ---
_event_loop: asyncio.AbstractEventLoop | None = None
_notification_queue: asyncio.Queue | None = None
task_manager = TaskManager()


def set_notification_channel(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    """Register the event loop and queue so background tasks can send notifications."""
    global _event_loop, _notification_queue
    _event_loop = loop
    _notification_queue = queue

    # Wire TaskManager notifications into the async queue
    def _threadsafe_notify(message: str):
        if _event_loop and _notification_queue:
            _event_loop.call_soon_threadsafe(_notification_queue.put_nowait, message)
            logger.info(f"Queued notification: {message[:100]}")
        else:
            logger.warning(f"No notification channel, dropping: {message[:100]}")

    task_manager.set_notify(_threadsafe_notify)


# --- Synchronous ask_agent for turn-based providers (GLM/Qwen) ---

def ask_agent_sync(agent: str, prompt: str) -> str:
    """Run an agent synchronously and return the actual result.

    Unlike ask_agent (which is fire-and-forget for Gemini Live's
    notification channel), this blocks until the agent finishes and
    returns the output directly. Still registers with TaskManager so
    the agent pane shows status and trace data.
    """
    if agent not in AGENTS:
        available = ", ".join(AGENTS.keys())
        return f"Unknown agent '{agent}'. Available: {available}"

    agent_config = AGENTS[agent]
    system_prompt = agent_config.get("system_prompt", "").strip()
    tool_names = agent_config.get("tools", [])
    agent_options = agent_config.get("options")
    attempts = _get_attempts(agent_config)

    if not attempts:
        return f"Agent '{agent}' has no available configs (check env vars)."

    # Build the ordered list of (command_or_callable, timeout, label) tuples
    run_list = []
    for cfg in attempts:
        try:
            backend = cfg.get("backend", "").lower()

            if backend == "ollama" and tool_names:
                _model = cfg.get("model", "")
                _timeout = cfg.get("timeout", 120)
                _tool_names = list(tool_names)
                _opts = dict(agent_options) if agent_options else None
                _sp = system_prompt

                def _make_callable(m, sp, p, tn, o, t):
                    def _run(trace_callback=None):
                        return run_ollama_agent(m, sp, p, tn, o, t, trace_callback=trace_callback)
                    return _run

                cmd = _make_callable(_model, _sp, prompt, _tool_names, _opts, _timeout)
            else:
                cmd = _build_command(cfg, system_prompt, prompt)

            timeout = cfg.get("timeout", 120)
            label = f"{cfg.get('backend', '?')}/{cfg.get('model', '?')}"
            run_list.append((cmd, timeout, label))
        except Exception as e:
            logger.warning(f"Skipping config for agent '{agent}': {e}")

    if not run_list:
        return f"Agent '{agent}' — all configs failed to build commands."

    logger.info(
        f"ask_agent_sync: agent={agent} attempts={len(run_list)} "
        f"primary={attempts[0].get('backend','?')}/{attempts[0].get('model','?')} "
        f"prompt={prompt[:80]}"
    )

    # Start task in TaskManager so agent pane gets status + trace
    task = task_manager.start(agent, run_list)

    # Block until the task finishes
    if task.thread:
        task.thread.join()

    # Return the actual output
    result = task.output or "(no output)"
    max_chars = 8000
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n... (truncated, {len(result)} total chars)"
    return result


# --- Tool Implementations ---

def run_bash(command: str) -> str:
    """Execute a bash shell command and return its output.

    Args:
        command: The bash command to run.
    """
    logger.info(f"run_bash executing: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout
        if result.stderr:
            noisy = [
                "Warning: no stdin data received",
                "proceeding without it",
                "redirect stdin explicitly",
            ]
            filtered = "\n".join(
                line for line in result.stderr.strip().splitlines()
                if not any(n in line for n in noisy)
            )
            if filtered.strip():
                output += "\nSTDERR: " + filtered
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        output = output.strip() or "(no output)"
        # Truncate to prevent overwhelming the model's context
        max_chars = 8000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... (truncated, {len(output)} total chars)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def ask_agent(agent: str, prompt: str) -> str:
    """Route a prompt to a named agent. All agents run async in the background.

    Returns immediately with a task ID. When the agent finishes, a notification
    is sent with the result. If the primary config fails, fallbacks are tried
    in order until one succeeds or all are exhausted.

    Args:
        agent: The agent name to use (must match a key in agents.yaml).
        prompt: The prompt or instruction to send to the agent.
    """
    if agent not in AGENTS:
        available = ", ".join(AGENTS.keys())
        return f"Unknown agent '{agent}'. Available: {available}"

    agent_config = AGENTS[agent]
    system_prompt = agent_config.get("system_prompt", "").strip()
    tool_names = agent_config.get("tools", [])
    agent_options = agent_config.get("options")
    attempts = _get_attempts(agent_config)

    if not attempts:
        return f"Agent '{agent}' has no available configs (check env vars)."

    # Build the ordered list of (command_or_callable, timeout, label) tuples
    run_list = []
    for cfg in attempts:
        try:
            backend = cfg.get("backend", "").lower()

            if backend == "ollama" and tool_names:
                # Ollama with tools — use the agentic tool loop
                _model = cfg.get("model", "")
                _timeout = cfg.get("timeout", 120)
                _tool_names = list(tool_names)
                _opts = dict(agent_options) if agent_options else None
                _sp = system_prompt

                def _make_callable(m, sp, p, tn, o, t):
                    def _run(trace_callback=None):
                        return run_ollama_agent(m, sp, p, tn, o, t, trace_callback=trace_callback)
                    return _run

                cmd = _make_callable(_model, _sp, prompt, _tool_names, _opts, _timeout)
            else:
                cmd = _build_command(cfg, system_prompt, prompt)

            timeout = cfg.get("timeout", 120)
            label = f"{cfg.get('backend', '?')}/{cfg.get('model', '?')}"
            run_list.append((cmd, timeout, label))
        except Exception as e:
            logger.warning(f"Skipping config for agent '{agent}': {e}")

    if not run_list:
        return f"Agent '{agent}' — all configs failed to build commands."

    logger.info(
        f"ask_agent: agent={agent} attempts={len(run_list)} "
        f"primary={attempts[0].get('backend','?')}/{attempts[0].get('model','?')} "
        f"prompt={prompt[:80]}"
    )

    task = task_manager.start(agent, run_list)
    return f"Task {task.id} started. Agent: {agent} ({len(run_list)} config(s) queued). You'll be notified when it completes."


def list_tasks(status_filter: str = "") -> str:
    """List tracked background tasks.

    Args:
        status_filter: Optional filter — 'running', 'completed', 'failed', 'timed_out', 'cancelled'.
    """
    tasks = task_manager.list_tasks(status_filter or None)
    if not tasks:
        return "No tasks"
    lines = []
    for t in tasks:
        short_cmd = t["command"][:60]
        if len(t["command"]) > 60:
            short_cmd += "..."
        lines.append(f"[{t['id']}] {t['status']} | {t['agent']} | {short_cmd}")
    return "\n".join(lines)


def cancel_task(task_id: str) -> str:
    """Cancel a running background task.

    Args:
        task_id: The task ID to cancel.
    """
    return task_manager.cancel(task_id)


# --- Ollama Agent Tool Support ---
# Tools are now defined in ollama_tools.py using the @ollama_tool decorator.
# Schemas and functions are resolved dynamically from the registry.
# Use get_schemas(names) and get_funcs(names) to build what's needed at runtime.

MAX_OLLAMA_TOOL_ITERATIONS = 15


def _ollama_chat_request(payload: dict, timeout: int = 120) -> dict:
    """POST to Ollama /api/chat and return the parsed JSON response."""
    url = "http://localhost:11434/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama /api/chat request failed: {e}")


def run_ollama_agent(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tool_names: list[str],
    options: dict | None = None,
    timeout: int = 120,
    trace_callback=None,
) -> str:
    """Run an Ollama agent with native tool calling via /api/chat.

    Repeatedly calls the model, executes any tool calls it makes,
    feeds results back, and returns the final text answer.

    If trace_callback is provided, it's called with dicts describing each step:
      {"type": "thinking", "data": {"content": "..."}}
      {"type": "tool_call", "data": {"name": "...", "args": {...}}}
      {"type": "tool_result", "data": {"name": "...", "result": "..."}}
      {"type": "response", "data": {"content": "..."}}
    """
    def _trace(event: dict):
        if trace_callback:
            trace_callback(event)

    # Build schemas and function map for requested tools from the registry
    tool_schemas = get_schemas(tool_names)
    tool_funcs = get_funcs(tool_names)

    if not tool_schemas:
        # No valid tools — fall back to simple /api/generate
        return _ollama_simple_generate(model, system_prompt, user_prompt, options, timeout)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    opts = options or {"temperature": 0, "top_p": 0.9, "repeat_penalty": 1.1}
    deadline = time.time() + timeout

    for _ in range(MAX_OLLAMA_TOOL_ITERATIONS):
        remaining = deadline - time.time()
        if remaining <= 5:
            _trace({"type": "error", "data": {"message": "Agent timed out"}})
            return "Error: agent timed out"

        payload = {
            "model": model,
            "messages": messages,
            "tools": tool_schemas,
            "stream": False,
            "options": opts,
        }

        try:
            data = _ollama_chat_request(payload, timeout=min(int(remaining) - 2, 120))
        except Exception as e:
            _trace({"type": "error", "data": {"message": str(e)}})
            return f"Error: {e}"

        msg = data.get("message", {})
        messages.append(msg)

        # Emit thinking if present (some models include chain-of-thought)
        thinking = msg.get("thinking", "")
        if thinking:
            _trace({"type": "thinking", "data": {"content": thinking[:2000]}})

        # Emit any text content the model produced alongside tool calls
        text_content = msg.get("content", "")
        if text_content and text_content.strip():
            _trace({"type": "response", "data": {"content": text_content[:2000]}})

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                func_info = call.get("function", {})
                name = func_info.get("name", "")
                args = func_info.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                _trace({"type": "tool_call", "data": {"name": name, "args": args}})

                func = tool_funcs.get(name)
                if func:
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Error: unknown tool '{name}'"

                _trace({"type": "tool_result", "data": {"name": name, "result": str(result)[:2000]}})

                logger.info(f"Ollama tool call: {name}({json.dumps(args)[:100]}) -> {str(result)[:100]}")
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
            continue

        # No tool calls — final answer
        content = msg.get("content", "")
        if content and content.strip():
            return content.strip()

        # Empty content with no tool calls — the model finished but said nothing.
        # This can happen when the model's work was entirely via tool calls.
        # Return a summary of what was done instead.
        tool_log = []
        for m in messages:
            if m.get("role") == "tool":
                tool_log.append(f"- {m.get('tool_name', m.get('name','?'))}: {str(m.get('content',''))[:80]}")
        if tool_log:
            return f"Done. Tool calls made:\n" + "\n".join(tool_log)
        return "(no response)"

    _trace({"type": "error", "data": {"message": "Max tool iterations reached"}})
    return "Error: max tool iterations reached"


def _ollama_simple_generate(
    model: str, system_prompt: str, user_prompt: str,
    options: dict | None, timeout: int,
) -> str:
    """Fallback for Ollama agents with no tools — uses /api/generate."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    payload = {"model": model, "prompt": full_prompt, "stream": False}
    if options:
        payload["options"] = options
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "(no response)")
    except urllib.error.URLError as e:
        return f"Error: {e}"


# --- Build tool declarations dynamically from agents.yaml ---

def _build_agent_descriptions() -> str:
    """Build a description string listing available agents and when to use each."""
    parts = []
    for name, config in AGENTS.items():
        desc = config.get("description", "").strip().replace("\n", " ")
        fb_count = len(config.get("fallbacks", []))
        fb_note = f" ({fb_count} fallback{'s' if fb_count != 1 else ''})" if fb_count else ""
        parts.append(f"- {name}: {desc}{fb_note}")
    return "\n".join(parts)


_agent_descriptions = _build_agent_descriptions()

bash_tool_declaration = types.FunctionDeclaration(
    name="run_bash",
    description="Execute a bash shell command and return the output. Use for quick one-liners: checking files, system info, etc. 30s timeout.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run",
            }
        },
        "required": ["command"],
    },
)

ask_agent_declaration = types.FunctionDeclaration(
    name="ask_agent",
    description=(
        "Send a task to a specialist agent. Runs in the background, returns a task ID immediately. "
        "You will receive a notification when the agent finishes.\n\n"
        "If the primary config fails, fallbacks are tried automatically.\n\n"
        "Available agents:\n" + _agent_descriptions
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Agent name to use: " + ", ".join(AGENTS.keys()),
                "enum": list(AGENTS.keys()),
            },
            "prompt": {
                "type": "string",
                "description": "The prompt or instruction to send to the agent. Include all relevant context from the conversation since the agent has no memory of previous turns.",
            },
        },
        "required": ["agent", "prompt"],
    },
)

list_tasks_declaration = types.FunctionDeclaration(
    name="list_tasks",
    description="List running and recent background tasks. Use when the user asks 'what's running' or 'status of my tasks'.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "status_filter": {
                "type": "string",
                "description": "Optional filter: running, completed, failed, timed_out, cancelled",
            }
        },
        "required": [],
    },
)

cancel_task_declaration = types.FunctionDeclaration(
    name="cancel_task",
    description="Cancel a running background task by its ID.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel",
            }
        },
        "required": ["task_id"],
    },
)


# --- Aggregated lists for wiring into GeminiLive ---

TOOL_DECLARATIONS = [
    types.Tool(function_declarations=[
        bash_tool_declaration,
        ask_agent_declaration,
        list_tasks_declaration,
        cancel_task_declaration,
    ])
]

TOOL_MAPPING = {
    "run_bash": run_bash,
    "ask_agent": ask_agent,
    "list_tasks": list_tasks,
    "cancel_task": cancel_task,
}
