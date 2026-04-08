import asyncio
import json
import logging
import os
import shlex
import subprocess
import threading
import uuid
from pathlib import Path

import yaml
from google.genai import types

from task_manager import TaskManager

logger = logging.getLogger(__name__)

# --- Agent config ---
AGENTS_CONFIG_PATH = Path(__file__).parent / "agents.yaml"


def _load_agents() -> dict:
    """Load agent definitions from agents.yaml."""
    with open(AGENTS_CONFIG_PATH) as f:
        return yaml.safe_load(f)["agents"]


AGENTS = _load_agents()


def _build_command(config: dict, prompt: str) -> str:
    """Build the shell command for an agent based on its backend type."""
    backend = config["backend"].lower()
    model = config.get("model", "")
    system_prompt = config.get("system_prompt", "").strip()

    # Combine system prompt + user prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    # Escape for shell
    escaped_prompt = shlex.quote(full_prompt)

    if backend == "ollama":
        # Use Ollama HTTP API directly (like scripts/ask_ollama did)
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
            output += "\nSTDERR: " + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def ask_agent(agent: str, prompt: str) -> str:
    """Route a prompt to a named agent. All agents run async in the background.

    Returns immediately with a task ID. When the agent finishes, a notification
    is sent with the result.

    Args:
        agent: The agent name to use (must match a key in agents.yaml).
        prompt: The prompt or instruction to send to the agent.
    """
    if agent not in AGENTS:
        available = ", ".join(AGENTS.keys())
        return f"Unknown agent '{agent}'. Available: {available}"

    config = AGENTS[agent]
    command = _build_command(config, prompt)
    timeout = config.get("timeout", 120)

    logger.info(f"ask_agent: agent={agent} backend={config['backend']} model={config.get('model','')} prompt={prompt[:80]}")

    task = task_manager.start(agent, command, timeout=timeout)
    return f"Task {task.id} started. Agent: {agent}. You'll be notified when it completes."


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


# --- Build tool declarations dynamically from agents.yaml ---

def _build_agent_descriptions() -> str:
    """Build a description string listing available agents and when to use each."""
    parts = []
    for name, config in AGENTS.items():
        desc = config.get("description", "").strip().replace("\n", " ")
        parts.append(f"- {name}: {desc}")
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
