import asyncio
import subprocess
import uuid
import threading
import logging

from google.genai import types

logger = logging.getLogger(__name__)

# --- Shared state for notification callback ---
# Set by main.py at startup so background threads can notify the live session
_event_loop: asyncio.AbstractEventLoop | None = None
_notification_queue: asyncio.Queue | None = None


def set_notification_channel(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    """Register the event loop and queue so background tasks can send notifications."""
    global _event_loop, _notification_queue
    _event_loop = loop
    _notification_queue = queue


def _notify(message: str):
    """Thread-safe way to push a notification into the async queue."""
    if _event_loop and _notification_queue:
        _event_loop.call_soon_threadsafe(_notification_queue.put_nowait, message)
        logger.info(f"Queued notification: {message[:100]}")
    else:
        logger.warning(f"No notification channel set, dropping: {message[:100]}")


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


# --- Background Task Runner ---

def _run_in_background(task_id: str, command: str):
    """Runs a command in a background thread and notifies on completion."""
    logger.info(f"Background task {task_id} started: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR: " + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        output = output or "(no output)"
        logger.info(f"Background task {task_id} completed: {output[:200]}")
        _notify(f"[Background task {task_id} completed] Command: {command}\nOutput: {output[:500]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Background task {task_id} timed out after 120s")
        _notify(f"[Background task {task_id} timed out] Command: {command} — exceeded 120s limit")
    except Exception as e:
        logger.error(f"Background task {task_id} error: {e}")
        _notify(f"[Background task {task_id} failed] Command: {command}\nError: {e}")


def dispatch_task(command: str) -> str:
    """Spawn a background task to execute a bash command. Returns immediately with a task ID.

    Args:
        command: The bash command to run in the background.
    """
    task_id = str(uuid.uuid4())[:8]
    thread = threading.Thread(
        target=_run_in_background,
        args=(task_id, command),
        daemon=True,
    )
    thread.start()
    logger.info(f"Dispatched background task {task_id}: {command}")
    return f"Task {task_id} started in background. Command: {command}"


# --- Tool Schemas ---

bash_tool_declaration = types.FunctionDeclaration(
    name="run_bash",
    description="Execute a bash shell command on the user's local machine and return the output. Use this to inspect files, run scripts, check system info, etc. Commands run with a 30 second timeout.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run, e.g. 'ls ~/Desktop' or 'wc -l *.txt'",
            }
        },
        "required": ["command"],
    },
)

dispatch_task_declaration = types.FunctionDeclaration(
    name="dispatch_task",
    description="Spawn a background task to execute a bash command. Returns immediately with a task ID. When the task completes, its output is sent back as a notification. Use this for long-running tasks so the conversation can continue.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run in the background, e.g. 'find / -name *.pdf 2>/dev/null > /tmp/pdfs.txt'",
            }
        },
        "required": ["command"],
    },
)


# --- Aggregated lists for wiring into GeminiLive ---

TOOL_DECLARATIONS = [types.Tool(function_declarations=[bash_tool_declaration, dispatch_task_declaration])]
TOOL_MAPPING = {
    "run_bash": run_bash,
    "dispatch_task": dispatch_task,
}
