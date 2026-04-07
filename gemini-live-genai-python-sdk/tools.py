import subprocess
import uuid
import threading
import logging
from datetime import datetime

from google.genai import types

logger = logging.getLogger(__name__)

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
    """Runs a command in a background thread. Result is logged but discarded (for now)."""
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
        logger.info(f"Background task {task_id} completed: {output[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Background task {task_id} timed out after 120s")
    except Exception as e:
        logger.error(f"Background task {task_id} error: {e}")


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
    description="Spawn a background task to execute a bash command. Returns immediately with a task ID. The command runs asynchronously — use this for long-running tasks so the conversation can continue. The result is NOT returned to the conversation (for now). Use run_bash for quick commands where you need the result.",
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
