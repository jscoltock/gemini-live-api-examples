"""
Self-registering Ollama tool registry.

Define a tool by decorating a function with @ollama_tool.
The decorator registers it automatically — agents.yaml just references the name.

Usage:
    from ollama_tools import ollama_tool, get_schemas, get_funcs, list_tools

    @ollama_tool(
        description="Read the contents of a file",
        parameters={
            "path": {"type": "string", "description": "Path to the file"},
        },
        required=["path"],
    )
    def read_file(path: str) -> str:
        return Path(path).read_text()

Then in tools.py / run_ollama_agent, resolve tool names to schemas and funcs:
    schemas = get_schemas(["read_file", "bash"])
    funcs   = get_funcs(["read_file", "bash"])
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Global registries: name -> entry
_REGISTRY: dict[str, dict] = {}


class OllamaToolDef:
    """Holds the schema and callable for one registered tool."""

    __slots__ = ("name", "schema", "func")

    def __init__(self, name: str, schema: dict, func: Callable):
        self.name = name
        self.schema = schema
        self.func = func

    def __repr__(self):
        return f"OllamaToolDef({self.name!r})"


def ollama_tool(
    description: str,
    parameters: dict[str, dict],
    required: list[str] | None = None,
):
    """Decorator that registers an Ollama tool.

    Args:
        description: What the tool does (shown to the LLM).
        parameters:  JSON Schema properties dict, e.g.
                     {"path": {"type": "string", "description": "File path"}}
        required:    List of required parameter names.
    """
    required = required or []

    def decorator(func: Callable) -> Callable:
        name = func.__name__
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }
        if name in _REGISTRY:
            logger.warning(f"Overwriting already-registered Ollama tool: {name}")
        _REGISTRY[name] = OllamaToolDef(name=name, schema=schema, func=func)
        logger.debug(f"Registered Ollama tool: {name}")
        # Return the original function unchanged so it can still be called directly
        return func

    return decorator


# --- Lookup helpers ---

def get_schemas(tool_names: list[str]) -> list[dict]:
    """Return Ollama-format tool schemas for the given names (skips unknowns)."""
    schemas = []
    for name in tool_names:
        entry = _REGISTRY.get(name)
        if entry:
            schemas.append(entry.schema)
        else:
            logger.warning(f"Unknown Ollama tool '{name}', skipping")
    return schemas


def get_funcs(tool_names: list[str]) -> dict[str, Callable]:
    """Return {name: callable} for the given names (skips unknowns)."""
    funcs = {}
    for name in tool_names:
        entry = _REGISTRY.get(name)
        if entry:
            funcs[name] = entry.func
        else:
            logger.warning(f"Unknown Ollama tool '{name}', skipping")
    return funcs


def list_tools() -> list[str]:
    """Return all registered tool names."""
    return sorted(_REGISTRY.keys())


def get_tool(name: str) -> OllamaToolDef | None:
    """Return a single tool definition, or None."""
    return _REGISTRY.get(name)


# =====================================================================
# Built-in tools — each one self-registers on import.
# To add YOUR OWN tools, either add them below or create a new file
# that imports `ollama_tool` and uses the same pattern.
# =====================================================================

import json as _json
import subprocess
from pathlib import Path

MAX_OUTPUT_CHARS = 8000


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"
    return text


# --- read_file ---

@ollama_tool(
    description="Read the contents of a file",
    parameters={
        "path": {"type": "string", "description": "Path to the file"},
    },
    required=["path"],
)
def read_file(path: str) -> str:
    p = Path(path).resolve()
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        return _truncate(p.read_text())
    except Exception as e:
        return f"Error reading file: {e}"


# --- write_file ---

@ollama_tool(
    description="Write content to a file, creating parent directories if needed",
    parameters={
        "path": {"type": "string", "description": "Path to the file"},
        "content": {"type": "string", "description": "Content to write"},
    },
    required=["path", "content"],
)
def write_file(path: str, content: str) -> str:
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# --- edit_file ---

@ollama_tool(
    description="Find and replace text in a file",
    parameters={
        "path": {"type": "string", "description": "Path to the file"},
        "find": {"type": "string", "description": "Text to find"},
        "replace": {"type": "string", "description": "Text to replace with"},
    },
    required=["path", "find", "replace"],
)
def edit_file(path: str, find: str, replace: str) -> str:
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: file not found: {path}"
        text = p.read_text()
        count = text.count(find)
        if count == 0:
            return f"Error: text not found in {path}"
        text = text.replace(find, replace)
        p.write_text(text)
        return f"Replaced {count} occurrence(s) in {path}"
    except Exception as e:
        return f"Error editing file: {e}"


# --- bash ---

@ollama_tool(
    description="Run a bash command and return its output",
    parameters={
        "command": {"type": "string", "description": "Bash command to run"},
    },
    required=["command"],
)
def bash(command: str) -> str:
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
        return _truncate(output)
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


# --- list_files ---

@ollama_tool(
    description="List files and directories at a given path",
    parameters={
        "path": {"type": "string", "description": "Directory path to list (default: current dir)"},
    },
    required=[],
)
def list_files(path: str = ".") -> str:
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return f"Error: not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries:
            prefix = "DIR " if entry.is_dir() else "FILE"
            lines.append(f"{prefix}  {entry.name}")
        if not lines:
            return "(empty directory)"
        return _truncate("\n".join(lines))
    except Exception as e:
        return f"Error listing files: {e}"


# --- web_fetch ---

@ollama_tool(
    description="Fetch a URL and return its text content (good for reading web pages or APIs)",
    parameters={
        "url": {"type": "string", "description": "The URL to fetch"},
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return (default 4000)",
        },
    },
    required=["url"],
)
def web_fetch(url: str, max_chars: int = 4000) -> str:
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ollama-agent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            return _truncate(data, max_chars)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return f"Error fetching URL: {e}"


# --- grep_files ---

@ollama_tool(
    description="Search for a text pattern in files under a directory",
    parameters={
        "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
        "path": {"type": "string", "description": "Directory to search in (default: current dir)"},
        "file_glob": {"type": "string", "description": "File pattern filter, e.g. '*.py' (default: all files)"},
    },
    required=["pattern"],
)
def grep_files(pattern: str, path: str = ".", file_glob: str = "") -> str:
    import subprocess
    try:
        cmd = ["grep", "-rn", "--color=never", pattern]
        if file_glob:
            cmd.extend(["--include", file_glob])
        cmd.append(path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip() or "(no matches)"
        return _truncate(output)
    except subprocess.TimeoutExpired:
        return "Error: grep timed out"
    except Exception as e:
        return f"Error: {e}"


# --- gws (Google Workspace) ---

@ollama_tool(
    description=(
        "Call the gws CLI to interact with Google Workspace (Gmail, Drive, "
        "Calendar, Sheets, Docs, etc). Examples: "
        "'gws gmail users messages list --params \"{\\\"userId\\\": \\\"me\\\"}\"' "
        "'gws gmail users messages send --params \"{\\\"userId\\\": \\\"me\\\"}\" --json \"{\\\"raw\\\": \\\"...\\\"}\"' "
        "'gws drive files list --params \"{\\\"pageSize\\\": 10}\"' "
        "Returns JSON output from the Google Workspace API."
    ),
    parameters={
        "command": {
            "type": "string",
            "description": "Full gws CLI command (e.g. 'gws gmail users messages list --params ...')",
        },
    },
    required=["command"],
)
def gws(command: str) -> str:
    try:
        cmd = command.strip()
        if not cmd.startswith("gws"):
            cmd = "gws " + cmd
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if result.stderr:
            output += "\nSTDERR: " + result.stderr.strip()
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return _truncate(output) if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: gws command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
