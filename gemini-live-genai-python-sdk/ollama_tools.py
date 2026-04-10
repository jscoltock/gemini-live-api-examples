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


# --- Google Workspace tools (using gws CLI helper commands) ---

def _run_gws(args: list[str], timeout: int = 30) -> str:
    """Run a gws CLI command with list-form args (no shell escaping issues)."""
    try:
        result = subprocess.run(
            ["gws"] + args, capture_output=True, text=True, timeout=timeout
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


# --- gmail_send ---

@ollama_tool(
    description=(
        "Send an email via Gmail. Handles RFC 2822 formatting and base64 "
        "encoding automatically. Use this for all outgoing emails."
    ),
    parameters={
        "to": {
            "type": "string",
            "description": "Recipient email address(es), comma-separated for multiple",
        },
        "subject": {
            "type": "string",
            "description": "Email subject line",
        },
        "body": {
            "type": "string",
            "description": "Email body (plain text, or HTML if html=true)",
        },
        "cc": {
            "type": "string",
            "description": "CC email address(es), comma-separated (optional)",
        },
        "bcc": {
            "type": "string",
            "description": "BCC email address(es), comma-separated (optional)",
        },
        "html": {
            "type": "boolean",
            "description": "Set true to send body as HTML (default: plain text)",
        },
    },
    required=["to", "subject", "body"],
)
def gmail_send(to: str, subject: str, body: str, cc: str = "", bcc: str = "", html: bool = False) -> str:
    args = ["gmail", "+send", "--to", to, "--subject", subject, "--body", body]
    if cc:
        args.extend(["--cc", cc])
    if bcc:
        args.extend(["--bcc", bcc])
    if html:
        args.append("--html")
    return _run_gws(args)


# --- gmail_triage ---

@ollama_tool(
    description=(
        "Show unread inbox summary (sender, subject, date). "
        "Use to check what emails have come in. Read-only."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Gmail search query (default: is:unread). Examples: 'from:boss', 'subject:urgent', 'is:unread'",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum messages to show (default: 20)",
        },
    },
    required=[],
)
def gmail_triage(query: str = "is:unread", max_results: int = 20) -> str:
    args = ["gmail", "+triage", "--query", query, "--max", str(max_results)]
    return _run_gws(args)


# --- gmail_reply ---

@ollama_tool(
    description=(
        "Reply to a Gmail message. Handles threading (In-Reply-To, References, threadId) "
        "automatically. Quotes the original message in the reply body."
    ),
    parameters={
        "message_id": {
            "type": "string",
            "description": "Gmail message ID to reply to",
        },
        "body": {
            "type": "string",
            "description": "Reply body text (plain text, or HTML if html=true)",
        },
        "cc": {
            "type": "string",
            "description": "Additional CC email address(es), comma-separated (optional)",
        },
        "html": {
            "type": "boolean",
            "description": "Set true to send body as HTML (default: plain text)",
        },
    },
    required=["message_id", "body"],
)
def gmail_reply(message_id: str, body: str, cc: str = "", html: bool = False) -> str:
    args = ["gmail", "+reply", "--message-id", message_id, "--body", body]
    if cc:
        args.extend(["--cc", cc])
    if html:
        args.append("--html")
    return _run_gws(args)


# --- gmail_forward ---

@ollama_tool(
    description=(
        "Forward a Gmail message to new recipients. Includes the original "
        "message with sender, date, subject, and recipients."
    ),
    parameters={
        "message_id": {
            "type": "string",
            "description": "Gmail message ID to forward",
        },
        "to": {
            "type": "string",
            "description": "Recipient email address(es), comma-separated",
        },
        "body": {
            "type": "string",
            "description": "Optional note to include above the forwarded message",
        },
    },
    required=["message_id", "to"],
)
def gmail_forward(message_id: str, to: str, body: str = "") -> str:
    args = ["gmail", "+forward", "--message-id", message_id, "--to", to]
    if body:
        args.extend(["--body", body])
    return _run_gws(args)


# --- gmail_read ---

@ollama_tool(
    description=(
        "Read a specific Gmail message by ID. Returns full message content "
        "(headers, body, labels). Use gmail_triage first to find message IDs."
    ),
    parameters={
        "message_id": {
            "type": "string",
            "description": "Gmail message ID to read",
        },
    },
    required=["message_id"],
)
def gmail_read(message_id: str) -> str:
    args = [
        "gmail", "users", "messages", "get",
        "--params", _json.dumps({"userId": "me", "id": message_id, "format": "full"}),
    ]
    return _run_gws(args)


# --- gmail_search ---

@ollama_tool(
    description=(
        "Search Gmail messages. Returns a list of matching message IDs and snippets. "
        "Use gmail_read to get full content of a specific message."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Gmail search query. Examples: 'from:someone@example.com', 'subject:meeting', 'newer_than:7d'",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum messages to return (default: 10)",
        },
    },
    required=["query"],
)
def gmail_search(query: str, max_results: int = 10) -> str:
    args = [
        "gmail", "users", "messages", "list",
        "--params", _json.dumps({"userId": "me", "q": query, "maxResults": max_results}),
    ]
    return _run_gws(args)


# --- calendar_agenda ---

@ollama_tool(
    description=(
        "Show upcoming calendar events. Read-only. "
        "Defaults to showing upcoming events across all calendars."
    ),
    parameters={
        "period": {
            "type": "string",
            "description": "Time period: 'today', 'tomorrow', 'week', or a number of days (e.g. '3'). Default: shows next few days.",
        },
        "calendar": {
            "type": "string",
            "description": "Filter to specific calendar name or ID (optional)",
        },
    },
    required=[],
)
def calendar_agenda(period: str = "", calendar: str = "") -> str:
    args = ["calendar", "+agenda"]
    if period == "today":
        args.append("--today")
    elif period == "tomorrow":
        args.append("--tomorrow")
    elif period == "week":
        args.append("--week")
    elif period.isdigit():
        args.extend(["--days", period])
    if calendar:
        args.extend(["--calendar", calendar])
    return _run_gws(args)


# --- calendar_insert ---

@ollama_tool(
    description=(
        "Create a new calendar event. Times in RFC 3339 format "
        "(e.g. '2026-06-17T09:00:00-07:00')."
    ),
    parameters={
        "summary": {
            "type": "string",
            "description": "Event title/summary",
        },
        "start": {
            "type": "string",
            "description": "Start time in RFC 3339 format (e.g. 2026-06-17T09:00:00-07:00)",
        },
        "end": {
            "type": "string",
            "description": "End time in RFC 3339 format",
        },
        "attendees": {
            "type": "string",
            "description": "Attendee email(s), comma-separated (optional)",
        },
        "location": {
            "type": "string",
            "description": "Event location (optional)",
        },
        "description": {
            "type": "string",
            "description": "Event description (optional)",
        },
    },
    required=["summary", "start", "end"],
)
def calendar_insert(
    summary: str, start: str, end: str,
    attendees: str = "", location: str = "", description: str = "",
) -> str:
    args = ["calendar", "+insert", "--summary", summary, "--start", start, "--end", end]
    if attendees:
        for addr in attendees.split(","):
            args.extend(["--attendee", addr.strip()])
    if location:
        args.extend(["--location", location])
    if description:
        args.extend(["--description", description])
    return _run_gws(args)


# --- sheets_read ---

@ollama_tool(
    description=(
        "Read values from a Google Sheets spreadsheet. Read-only."
    ),
    parameters={
        "spreadsheet_id": {
            "type": "string",
            "description": "Spreadsheet ID (from the URL)",
        },
        "range": {
            "type": "string",
            "description": "Range to read (e.g. 'Sheet1!A1:D10' or 'Sheet1')",
        },
    },
    required=["spreadsheet_id", "range"],
)
def sheets_read(spreadsheet_id: str, range: str) -> str:
    return _run_gws(["sheets", "+read", "--spreadsheet", spreadsheet_id, "--range", range])


# --- sheets_append ---

@ollama_tool(
    description=(
        "Append row(s) to a Google Sheets spreadsheet. "
        "Use values for a single row of simple strings, or json_values for multiple rows."
    ),
    parameters={
        "spreadsheet_id": {
            "type": "string",
            "description": "Spreadsheet ID (from the URL)",
        },
        "values": {
            "type": "string",
            "description": "Comma-separated values for a single row (e.g. 'Alice,100,true')",
        },
        "json_values": {
            "type": "string",
            "description": "JSON array of rows for multi-row insert (e.g. '[[\"a\",\"b\"],[\"c\",\"d\"]]')",
        },
    },
    required=["spreadsheet_id"],
)
def sheets_append(spreadsheet_id: str, values: str = "", json_values: str = "") -> str:
    args = ["sheets", "+append", "--spreadsheet", spreadsheet_id]
    if values:
        args.extend(["--values", values])
    if json_values:
        args.extend(["--json-values", json_values])
    return _run_gws(args)


# --- docs_write ---

@ollama_tool(
    description=(
        "Append text to a Google Docs document. Text is inserted at the end."
    ),
    parameters={
        "document_id": {
            "type": "string",
            "description": "Google Docs document ID (from the URL)",
        },
        "text": {
            "type": "string",
            "description": "Text to append",
        },
    },
    required=["document_id", "text"],
)
def docs_write(document_id: str, text: str) -> str:
    return _run_gws(["docs", "+write", "--document", document_id, "--text", text])


# --- drive_upload ---

@ollama_tool(
    description=(
        "Upload a local file to Google Drive. MIME type detected automatically."
    ),
    parameters={
        "file_path": {
            "type": "string",
            "description": "Local path to the file to upload",
        },
        "name": {
            "type": "string",
            "description": "Target filename in Drive (defaults to source filename)",
        },
        "parent": {
            "type": "string",
            "description": "Parent folder ID in Drive (optional, defaults to root)",
        },
    },
    required=["file_path"],
)
def drive_upload(file_path: str, name: str = "", parent: str = "") -> str:
    args = ["drive", "+upload", file_path]
    if name:
        args.extend(["--name", name])
    if parent:
        args.extend(["--parent", parent])
    return _run_gws(args)


# --- drive_list ---

@ollama_tool(
    description=(
        "List files in Google Drive. Returns file IDs, names, and mime types."
    ),
    parameters={
        "page_size": {
            "type": "integer",
            "description": "Number of files to return (default: 10)",
        },
        "query": {
            "type": "string",
            "description": "Drive query filter (e.g. \"name contains 'report'\", \"mimeType='application/vnd.google-apps.folder'\")",
        },
    },
    required=[],
)
def drive_list(page_size: int = 10, query: str = "") -> str:
    params: dict = {"pageSize": page_size}
    if query:
        params["q"] = query
    args = ["drive", "files", "list", "--params", _json.dumps(params)]
    return _run_gws(args)


# --- people_search ---

@ollama_tool(
    description=(
        "Search Google Contacts. Returns names and email addresses."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Name or email to search for",
        },
    },
    required=["query"],
)
def people_search(query: str) -> str:
    args = [
        "people", "people", "searchContacts",
        "--params", _json.dumps({"query": query, "readMask": "names,emailAddresses"}),
    ]
    return _run_gws(args)


# --- Twilio tools (using Twilio Python SDK via phone-call-message skill venv) ---

_TWILIO_VENV_PYTHON = (
    "/Users/jscoltock/.claude/skills/phone-call-message/.venv/bin/python3"
)
_TWILIO_HELPER = (
    "/Users/jscoltock/.claude/skills/phone-call-message/scripts/twilio_helper.py"
)


def _run_twilio(subcommand: str, args: list[str], timeout: int = 30) -> str:
    """Run a twilio_helper.py subcommand with list-form args."""
    try:
        result = subprocess.run(
            [_TWILIO_VENV_PYTHON, _TWILIO_HELPER, subcommand] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip()
        if result.stderr:
            filtered = "\n".join(
                line for line in result.stderr.strip().splitlines()
                if "DeprecationWarning" not in line
            )
            if filtered.strip():
                output += "\nSTDERR: " + filtered
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return _truncate(output) if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Twilio command timed out"
    except FileNotFoundError:
        return (
            "Error: Twilio helper not found. Ensure "
            "~/.claude/skills/phone-call-message/ exists with .venv."
        )
    except Exception as e:
        return f"Error: {e}"


# --- twilio_call ---

@ollama_tool(
    description=(
        "Make a phone call that speaks a text-to-speech message to the recipient. "
        "The recipient hears the message read aloud. Returns the call SID and status."
    ),
    parameters={
        "to": {
            "type": "string",
            "description": "Phone number in E.164 format (e.g. +15551234567)",
        },
        "message": {
            "type": "string",
            "description": "The text to speak during the call",
        },
        "voice": {
            "type": "string",
            "description": "TTS voice to use (default: Polly.Matthew). Examples: Polly.Matthew, Polly.Joanna, alice.",
        },
    },
    required=["to", "message"],
)
def twilio_call(to: str, message: str, voice: str = "") -> str:
    args = [to, message]
    if voice:
        args.append(voice)
    return _run_twilio("call", args)


# --- twilio_sms ---

@ollama_tool(
    description=(
        "Send an SMS text message. Returns the message SID and status."
    ),
    parameters={
        "to": {
            "type": "string",
            "description": "Phone number in E.164 format (e.g. +15551234567)",
        },
        "body": {
            "type": "string",
            "description": "SMS message text (max 1600 chars)",
        },
    },
    required=["to", "body"],
)
def twilio_sms(to: str, body: str) -> str:
    return _run_twilio("sms", [to, body])


# --- twilio_list_calls ---

@ollama_tool(
    description=(
        "List recent phone calls. Returns call SID, from/to numbers, status, "
        "time, and duration. Read-only."
    ),
    parameters={
        "limit": {
            "type": "integer",
            "description": "Maximum number of calls to return (default: 10)",
        },
    },
    required=[],
)
def twilio_list_calls(limit: int = 10) -> str:
    return _run_twilio("list-calls", [str(limit)])


# --- twilio_list_messages ---

@ollama_tool(
    description=(
        "List recent SMS messages. Returns message SID, from/to numbers, status, "
        "date, and body preview. Read-only."
    ),
    parameters={
        "limit": {
            "type": "integer",
            "description": "Maximum number of messages to return (default: 10)",
        },
    },
    required=[],
)
def twilio_list_messages(limit: int = 10) -> str:
    return _run_twilio("list-messages", [str(limit)])


# --- Tavily tools (web search & extract via Tavily API, raw urllib) ---

import os as _os
import urllib.request as _urllib_request
import urllib.error as _urllib_error


def _get_tavily_key() -> str:
    """Get Tavily API key from env. Loaded by dotenv at startup."""
    key = _os.environ.get("TAVILY_API_KEY", "")
    if not key:
        # Try loading .env manually as fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = _os.environ.get("TAVILY_API_KEY", "")
        except ImportError:
            pass
    return key


def _tavily_api(endpoint: str, payload: dict, timeout: int = 15) -> str:
    """Call a Tavily API endpoint and return the result as formatted text."""
    api_key = _get_tavily_key()
    if not api_key:
        return "Error: TAVILY_API_KEY not set. Add it to .env"
    payload["api_key"] = api_key
    try:
        data = _json.dumps(payload).encode()
        req = _urllib_request.Request(
            f"https://api.tavily.com/{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with _urllib_request.urlopen(req, timeout=timeout) as resp:
            result = _json.loads(resp.read().decode())
            return _truncate(_json.dumps(result, indent=2))
    except _urllib_error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return f"HTTP {e.code}: {e.reason} — {body[:200]}"
    except Exception as e:
        return f"Error: {e}"


# --- web_search ---

@ollama_tool(
    description=(
        "Search the web for information. Returns titles, URLs, and content "
        "snippets for each result. Use this to look up facts, news, documentation, "
        "or any information not already known."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default: 5, max: 10)",
        },
        "search_depth": {
            "type": "string",
            "description": "Search depth: 'basic' (fast) or 'advanced' (thorough, slower). Default: basic.",
        },
        "include_answer": {
            "type": "boolean",
            "description": "Include an AI-generated answer summarizing the results (default: true)",
        },
    },
    required=["query"],
)
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
) -> str:
    return _tavily_api("search", {
        "query": query,
        "max_results": min(max_results, 10),
        "search_depth": search_depth,
        "include_answer": include_answer,
    })


# --- web_search_news ---

@ollama_tool(
    description=(
        "Search for recent news articles. Returns titles, URLs, publish dates, "
        "and content snippets. Use for current events, recent developments, "
        "or time-sensitive information."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "News search query",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default: 5, max: 10)",
        },
    },
    required=["query"],
)
def web_search_news(query: str, max_results: int = 5) -> str:
    return _tavily_api("search", {
        "query": query,
        "max_results": min(max_results, 10),
        "topic": "news",
        "search_depth": "basic",
        "include_answer": False,
    })


# --- web_extract_pages ---

@ollama_tool(
    description=(
        "Extract the text content from one or more web page URLs. "
        "Returns the main content of each page. Good for reading articles, "
        "documentation, or any web page."
    ),
    parameters={
        "urls": {
            "type": "string",
            "description": "URLs to extract, comma-separated (max 5)",
        },
    },
    required=["urls"],
)
def web_extract_pages(urls: str) -> str:
    url_list = [u.strip() for u in urls.split(",") if u.strip()][:5]
    return _tavily_api("extract", {"urls": url_list})
