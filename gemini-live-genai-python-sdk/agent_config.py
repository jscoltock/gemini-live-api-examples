"""
Agent configuration CRUD operations for agents.yaml.

Provides read, create, update, delete, and hot-reload functionality.
Uses ruamel.yaml if available (preserves comments/formatting),
falls back to pyyaml.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import yaml
from google.genai import types

logger = logging.getLogger(__name__)

AGENTS_CONFIG_PATH = Path(__file__).parent / "agents.yaml"
VALID_BACKENDS = {"ollama", "claude-code"}
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


def _load_full() -> dict:
    """Load the full agents.yaml (with comments if ruamel available)."""
    with open(AGENTS_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _save_full(data: dict) -> None:
    """Backup then write agents.yaml."""
    shutil.copy2(AGENTS_CONFIG_PATH, str(AGENTS_CONFIG_PATH) + ".bak")
    with open(AGENTS_CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def list_agents() -> list[dict]:
    """Return all agents as a list of dicts with name included."""
    data = _load_full()
    agents = data.get("agents", {})
    result = []
    for name, config in agents.items():
        result.append({
            "name": name,
            "description": config.get("description", "").strip(),
            "backend": config.get("backend", ""),
            "model": config.get("model", ""),
            "timeout": config.get("timeout", 120),
            "system_prompt": config.get("system_prompt", ""),
            "tools": config.get("tools", []),
            "options": config.get("options"),
            "fallbacks": config.get("fallbacks", []),
        })
    return result


def get_agent(name: str) -> Optional[dict]:
    """Return a single agent's full config, or None."""
    data = _load_full()
    agent = data.get("agents", {}).get(name)
    if not agent:
        return None
    return {
        "name": name,
        "description": agent.get("description", "").strip(),
        "backend": agent.get("backend", ""),
        "model": agent.get("model", ""),
        "timeout": agent.get("timeout", 120),
        "system_prompt": agent.get("system_prompt", ""),
        "tools": agent.get("tools", []),
        "options": agent.get("options"),
        "fallbacks": agent.get("fallbacks", []),
    }


def validate_agent(data: dict, is_create: bool = False, existing_name: str = None) -> list[str]:
    """Validate agent data. Returns list of error strings (empty = valid)."""
    errors = []

    # Name validation
    name = data.get("name", "").strip()
    if not name:
        errors.append("Agent name is required")
    elif not SLUG_PATTERN.match(name):
        errors.append("Agent name must be a lowercase slug (letters, numbers, hyphens, must start with a letter)")
    elif is_create:
        # Check uniqueness on create
        existing = _load_full().get("agents", {})
        if name in existing:
            errors.append(f"Agent '{name}' already exists")

    # Backend validation
    backend = data.get("backend", "").strip()
    if not backend:
        errors.append("Backend is required")
    elif backend not in VALID_BACKENDS:
        errors.append(f"Backend must be one of: {', '.join(sorted(VALID_BACKENDS))}")

    # Model validation
    model = data.get("model", "").strip()
    if not model:
        errors.append("Model is required")

    # Timeout validation
    timeout = data.get("timeout")
    if timeout is None:
        errors.append("Timeout is required")
    elif not isinstance(timeout, (int, float)) or timeout <= 0:
        errors.append("Timeout must be a positive number")

    # Tools validation (optional, must be list of strings matching registered tools)
    tools = data.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            errors.append("Tools must be a list")
        elif not all(isinstance(t, str) for t in tools):
            errors.append("Each tool name must be a string")
        else:
            from ollama_tools import list_tools as list_ollama_tools
            available = set(list_ollama_tools())
            unknown = [t for t in tools if t not in available]
            if unknown:
                errors.append(f"Unknown tool(s): {', '.join(unknown)}. Available: {', '.join(sorted(available))}")

    # Options validation (optional, must be dict with numeric values)
    options = data.get("options")
    if options is not None:
        if not isinstance(options, dict):
            errors.append("Options must be a dict")
        else:
            for key, val in options.items():
                if not isinstance(val, (int, float)):
                    errors.append(f"Option '{key}' must be a number")

    # Fallbacks validation
    fallbacks = data.get("fallbacks", [])
    if not isinstance(fallbacks, list):
        errors.append("Fallbacks must be a list")
    else:
        for i, fb in enumerate(fallbacks):
            fb_errors = []
            if not fb.get("backend", "").strip():
                fb_errors.append("backend required")
            elif fb["backend"].strip() not in VALID_BACKENDS:
                fb_errors.append(f"invalid backend '{fb['backend']}'")
            if not fb.get("model", "").strip():
                fb_errors.append("model required")
            fb_timeout = fb.get("timeout")
            if fb_timeout is not None and (not isinstance(fb_timeout, (int, float)) or fb_timeout <= 0):
                fb_errors.append("timeout must be positive")
            if fb_errors:
                errors.append(f"Fallback {i + 1}: {', '.join(fb_errors)}")

    return errors


def create_agent(data: dict) -> tuple[dict, list[str]]:
    """Create a new agent. Returns (agent_dict, errors)."""
    errors = validate_agent(data, is_create=True)
    if errors:
        return None, errors

    name = data["name"].strip()
    full_data = _load_full()
    if "agents" not in full_data:
        full_data["agents"] = {}

    full_data["agents"][name] = _build_agent_config(data)
    _save_full(full_data)

    logger.info(f"Created agent '{name}'")
    return get_agent(name), []


def update_agent(name: str, data: dict) -> tuple[Optional[dict], list[str]]:
    """Update an existing agent. Returns (agent_dict, errors)."""
    full_data = _load_full()
    if name not in full_data.get("agents", {}):
        return None, [f"Agent '{name}' not found"]

    # Don't allow name changes via update (the key would change)
    # If they send a different name, reject it
    if data.get("name", name) != name:
        return None, ["Agent name cannot be changed (delete and recreate instead)"]

    errors = validate_agent(data, is_create=False, existing_name=name)
    if errors:
        return None, errors

    full_data["agents"][name] = _build_agent_config(data)
    _save_full(full_data)

    logger.info(f"Updated agent '{name}'")
    return get_agent(name), []


def delete_agent(name: str) -> tuple[bool, str]:
    """Delete an agent. Returns (success, message)."""
    full_data = _load_full()
    if name not in full_data.get("agents", {}):
        return False, f"Agent '{name}' not found"

    del full_data["agents"][name]
    _save_full(full_data)

    logger.info(f"Deleted agent '{name}'")
    return True, f"Agent '{name}' deleted"


def reload_agents() -> list[str]:
    """
    Hot-reload the AGENTS dict in tools.py from the file on disk.
    Returns list of agent names that were loaded.
    """
    import tools

    new_agents = {}
    with open(AGENTS_CONFIG_PATH) as f:
        new_agents = yaml.safe_load(f)["agents"]

    tools.AGENTS = new_agents

    # Rebuild the tool declaration's agent list
    tools._agent_descriptions = tools._build_agent_descriptions()

    # Rebuild the ask_agent declaration with updated enum
    tools.ask_agent_declaration = types.FunctionDeclaration(
        name="ask_agent",
        description=(
            "Send a task to a specialist agent. Runs in the background, returns a task ID immediately. "
            "You will receive a notification when the agent finishes.\n\n"
            "If the primary config fails, fallbacks are tried automatically.\n\n"
            "Available agents:\n" + tools._agent_descriptions
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent name to use: " + ", ".join(new_agents.keys()),
                    "enum": list(new_agents.keys()),
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt or instruction to send to the agent.",
                },
            },
            "required": ["agent", "prompt"],
        },
    )

    # Update the tool declarations list
    tools.TOOL_DECLARATIONS = [
        types.Tool(function_declarations=[
            tools.bash_tool_declaration,
            tools.ask_agent_declaration,
            tools.list_tasks_declaration,
            tools.cancel_task_declaration,
        ])
    ]

    names = list(new_agents.keys())

    # Also rebuild Gemini Live tool declarations (includes gemini_session.tools)
    tools.TOOL_DECLARATIONS, tools.TOOL_MAPPING = tools.build_gemini_tools()

    logger.info(f"Hot-reloaded agents: {names}")
    return names


# --- Gemini Session Config ---

def get_gemini_session() -> dict:
    """Return the gemini_session config (model, voice, system_prompt, tools)."""
    data = _load_full()
    session = data.get("gemini_session", {})
    return {
        "model": session.get("model", "gemini-3.1-flash-live-preview"),
        "voice": session.get("voice", "Puck"),
        "system_prompt": session.get("system_prompt", "").strip(),
        "tools": session.get("tools", []),
    }


def update_gemini_session(updates: dict) -> dict:
    """Update gemini_session fields. Writable: system_prompt, tools, voice, model.
    Returns the updated config."""
    data = _load_full()
    if "gemini_session" not in data:
        data["gemini_session"] = {}

    session = data["gemini_session"]

    if "system_prompt" in updates:
        session["system_prompt"] = updates["system_prompt"].strip()

    if "tools" in updates:
        tools = updates["tools"]
        if tools is not None:
            # Validate tool names against registry
            from ollama_tools import list_tools as list_ollama_tools
            available = set(list_ollama_tools())
            unknown = [t for t in tools if t not in available]
            if unknown:
                raise ValueError(f"Unknown tool(s): {', '.join(unknown)}. Available: {', '.join(sorted(available))}")
            session["tools"] = [str(t).strip() for t in tools if str(t).strip()]

    if "voice" in updates:
        session["voice"] = updates["voice"].strip()

    if "model" in updates:
        session["model"] = updates["model"].strip()

    _save_full(data)
    logger.info("Updated gemini_session config")
    return get_gemini_session()


def _build_agent_config(data: dict) -> dict:
    """Build the YAML-compatible agent config dict from form data."""
    config = {}
    config["description"] = data.get("description", "").strip()
    config["backend"] = data["backend"].strip()
    config["model"] = data["model"].strip()
    config["timeout"] = int(data["timeout"]) if data.get("timeout") else 120

    system_prompt = data.get("system_prompt", "").strip()
    config["system_prompt"] = system_prompt

    # Tools (optional list of strings)
    tools = data.get("tools")
    if tools is not None:
        config["tools"] = [str(t).strip() for t in tools if str(t).strip()]

    # Options (optional dict of generation settings)
    options = data.get("options")
    if options is not None:
        config["options"] = {k: v for k, v in options.items() if isinstance(v, (int, float))}

    fallbacks = data.get("fallbacks", [])
    if fallbacks:
        config["fallbacks"] = []
        for fb in fallbacks:
            fb_config = {}
            fb_config["backend"] = fb.get("backend", "").strip()
            fb_config["model"] = fb.get("model", "").strip()
            fb_timeout = fb.get("timeout")
            if fb_timeout is not None:
                fb_config["timeout"] = int(fb_timeout)
            config["fallbacks"].append(fb_config)

    return config
