# Ollama Tool Calling — Chat Reference

> This is a copy of a conversation between ChatGPT & Anonymous.

---

## Best ~9B Models for Tool Calling

**Goal:** Identify ~9B-parameter local models in Ollama that reliably support tool/function calling.

### Qwen2.5-7B-Instruct (acts like a 9B)

- Exceptional JSON discipline and native function-calling training
- Very reliable structured outputs
- Handles multi-step tool chains well
- Fast locally, low hallucination in schemas

```
ollama run qwen2.5:7b-instruct
```

Despite 7B size, behaves like many 9B–13B models for agents.

### Llama 3.1 8B Instruct

- Strong reasoning + good tool usage
- Needs a stricter system prompt for JSON compliance
- Excellent general agent model

```
ollama run llama3.1:8b-instruct
```

### Mistral Nemo 12B Instruct (slightly above 9B but worth it)

- Outstanding at tool orchestration
- Very stable with schemas
- Often the **best** local tool-calling model under 15B

```
ollama run mistral-nemo:12b-instruct
```

### Phi-3 Medium 14B Instruct (above 9B, very capable)

- Extremely disciplined JSON
- Great for deterministic tool loops
- Lower creativity, higher reliability

```
ollama run phi3:14b-instruct
```

### Gemma 2 9B Instruct

- True 9B, good reasoning
- Weaker at strict JSON/tool schema unless carefully prompted

```
ollama run gemma2:9b-instruct
```

---

## Practical Ranking for Tool Calling (local)

| Rank | Model | Strength |
|------|-------|----------|
| 1 | Qwen2.5-7B-Instruct | best overall for tools |
| 2 | Mistral Nemo 12B | best orchestration |
| 3 | Phi-3 Medium 14B | most deterministic |
| 4 | Llama 3.1 8B | best generalist |
| 5 | Gemma 2 9B | usable but prompt-sensitive |

### Why these work (and most 9B don't)

Tool calling requires:

- Schema obedience
- Low hallucination
- Deterministic formatting
- Instruction tuning specifically for function calls

Most 9B models were trained for chat, not structured agents. These were trained with function/JSON data.

### Pro tip (critical)

For Ollama tool use, set:

```
temperature: 0
top_p: 0.9
repeat_penalty: 1.1
```

These models become dramatically more reliable for function calls.

---

## Building an Agent with 4 Tools: read, write, edit, bash

**Goal:** Wire `qwen2.5:7b-instruct` in Ollama to reliably call four local tools: read, write, edit files, and bash.

Uses Ollama's native tool calling with Qwen2.5-7B-Instruct and a small Python loop that executes the tools.

### 1) Pull the model

```
ollama run qwen2.5:7b-instruct
```

### 2) agent.py

```python
# agent.py
import json
import subprocess
from pathlib import Path

from ollama import Client

client = Client(host='http://localhost:11434')

MODEL = "qwen2.5:7b-instruct"
ROOT = Path.cwd()  # sandbox root


# -------------------- TOOLS --------------------

def read_file(path: str) -> str:
    p = (ROOT / path).resolve()
    return p.read_text()

def write_file(path: str, content: str) -> str:
    p = (ROOT / path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {path}"

def edit_file(path: str, find: str, replace: str) -> str:
    p = (ROOT / path).resolve()
    text = p.read_text()
    text = text.replace(find, replace)
    p.write_text(text)
    return f"Edited {path}"

def run_bash(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    return result.stdout + result.stderr


# -------------------- TOOL SCHEMA --------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find and replace text in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string"},
                    "replace": {"type": "string"}
                },
                "required": ["path", "find", "replace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
]


# -------------------- AGENT LOOP --------------------

def call_tool(name, args):
    if name == "read_file":
        return read_file(**args)
    if name == "write_file":
        return write_file(**args)
    if name == "edit_file":
        return edit_file(**args)
    if name == "bash":
        return run_bash(**args)
    raise ValueError(name)


def run_agent(user_prompt: str):
    messages = [{"role": "user", "content": user_prompt}]

    while True:
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            options={
                "temperature": 0,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        )

        msg = response["message"]

        # If model wants to call a tool
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])

                result = call_tool(name, args)

                messages.append(msg)
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result
                })
            continue

        # Final answer
        print(msg["content"])
        break


if __name__ == "__main__":
    while True:
        prompt = input("\n> ")
        run_agent(prompt)
```

### 3) Install dependency

```
pip install ollama
```

### 4) Run

```
python agent.py
```

Now you can type things like:

> Create a file test.py with a hello world script and run it

Qwen will call `write_file` then `bash python test.py` without you parsing anything.

### Why this works so well with Qwen

Qwen was trained on function-calling data. It **natively emits** proper tool calls that Ollama passes through cleanly. No regex. No parsing hacks. This same loop is what most local "coding agents" do internally.

---

## Adding Smart Bulb Control: `control_light` Tool

**Goal:** Add reliable smart-bulb control to your Ollama + Qwen tool loop without abusing bash.

Use a **separate tool** for bulbs — not prompt tricks and not bash. Bulbs are an API action, not a shell action. Treat them as first-class tools just like files.

### Why not bash?

`bash` is nondeterministic and unsafe for devices. A dedicated tool gives:

- Clean JSON arguments
- Validation
- Device safety
- Model reliability

### Changes to agent.py

#### 1) Add the function

```python
def control_light(entity_id: str, state: str, brightness: int | None = None, color: str | None = None) -> str:
    """
    Example using Home Assistant REST API
    """
    import requests

    HA_URL = "http://homeassistant.local:8123/api/services/light/turn_on"
    TOKEN = "YOUR_L...OKEN"

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }

    data = {"entity_id": entity_id}

    if state == "off":
        HA_URL = "http://homeassistant.local:8123/api/services/light/turn_off"
    else:
        if brightness is not None:
            data["brightness"] = brightness
        if color is not None:
            data["color_name"] = color

    r = requests.post(HA_URL, headers=headers, json=data)
    return r.text
```

#### 2) Add to TOOLS

```python
{
    "type": "function",
    "function": {
        "name": "control_light",
        "description": "Control a smart light bulb",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "state": {"type": "string", "enum": ["on", "off"]},
                "brightness": {"type": "integer"},
                "color": {"type": "string"}
            },
            "required": ["entity_id", "state"]
        }
    }
},
```

#### 3) Add to call_tool

```python
if name == "control_light":
    return control_light(**args)
```

### What Qwen now does

You can say:

> turn the office lamp warm orange at 30% brightness

Qwen emits:

```json
{
  "name": "control_light",
  "arguments": {
    "entity_id": "light.office_lamp",
    "state": "on",
    "brightness": 77,
    "color": "orange"
  }
}
```

No prompt tricks. No bash. Deterministic.

### If you are NOT using Home Assistant

Same tool works for:

- Hue bridge HTTP
- WLED
- Tasmota
- TP-Link Kasa
- Shelly

Only the Python inside `control_light` changes. The model schema stays identical.

### Architecture rule (very important)

| Capability | Tool | Never use |
|------------|------|-----------|
| Files | `read/write/edit_file` | bash |
| Shell | `bash` | files |
| Smart devices | `control_light` | bash |
| APIs | dedicated tool | bash |

This separation is what makes small models behave like large agents.

---

## What about Gemma 4 models?

**Goal:** Evaluate whether the new Gemma 4 models are good choices for tool calling in Ollama compared to Qwen.

### What Gemma 4 improved over Gemma 2

- Much stronger instruction tuning
- Better JSON / structured output discipline
- Exposure to tool / function style data
- Significantly better reasoning at small sizes

Gemma 2: fights tool schemas, drifts, needs heavy prompting
Gemma 4: willingly emits clean tool calls like Qwen

### Practical reality at ~9B size

| Model | Tool reliability | JSON discipline | Reasoning | Speed |
|-------|-----------------|-----------------|-----------|-------|
| Qwen2.5-7B | ★★★★★ | ★★★★★ | ★★★★ | ★★★★★ |
| Gemma 4 9B | ★★★★½ | ★★★★½ | ★★★★★ | ★★★★ |
| Llama 3.1 8B | ★★★★ | ★★★ | ★★★★★ | ★★★★ |

Gemma 4 is now **very close** to Qwen for tools, and **better at reasoning**.

### Where Gemma 4 is actually better than Qwen

Gemma 4 will:

- Plan multi-step tool sequences more intelligently
- Understand vague human intent better
- Recover from tool errors more gracefully

Qwen is still slightly more "robotic JSON perfect". Gemma 4 is more "agentic".

### The sweet spot model

```
ollama run gemma4:9b-instruct
```

One important setting (critical) — Gemma 4 needs slightly lower creativity than defaults:

```python
options={
    "temperature": 0,
    "top_p": 0.85,
    "repeat_penalty": 1.15,
}
```

### When to pick Gemma 4 over Qwen

Choose Gemma 4 if your agent:

- Controls devices (like your bulbs)
- Edits code across multiple files
- Needs planning before tool use
- Handles messy human requests

Choose Qwen if:

- You want absolute deterministic tool JSON every time
- You are building a strict coding agent

**Bottom line:** For your exact setup (files + bash + smart bulbs), Gemma 4 9B is now a first-class tool-calling model in Ollama and is arguably the best "general purpose local agent" under 10B.

---

## Same Tool Code for Qwen and Gemma?

Yes — the tool code is **identical**. You do not change tools, schemas, or the agent loop between models. Ollama normalizes tool calling. The model only decides **when** to call a tool.

You can swap models with one line:

```python
MODEL = "qwen2.5:7b-instruct"
# or
MODEL = "gemma4:9b-instruct"
```

Nothing else changes.

### The only difference: generation settings

For Qwen:

```python
options={
    "temperature": 0,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}
```

For Gemma 4:

```python
options={
    "temperature": 0,
    "top_p": 0.85,
    "repeat_penalty": 1.15,
}
```

---

## How Far From Defaults?

| Setting | Ollama default | Qwen tools | Gemma 4 tools |
|---------|---------------|------------|---------------|
| temperature | 0.8 | 0.0 | 0.0 |
| top_p | 0.9 | 0.9 | 0.85 |
| repeat_penalty | 1.0 | 1.1 | 1.15 |

Although the numbers look "close to each other", they are **very far from default behavior**.

### The big one: temperature 0.8 → 0.0

This is not a tweak. This is turning the model from "creative chat model" into "deterministic schema emitter".

At 0.8 the model will:

- Paraphrase JSON keys
- Add commentary
- Drift from schema
- "Explain" instead of calling tools

At 0.0 it behaves like a compiler.

### Why top_p barely changes

`top_p` controls token diversity **after** temperature. Once temperature is 0, `top_p` becomes a minor guardrail. Gemma needs it slightly lower because it has a stronger tendency to "embellish" tokens even at temp 0.

### Why repeat_penalty increases

Tool schemas contain repeated structural tokens like:

```json
"name": "control_light",
"arguments": {
```

Small models like to "loop" or re-emit JSON fragments. A slight repeat penalty prevents:

- Duplicated keys
- Partial JSON restarts
- Tool call loops

Gemma needs it a touch higher than Qwen because it's more verbose by nature.

### What happens if you run defaults

If you remove these options entirely in Ollama, you'll see:

- Random English before tool calls
- Malformed JSON
- Tools called less often
- Model tries to solve things without tools

This is why people think "small models are bad at tools". They're not. They're just running at chat settings.

### Mental model

You are not "tuning" the model. You are switching it from:

**Chat mode → Agent mode**

That's what these settings do.
