# Gemini Live API Examples

The Live API enables low-latency, real-time voice and video interactions with
Gemini. It processes continuous streams of audio, video, or text to deliver
immediate, human-like spoken responses, creating a natural conversational
experience for your users.

![Live API Overview](https://ai.google.dev/gemini-api/docs/images/live-api-overview.png)

[Try the Live API in Google AI Studio](https://aistudio.google.com/live)

## Example use cases

Live API can be used to build real-time voice and video agents for a
variety of industries, including:

*   **E-commerce and retail:** Shopping assistants that offer personalized
    recommendations and support agents that resolve customer issues.
*   **Gaming:** Interactive non-player characters (NPCs), in-game help
    assistants, and real-time translation of in-game content.
*   **Next-gen interfaces:** Voice- and video-enabled experiences in robotics,
    smart glasses, and vehicles.
*   **Healthcare:** Health companions for patient support and education.
*   **Financial services:** AI advisors for wealth management and investment
    guidance.
*   **Education:** AI mentors and learner companions that provide personalized
    instruction and feedback.

## Key features

Live API offers a comprehensive set of features for building
robust voice and video agents:

*   [**Multilingual support**](https://ai.google.dev/gemini-api/docs/live-guide#supported-languages):
    Converse in 70 supported languages.
*   [**Barge-in**](https://ai.google.dev/gemini-api/docs/live-guide#interruptions):
    Users can interrupt the model at any time for responsive interactions.
*   [**Tool use**](https://ai.google.dev/gemini-api/docs/live-tools):
    Integrates tools like function calling and Google Search for dynamic
    interactions.
*   [**Audio transcriptions**](https://ai.google.dev/gemini-api/docs/live-guide#audio-transcription):
    Provides text transcripts of both user input and model output.
*   [**Proactive audio**](https://ai.google.dev/gemini-api/docs/live-guide#proactive-audio):
    Lets you control when the model responds and in what contexts.
*   [**Affective dialog**](https://ai.google.dev/gemini-api/docs/live-guide#affective-dialog):
    Adapts response style and tone to match the user's input expression.

## Technical specifications

The following table outlines the technical specifications for the
Live API:

| Category          | Details                                                                                     |
| :---------------- | :------------------------------------------------------------------------------------------ |
| Input modalities  | Audio (raw 16-bit PCM audio, 16kHz, little-endian), images/video (JPEG <= 1FPS), text       |
| Output modalities | Audio (raw 16-bit PCM audio, 24kHz, little-endian), text                                    |
| Protocol          | Stateful WebSocket connection (WSS)                                                         |

## Examples

### Multi-Agent Voice Assistant (Python + Vanilla JS)

> [./gemini-live-genai-python-sdk/](./gemini-live-genai-python-sdk/)

A full-featured multi-agent voice assistant with a FastAPI backend and vanilla
JS frontend. Gemini Live handles voice I/O and routes tasks to pluggable backend
agents (Ollama or Claude Code) with automatic fallback.

**What it does:**

- Real-time voice/video conversation powered by Gemini Live
- Routes tasks to named backend agents configured in `agents.yaml`
- Supports Ollama (local models with tool calling) and Claude Code (cloud CLI)
- Background task execution with status tracking and notifications
- Browser-based agent CRUD UI (create, edit, delete, hot-reload)
- Multi-API-key rotation with automatic fallback
- Session usage tracking with estimated cost display

### Other examples

*   **[Ephemeral tokens and raw WebSocket example](./gemini-live-ephemeral-tokens-websocket/README.md)**: RAW protocol control. Connect to the Gemini Live API using WebSockets to build a real-time multimodal application with a JavaScript frontend and a Python backend.
*   **[Command-line Python example](./command-line/python/README.md)**: A minimal command-line app that streams microphone audio to the Gemini Live API and plays back the response in real time using Python.
*   **[Command-line Node.js example](./command-line/node/README.md)**: A minimal command-line app that streams microphone audio to the Gemini Live API and plays back the response in real time using Node.js.

> [!TIP]
> Install the [Gemini Live API Dev](https://github.com/google-gemini/gemini-skills?tab=readme-ov-file#gemini-live-api-dev) skill for AI-assisted development with the Live API in your coding agents.

---

## Multi-Agent Voice Assistant — Deep Dive

### Architecture

```
Browser  <--WebSocket-->  FastAPI (main.py)
                               |
                         Gemini Live API (gemini_live.py)
                               |
                         Tool calls (ask_agent, run_bash, etc.)
                               |
                         tools.py  -->  TaskManager  -->  background threads
                                                          |            |
                                                    Ollama API    Claude Code CLI
                                                   (tool loop)    (subprocess)
```

1. The browser captures mic/camera and streams PCM audio + JPEG frames to the
   FastAPI server over WebSocket.
2. The server forwards media to the Gemini Live API, which responds with
   synthesized speech and tool calls.
3. When Gemini calls `ask_agent`, tools.py dispatches the task to a
   background thread via TaskManager.
4. The thread either runs an Ollama tool-calling loop or shells out to
   the `claude` CLI, with fallback to an alternate backend on failure.
5. Results flow back: TaskManager notification -> async queue -> Gemini
   session -> browser WebSocket -> user sees/hears the result.

### Quick start

```bash
cd gemini-live-genai-python-sdk

# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the server
uv run main.py
# Or: python main.py
```

Open http://localhost:8000 in your browser.

### Configuration

#### Environment variables (.env)

| Variable           | Required | Default                       | Description                                        |
| :----------------- | :------- | :---------------------------- | :------------------------------------------------- |
| `GEMINI_API_KEY`   | Yes      | -                             | Primary Gemini API key                             |
| `GEMINI_API_KEYS`  | No       | -                             | Comma-separated keys for automatic rotation        |
| `MODEL`            | No       | `gemini-3.1-flash-live-preview` | Gemini model to use                              |
| `PORT`             | No       | `8000`                        | Server port                                        |

#### agents.yaml

The system is configured via `agents.yaml` with two sections:

**`gemini_session`** — Gemini Live connection settings:

```yaml
gemini_session:
  model: gemini-3.1-flash-live-preview
  voice: Puck
  system_prompt: |
    You are a voice assistant...
```

**`agents`** — Named backend agents with primary + fallback configs:

```yaml
agents:
  coder:
    description: Software engineering agent
    backend: claude-code
    model: glm-5.1
    timeout: 600
    system_prompt: |
      You are an expert software engineer...
    fallbacks:
      - backend: ollama
        model: qwen3.5:9b-64K
        timeout: 600
```

Each agent supports:

- `backend` — `ollama` or `claude-code`
- `model` — Model identifier for the backend
- `timeout` — Max seconds before task is killed
- `system_prompt` — Instructions for the agent
- `tools` — (Ollama only) List of registered tool names to enable
- `options` — (Ollama only) Generation parameters (temperature, top_p, etc.)
- `fallbacks` — Ordered list of alternate backends to try on failure

### Built-in Ollama tools

When using Ollama agents, these tools are available and registered automatically:

| Tool         | Description                           |
| :----------- | :------------------------------------ |
| `read_file`  | Read file contents                    |
| `write_file` | Write content to a file               |
| `edit_file`  | Find-and-replace in a file            |
| `bash`       | Run a shell command (30s timeout)     |
| `list_files` | List directory contents               |
| `web_fetch`  | Fetch a URL and return text           |
| `grep_files` | Search for a pattern in files         |
| `gws`        | Google Workspace CLI (Gmail, etc.)    |

### Gemini-facing tools

These are the tools Gemini Live can call during conversation:

| Tool          | Description                                           |
| :------------ | :---------------------------------------------------- |
| `ask_agent`   | Route a task to a named backend agent                 |
| `run_bash`    | Execute a shell command directly                      |
| `list_tasks`  | List background tasks and their status                |
| `cancel_task` | Cancel a running background task                      |

### REST API endpoints

| Endpoint            | Method | Description                  |
| :------------------ | :----- | :--------------------------- |
| `/`                 | GET    | Serve the frontend           |
| `/ws`               | WS     | WebSocket for media + events |
| `/api/agents`       | GET    | List all agents              |
| `/api/agents`       | POST   | Create a new agent           |
| `/api/agents/{name}`| PUT    | Update an agent              |
| `/api/agents/{name}`| DELETE | Delete an agent              |
| `/api/tasks`        | GET    | List background tasks        |
| `/api/usage`        | GET    | Get token usage stats        |
| `/api/gemini-config`| GET    | Get Gemini session config    |
| `/api/ollama-tools` | GET    | List registered Ollama tools |

### Project structure

```
gemini-live-genai-python-sdk/
├── main.py              # FastAPI app, WebSocket bridge, API endpoints
├── gemini_live.py       # GeminiLive class — SDK wrapper for Live API
├── tools.py             # Gemini-facing tools + agent dispatch logic
├── task_manager.py      # Background task runner with fallback support
├── agent_config.py      # CRUD for agents.yaml + Gemini session config
├── ollama_tools.py      # Ollama tool registry (decorator pattern)
├── demo_3agent.py       # Standalone 3-agent orchestration demo
├── agents.yaml          # Agent definitions + Gemini session config
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── scripts/
│   └── ask_ollama       # Quick CLI to query an Ollama model
└── frontend/
    ├── index.html       # Single-page app shell
    ├── main.js          # App controller, WebSocket events, UI logic
    ├── gemini-client.js # WebSocket client for server communication
    ├── media-handler.js # Audio capture/playback, video framing
    ├── pcm-processor.js # AudioWorklet for PCM audio processing
    ├── agent-config.js  # Agent CRUD panel UI
    └── style.css        # Full stylesheet with responsive layout
```

### Dependencies

- **fastapi** — Web framework and WebSocket server
- **uvicorn** — ASGI server
- **google-genai** — Google Gen AI Python SDK
- **websockets** — WebSocket client library
- **python-dotenv** — .env file loading
- **python-multipart** — Form data parsing
- **pyyaml** — YAML parsing for agents.yaml (install separately)

> **Note:** `pyyaml` is used but not listed in `requirements.txt`. Install it
> with `pip install pyyaml`.

## Partner integrations

To streamline the development of real-time audio and video apps, you can use
a third-party integration that supports the Gemini Live
API over WebRTC or WebSockets.

*   [LiveKit](https://docs.livekit.io/agents/models/realtime/plugins/gemini/): Use the Gemini Live API with LiveKit Agents.
*   [Pipecat by Daily](https://docs.pipecat.ai/guides/features/gemini-live): Create a real-time AI chatbot using Gemini Live and Pipecat.
*   [Fishjam by Software Mansion](https://docs.fishjam.io/tutorials/gemini-live-integration): Create live video and audio streaming applications with Fishjam.
*   [Vision Agents by Stream](https://visionagents.ai/integrations/gemini): Build real-time voice and video AI applications with Vision Agents.
*   [Voximplant](https://voximplant.com/products/gemini-client): Connect inbound and outbound calls to Live API with Voximplant.
*   [Agent Development Kit (ADK)](https://google.github.io/adk-docs/streaming/): Create an agent and use the Agent Development Kit (ADK) Streaming to enable voice and video communication.
*   [Firebase AI SDK](https://firebase.google.com/docs/ai-logic/live-api?api=dev): Get started with the Gemini Live API using Firebase AI Logic.
