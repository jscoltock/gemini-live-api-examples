// --- Main Application Logic ---

const statusDiv = document.getElementById("status");
const authSection = document.getElementById("auth-section");
const appSection = document.getElementById("app-section");
const sessionEndSection = document.getElementById("session-end-section");
const restartBtn = document.getElementById("restartBtn");
const micBtn = document.getElementById("micBtn");
const voiceBtn = document.getElementById("voiceBtn");
const cameraBtn = document.getElementById("cameraBtn");
const screenBtn = document.getElementById("screenBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const videoPreview = document.getElementById("video-preview");
const videoPlaceholder = document.getElementById("video-placeholder");
const connectBtn = document.getElementById("connectBtn");
const chatLog = document.getElementById("chat-log");
const agentList = document.getElementById("agent-list");

let currentGeminiMessageDiv = null;
let currentUserMessageDiv = null;
let voiceEnabled = false;
let agentTasks = {};  // taskId -> {element, status, startTime}
let agentConfigs = {};  // agent name -> {backend, model, timeout}
let pollInterval = null;
let usageData = { prompt_tokens: 0, response_tokens: 0, total_tokens: 0, turns: 0, model: "--" };

const mediaHandler = new MediaHandler();
const geminiClient = new GeminiClient({
  onOpen: () => {
    statusDiv.textContent = "Connected";
    statusDiv.className = "status connected";
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");

    // Start polling for task status updates
    pollInterval = setInterval(pollTasks, 2000);

    // Fetch agent configs for display
    fetch("/api/agents").then(r => r.json()).then(agents => {
      agentConfigs = {};
      for (const a of agents) {
        agentConfigs[a.name] = a;
      }
    }).catch(() => {});

    // Fetch Gemini config (model name) for usage panel
    fetch("/api/gemini-config").then(r => r.json()).then(cfg => {
      usageData.model = cfg.model || "--";
      updateUsagePanel();
    }).catch(() => {});

    // Reset usage counters
    usageData = { prompt_tokens: 0, response_tokens: 0, total_tokens: 0, turns: 0, model: usageData.model };
    updateUsagePanel();

    // Send hidden instruction
    geminiClient.sendText(
      `System: Introduce yourself as a demo of the Gemini Live API.
       Suggest playing with features like the native audio for accents and multilingual support.
       Keep the intro concise and friendly.`
    );
  },
  onMessage: (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        handleJsonMessage(msg);
      } catch (e) {
        console.error("Parse error:", e);
      }
    } else {
      if (voiceEnabled) {
        mediaHandler.playAudio(event.data);
      }
    }
  },
  onClose: (e) => {
    console.log("WS Closed:", e);
    statusDiv.textContent = "Disconnected";
    statusDiv.className = "status disconnected";
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    showSessionEnd();
  },
  onError: (e) => {
    console.error("WS Error:", e);
    statusDiv.textContent = "Connection Error";
    statusDiv.className = "status error";
  },
});

function handleJsonMessage(msg) {
  if (msg.type === "interrupted") {
    mediaHandler.stopAudioPlayback();
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "turn_complete") {
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
    usageData.turns = (usageData.turns || 0) + 1;
    updateUsagePanel();
  } else if (msg.type === "user") {
    if (currentUserMessageDiv) {
      currentUserMessageDiv.textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage("user", msg.text);
    }
  } else if (msg.type === "gemini") {
    if (currentGeminiMessageDiv) {
      currentGeminiMessageDiv.textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentGeminiMessageDiv = appendMessage("gemini", msg.text);
    }
  } else if (msg.type === "tool_call") {
    // Agent dispatched — create card in agent panel
    if (msg.name === "ask_agent" && msg.args) {
      const result = msg.result || "";
      // Parse task ID from "Task abc123 started..."
      const match = result.match(/Task (\w+) started/);
      if (match) {
        addAgentTask(match[1], msg.args.agent || "unknown", msg.args.prompt || "");
      }
    }
  } else if (msg.type === "usage") {
    // Update usage from WebSocket events
    const u = msg.usage || {};
    usageData.prompt_tokens = u.prompt_token_count || usageData.prompt_tokens;
    usageData.response_tokens = u.response_token_count || usageData.response_tokens;
    usageData.total_tokens = u.total_token_count || usageData.total_tokens;
    updateUsagePanel();
  }
}

function appendMessage(type, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.textContent = text;
  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msgDiv;
}

// --- Agent Panel ---

function addAgentTask(taskId, agent, prompt) {
  const cfg = agentConfigs[agent] || {};
  const meta = [];
  if (cfg.backend) meta.push(cfg.backend);
  if (cfg.model) meta.push(cfg.model);
  if (cfg.timeout) meta.push(`${cfg.timeout}s`);

  const card = document.createElement("div");
  card.className = "agent-task running";
  card.id = `task-${taskId}`;
  card.innerHTML = `
    <div>
      <span class="task-agent">${escapeHtml(agent)}</span>
      <span class="task-id">${taskId}</span>
      <span class="task-status status-running">running</span>
    </div>
    <div class="task-meta">${escapeHtml(meta.join(" | "))}</div>
    <div class="task-prompt">${escapeHtml(prompt)}</div>
    <div class="task-time">${new Date().toLocaleTimeString()}</div>
  `;
  agentList.prepend(card);
  agentTasks[taskId] = { element: card, status: "running", startTime: Date.now() };
}

function updateAgentTask(taskId, status, output) {
  const task = agentTasks[taskId];
  if (!task) {
    // Task came from poll, create card
    const card = document.createElement("div");
    card.className = `agent-task ${status}`;
    card.id = `task-${taskId}`;
    agentList.prepend(card);
    task = { element: card, status: status, startTime: Date.now() };
    agentTasks[taskId] = task;
  }

  task.status = status;
  const card = task.element;
  card.className = `agent-task ${status}`;

  // Update status badge
  const statusEl = card.querySelector(".task-status");
  if (statusEl) {
    statusEl.className = `task-status status-${status}`;
    statusEl.textContent = status;
  }

  // Show output if present
  if (output) {
    let outputEl = card.querySelector(".task-output");
    if (!outputEl) {
      outputEl = document.createElement("div");
      card.appendChild(outputEl);
    }
    outputEl.className = "task-output";
    outputEl.textContent = output.substring(0, 500);
  }

  // Show elapsed time
  const elapsed = Math.round((Date.now() - task.startTime) / 1000);
  const timeEl = card.querySelector(".task-time");
  if (timeEl) {
    timeEl.textContent = `${new Date().toLocaleTimeString()} (${elapsed}s)`;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// --- Usage Panel ---

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function updateUsagePanel() {
  const el = (id) => document.getElementById(id);
  const input = usageData.prompt_tokens || 0;
  const output = usageData.response_tokens || 0;
  const total = usageData.total_tokens || 0;

  // Rough cost estimate: $0.30/M input, $2.50/M output (Flash Live rates)
  const cost = (input * 0.30 + output * 2.50) / 1_000_000;

  el("usage-model").textContent = usageData.model || "--";
  el("usage-turns").textContent = usageData.turns || 0;
  el("usage-input").textContent = formatNumber(input);
  el("usage-output").textContent = formatNumber(output);
  el("usage-cost").textContent = "$" + cost.toFixed(4);
}

async function pollTasks() {
  try {
    const res = await fetch("/api/tasks");
    if (!res.ok) return;
    const tasks = await res.json();
    for (const t of tasks) {
      const existing = agentTasks[t.id];
      if (!existing || existing.status !== t.status) {
        if (!existing) {
          // New task from poll — create card with available info
          const card = document.createElement("div");
          card.className = `agent-task ${t.status}`;
          card.id = `task-${t.id}`;
          card.innerHTML = `
            <div>
              <span class="task-agent">${t.agent || "unknown"}</span>
              <span class="task-id">${t.id}</span>
              <span class="task-status status-${t.status}">${t.status}</span>
            </div>
            <div class="task-prompt">${escapeHtml((t.command || "").substring(0, 100))}</div>
            <div class="task-time">${new Date().toLocaleTimeString()}</div>
          `;
          if (t.output) {
            const outputEl = document.createElement("div");
            outputEl.className = "task-output";
            outputEl.textContent = t.output.substring(0, 500);
            card.appendChild(outputEl);
          }
          agentList.prepend(card);
          agentTasks[t.id] = { element: card, status: t.status, startTime: Date.now() };
        } else {
          updateAgentTask(t.id, t.status, t.output);
        }
      }
    }
  } catch (e) {
    // Silently ignore — polling is best-effort
  }
}

// Connect Button Handler
connectBtn.onclick = async () => {
  statusDiv.textContent = "Connecting...";
  connectBtn.disabled = true;

  try {
    // Initialize audio context on user gesture
    await mediaHandler.initializeAudio();

    geminiClient.connect();
  } catch (error) {
    console.error("Connection error:", error);
    statusDiv.textContent = "Connection Failed: " + error.message;
    statusDiv.className = "status error";
    connectBtn.disabled = false;
  }
};

// UI Controls
disconnectBtn.onclick = () => {
  geminiClient.disconnect();
};

micBtn.onclick = async () => {
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.textContent = "Start Mic";
  } else {
    try {
      await mediaHandler.startAudio((data) => {
        if (geminiClient.isConnected()) {
          geminiClient.send(data);
        }
      });
      micBtn.textContent = "Stop Mic";
    } catch (e) {
      alert("Could not start audio capture");
    }
  }
};

voiceBtn.onclick = () => {
  voiceEnabled = !voiceEnabled;
  voiceBtn.textContent = voiceEnabled ? "Stop Voice" : "Start Voice";
  voiceBtn.classList.toggle("active", voiceEnabled);
};

cameraBtn.onclick = async () => {
  if (cameraBtn.textContent === "Stop Camera") {
    mediaHandler.stopVideo(videoPreview);
    cameraBtn.textContent = "Start Camera";
    screenBtn.textContent = "Share Screen";
    videoPlaceholder.classList.remove("hidden");
  } else {
    // If another stream is active (e.g. Screen), stop it first
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      screenBtn.textContent = "Share Screen";
    }

    try {
      await mediaHandler.startVideo(videoPreview, (base64Data) => {
        if (geminiClient.isConnected()) {
          geminiClient.sendImage(base64Data);
        }
      });
      cameraBtn.textContent = "Stop Camera";
      screenBtn.textContent = "Share Screen";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      alert("Could not access camera");
    }
  }
};

screenBtn.onclick = async () => {
  if (screenBtn.textContent === "Stop Sharing") {
    mediaHandler.stopVideo(videoPreview);
    screenBtn.textContent = "Share Screen";
    cameraBtn.textContent = "Start Camera";
    videoPlaceholder.classList.remove("hidden");
  } else {
    // If another stream is active (e.g. Camera), stop it first
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      cameraBtn.textContent = "Start Camera";
    }

    try {
      await mediaHandler.startScreen(
        videoPreview,
        (base64Data) => {
          if (geminiClient.isConnected()) {
            geminiClient.sendImage(base64Data);
          }
        },
        () => {
          // onEnded callback (e.g. user stopped sharing from browser)
          screenBtn.textContent = "Share Screen";
          videoPlaceholder.classList.remove("hidden");
        }
      );
      screenBtn.textContent = "Stop Sharing";
      cameraBtn.textContent = "Start Camera";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      alert("Could not share screen");
    }
  }
};

sendBtn.onclick = sendText;
textInput.onkeypress = (e) => {
  if (e.key === "Enter") sendText();
};

function sendText() {
  const text = textInput.value;
  if (text && geminiClient.isConnected()) {
    geminiClient.sendText(text);
    appendMessage("user", text);
    textInput.value = "";
  }
}

function resetUI() {
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");

  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  videoPlaceholder.classList.remove("hidden");

  micBtn.textContent = "Start Mic";
  voiceBtn.textContent = "Start Voice";
  voiceBtn.classList.remove("active");
  voiceEnabled = false;
  cameraBtn.textContent = "Start Camera";
  screenBtn.textContent = "Share Screen";
  chatLog.innerHTML = "";
  agentList.innerHTML = "";
  agentTasks = {};
  connectBtn.disabled = false;
}

function showSessionEnd() {
  appSection.classList.add("hidden");
  sessionEndSection.classList.remove("hidden");
  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
}

restartBtn.onclick = () => {
  resetUI();
};
