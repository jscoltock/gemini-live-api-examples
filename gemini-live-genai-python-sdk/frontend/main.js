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
const modelSelect = document.getElementById("modelSelect");
const newChatBtn = document.getElementById("newChatBtn");

let currentGeminiMessageDiv = null;
let currentUserMessageDiv = null;
let voiceEnabled = false;
let agentTasks = {};  // taskId -> {element, status, startTime, trace}
let agentConfigs = {};  // agent name -> {backend, model, timeout}
let pollInterval = null;
let usageData = { prompt_tokens: 0, response_tokens: 0, total_tokens: 0, turns: 0, model: "--" };
let isStreaming = false; // guard against concurrent SSE requests

// --- Mode state ---
// 'gemini-live' = WebSocket real-time, 'glm-5.1' or 'qwen3.5:9b-64K' = SSE turn-based
let currentModel = localStorage.getItem("selected_model") || "gemini-live";

// --- Conversation history (localStorage) ---
const HISTORY_KEY = "chat_history";
const MAX_HISTORY = 100;

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch { return []; }
}

function saveHistory(messages) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-MAX_HISTORY)));
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  chatLog.innerHTML = "";
  currentGeminiMessageDiv = null;
  currentUserMessageDiv = null;
}

function historyToMessages() {
  // Convert chat history to OpenAI-style messages array for API calls
  const history = loadHistory();
  return history.map(m => ({ role: m.role, content: m.content }));
}

function renderHistory() {
  chatLog.innerHTML = "";
  const history = loadHistory();
  for (const m of history) {
    appendMessage(m.role === "user" ? "user" : "gemini", m.content);
  }
}

// Accumulated totals persist across sessions (localStorage)
let accumulated = JSON.parse(localStorage.getItem("usage_accumulated") || "null") || {
  totalCost: 0,
  totalInput: 0,
  totalOutput: 0,
  totalTurns: 0,
  since: new Date().toISOString(),
};

function saveAccumulated() {
  localStorage.setItem("usage_accumulated", JSON.stringify(accumulated));
}

// --- Init model selector ---
modelSelect.value = currentModel;

// --- Web Speech API (input for non-Live models) ---
let speechRecognition = null;
let isListening = false;

function initSpeechRecognition(onResult) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return null;

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  let finalTranscript = "";

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += t;
      } else {
        interim += t;
      }
    }
    // Show interim results in chat as "listening" indicator
    if (interim) {
      updateListeningIndicator(interim);
    }
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.textContent = "Start Mic";
    micBtn.classList.remove("active");
    removeListeningIndicator();
    if (finalTranscript.trim()) {
      onResult(finalTranscript.trim());
      finalTranscript = "";
    }
  };

  recognition.onerror = (e) => {
    console.warn("SpeechRecognition error:", e.error);
    isListening = false;
    micBtn.textContent = "Start Mic";
    micBtn.classList.remove("active");
    removeListeningIndicator();
    if (e.error === "no-speech") return; // silent
    if (finalTranscript.trim()) {
      onResult(finalTranscript.trim());
      finalTranscript = "";
    }
  };

  return recognition;
}

let listeningIndicator = null;

function updateListeningIndicator(text) {
  if (!listeningIndicator) {
    listeningIndicator = document.createElement("div");
    listeningIndicator.className = "message user listening";
    chatLog.appendChild(listeningIndicator);
  }
  listeningIndicator.textContent = "🎤 " + text;
  chatLog.scrollTop = chatLog.scrollHeight;
}

function removeListeningIndicator() {
  if (listeningIndicator) {
    listeningIndicator.remove();
    listeningIndicator = null;
  }
}

// --- SpeechSynthesis (output for non-Live models) ---
let currentUtterance = null;

function speakText(text) {
  if (!voiceEnabled) return;
  window.speechSynthesis.cancel(); // stop any previous
  currentUtterance = new SpeechSynthesisUtterance(text);
  currentUtterance.rate = 1.0;
  currentUtterance.onend = () => { currentUtterance = null; };
  window.speechSynthesis.speak(currentUtterance);
}

function stopSpeaking() {
  window.speechSynthesis.cancel();
  currentUtterance = null;
}

// --- SSE Chat for non-Live models ---
async function sendChatMessage(text) {
  if (isStreaming) return;
  isStreaming = true;

  // Add user message to history
  const history = loadHistory();
  history.push({ role: "user", content: text });
  saveHistory(history);

  appendMessage("user", text);
  textInput.value = "";

  // Create assistant message div
  currentGeminiMessageDiv = appendMessage("gemini", "");
  let fullResponse = "";

  try {
    const messages = history.map(m => ({ role: m.role, content: m.content }));
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: currentModel, messages }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") break;
        try {
          const chunk = JSON.parse(data);
          if (chunk.error) {
            currentGeminiMessageDiv.textContent += " [Error: " + chunk.error + "]";
            break;
          }
          if (chunk.content) {
            fullResponse += chunk.content;
            currentGeminiMessageDiv.textContent = fullResponse;
            chatLog.scrollTop = chatLog.scrollHeight;
          }
        } catch {}
      }
    }
  } catch (e) {
    console.error("Chat error:", e);
    if (currentGeminiMessageDiv) {
      currentGeminiMessageDiv.textContent += " [Error: " + e.message + "]";
    }
  }

  // Save assistant response to history
  if (fullResponse) {
    const h = loadHistory();
    h.push({ role: "assistant", content: fullResponse });
    saveHistory(h);
    speakText(fullResponse);
  }

  currentGeminiMessageDiv = null;
  usageData.turns = (usageData.turns || 0) + 1;
  updateUsagePanel();
  isStreaming = false;
}

// --- Mode switching ---
function isLiveMode() {
  return currentModel === "gemini-live";
}

function switchModel(newModel) {
  if (newModel === currentModel) return;

  // Disconnect current session if active
  if (geminiClient.isConnected()) {
    geminiClient.disconnect();
  }
  stopSpeaking();
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }

  currentModel = newModel;
  localStorage.setItem("selected_model", currentModel);
  modelSelect.value = currentModel;

  // Reset to auth section
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");
  statusDiv.textContent = "Disconnected";
  statusDiv.className = "status disconnected";
  connectBtn.disabled = false;

  // Update description based on mode
  updateAuthDescription();
}

function updateAuthDescription() {
  const descBox = authSection.querySelector(".description-box");
  if (!descBox) return;

  if (isLiveMode()) {
    descBox.innerHTML = `
      <h3>Features Enabled:</h3>
      <ul style="margin: 10px 0 20px 20px">
        <li><strong>Native Audio:</strong> Low latency voice interaction</li>
        <li><strong>Multilingual:</strong> Speak in different languages</li>
      </ul>
      <p><em>Note: Real-time bidirectional audio with interruptibility.</em></p>
    `;
  } else {
    const label = modelSelect.options[modelSelect.selectedIndex].text;
    descBox.innerHTML = `
      <h3>${escapeHtml(label)}</h3>
      <ul style="margin: 10px 0 20px 20px">
        <li><strong>Voice Input:</strong> Push-to-talk via browser speech recognition</li>
        <li><strong>Voice Output:</strong> Browser text-to-speech (enable with Voice button)</li>
        <li><strong>Text Chat:</strong> Full streaming text responses</li>
      </ul>
      <p><em>Turn-based conversation — no interrupting mid-response.</em></p>
    `;
  }
}

// --- MediaHandler & GeminiClient init ---
const mediaHandler = new MediaHandler();
const geminiClient = new GeminiClient({
  onOpen: () => {
    statusDiv.textContent = "Connected";
    statusDiv.className = "status connected";
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");

    pollInterval = setInterval(pollTasks, 2000);

    fetch("/api/agents").then(r => r.json()).then(agents => {
      agentConfigs = {};
      for (const a of agents) {
        agentConfigs[a.name] = a;
      }
    }).catch(() => {});

    fetch("/api/gemini-config").then(r => r.json()).then(cfg => {
      usageData.model = cfg.model || "--";
      updateUsagePanel();
    }).catch(() => {});

    usageData = { prompt_tokens: 0, response_tokens: 0, total_tokens: 0, turns: 0, model: usageData.model };
    accumulated._lastInput = 0;
    accumulated._lastOutput = 0;
    updateUsagePanel();

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
    if (msg.name === "ask_agent" && msg.args) {
      const result = msg.result || "";
      const match = result.match(/Task (\w+) started/);
      if (match) {
        addAgentTask(match[1], msg.args.agent || "unknown", msg.args.prompt || "");
      }
    }
  } else if (msg.type === "usage") {
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
    <div class="task-header" data-task-id="${taskId}">
      <span class="task-agent">${escapeHtml(agent)}</span>
      <span class="task-id">${taskId}</span>
      <span class="task-status status-running">running</span>
      <span class="trace-toggle" title="Toggle trace">&#9654;</span>
    </div>
    <div class="task-meta">${escapeHtml(meta.join(" | "))}</div>
    <div class="task-prompt">${escapeHtml(prompt)}</div>
    <div class="task-time">${new Date().toLocaleTimeString()}</div>
    <div class="trace-container hidden" id="trace-${taskId}"></div>
  `;
  agentList.prepend(card);
  agentTasks[taskId] = { element: card, status: "running", startTime: Date.now(), trace: [] };

  card.querySelector(".task-header").addEventListener("click", () => {
    const traceEl = document.getElementById(`trace-${taskId}`);
    traceEl.classList.toggle("hidden");
    const toggle = card.querySelector(".trace-toggle");
    toggle.textContent = traceEl.classList.contains("hidden") ? "\u25B6" : "\u25BC";
  });
}

function updateAgentTask(taskId, status, output) {
  const task = agentTasks[taskId];
  if (!task) {
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

  const statusEl = card.querySelector(".task-status");
  if (statusEl) {
    statusEl.className = `task-status status-${status}`;
    statusEl.textContent = status;
  }

  if (output) {
    let outputEl = card.querySelector(".task-output");
    if (!outputEl) {
      outputEl = document.createElement("div");
      card.appendChild(outputEl);
    }
    outputEl.className = "task-output";
    outputEl.textContent = output.substring(0, 500);
  }

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

// --- Trace Rendering ---

function renderTrace(taskId, traceEvents) {
  const traceEl = document.getElementById(`trace-${taskId}`);
  if (!traceEl || !traceEvents || !traceEvents.length) return;

  const task = agentTasks[taskId];
  if (task) task.trace = traceEvents;

  traceEl.innerHTML = "";
  for (const evt of traceEvents) {
    const item = document.createElement("div");
    item.className = `trace-item trace-${evt.type}`;

    const ts = evt.ts ? new Date(evt.ts).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"}) : "";
    const badge = `<span class="trace-badge trace-badge-${evt.type}">${evt.type.replace("_"," ")}</span>`;

    let content = "";
    const d = evt.data || {};

    switch (evt.type) {
      case "thinking":
        content = `<div class="trace-text">${escapeHtml(d.content || "")}</div>`;
        break;
      case "tool_call":
        content = `<div class="trace-tool-name">${escapeHtml(d.name || "?")}(${escapeHtml(JSON.stringify(d.args || {}))})</div>`;
        break;
      case "tool_result":
        content = `<div class="trace-tool-result"><span class="trace-result-label">${escapeHtml(d.name || "?")}:</span> ${escapeHtml((d.result || "").substring(0, 500))}</div>`;
        break;
      case "response":
        content = `<div class="trace-text">${escapeHtml(d.content || "")}</div>`;
        break;
      case "attempt":
        content = `<div class="trace-attempt">${d.fallback ? "fallback" : "primary"}: ${escapeHtml(d.label || "?")}</div>`;
        break;
      case "error":
        content = `<div class="trace-error">${escapeHtml(d.message || "unknown error")}</div>`;
        break;
      default:
        content = `<div class="trace-text">${escapeHtml(JSON.stringify(d).substring(0, 200))}</div>`;
    }

    item.innerHTML = `<span class="trace-ts">${ts}</span>${badge}${content}`;
    traceEl.appendChild(item);
  }
  traceEl.scrollTop = traceEl.scrollHeight;
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

  const sessionCost = (input * 0.30 + output * 2.50) / 1_000_000;

  const dInput = input - (accumulated._lastInput || 0);
  const dOutput = output - (accumulated._lastOutput || 0);
  const dCost = (dInput * 0.30 + dOutput * 2.50) / 1_000_000;
  if (dInput > 0 || dOutput > 0) {
    accumulated.totalInput += dInput;
    accumulated.totalOutput += dOutput;
    accumulated.totalCost += dCost;
    accumulated._lastInput = input;
    accumulated._lastOutput = output;
    saveAccumulated();
  }

  el("usage-model").textContent = isLiveMode() ? (usageData.model || "--") : currentModel;
  el("usage-turns").textContent = usageData.turns || 0;
  el("usage-input").textContent = formatNumber(input);
  el("usage-output").textContent = formatNumber(output);
  el("usage-cost").textContent = "$" + sessionCost.toFixed(4);

  el("usage-total-cost").textContent = "$" + accumulated.totalCost.toFixed(4);
  const since = new Date(accumulated.since);
  el("usage-since").textContent = since.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function pollTasks() {
  try {
    const res = await fetch("/api/tasks");
    if (!res.ok) return;
    const tasks = await res.json();
    for (const t of tasks) {
      if (window._clearedTaskIds && window._clearedTaskIds.has(t.id)) continue;
      const existing = agentTasks[t.id];
      if (!existing || existing.status !== t.status) {
        if (!existing) {
          const card = document.createElement("div");
          card.className = `agent-task ${t.status}`;
          card.id = `task-${t.id}`;
          card.innerHTML = `
            <div class="task-header" data-task-id="${t.id}">
              <span class="task-agent">${t.agent || "unknown"}</span>
              <span class="task-id">${t.id}</span>
              <span class="task-status status-${t.status}">${t.status}</span>
              ${(t.trace && t.trace.length) ? '<span class="trace-toggle" title="Toggle trace">&#9654;</span>' : ''}
            </div>
            <div class="task-prompt">${escapeHtml((t.command || "").substring(0, 100))}</div>
            <div class="task-time">${new Date().toLocaleTimeString()}</div>
            <div class="trace-container hidden" id="trace-${t.id}"></div>
          `;
          if (t.output) {
            const outputEl = document.createElement("div");
            outputEl.className = "task-output";
            outputEl.textContent = t.output.substring(0, 500);
            card.appendChild(outputEl);
          }
          agentList.prepend(card);
          agentTasks[t.id] = { element: card, status: t.status, startTime: Date.now(), trace: t.trace || [] };

          card.querySelector(".task-header").addEventListener("click", () => {
            const traceEl = document.getElementById(`trace-${t.id}`);
            traceEl.classList.toggle("hidden");
            const toggle = card.querySelector(".trace-toggle");
            if (toggle) toggle.textContent = traceEl.classList.contains("hidden") ? "\u25B6" : "\u25BC";
          });

          if (t.trace && t.trace.length) {
            renderTrace(t.id, t.trace);
          }
        } else {
          updateAgentTask(t.id, t.status, t.output);
        }
      }

      if (t.trace && t.trace.length) {
        renderTrace(t.id, t.trace);
      }
    }
  } catch (e) {
    // Silently ignore — polling is best-effort
  }
}

// --- Connect Button Handler ---
connectBtn.onclick = async () => {
  statusDiv.textContent = "Connecting...";
  connectBtn.disabled = true;

  if (isLiveMode()) {
    // Original Gemini Live WebSocket connection
    try {
      await mediaHandler.initializeAudio();
      geminiClient.connect();
    } catch (error) {
      console.error("Connection error:", error);
      statusDiv.textContent = "Connection Failed: " + error.message;
      statusDiv.className = "status error";
      connectBtn.disabled = false;
    }
  } else {
    // Non-Live model — just show the app section immediately
    statusDiv.textContent = "Connected";
    statusDiv.className = "status connected";
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");
    connectBtn.disabled = false;

    usageData = { prompt_tokens: 0, response_tokens: 0, total_tokens: 0, turns: 0, model: currentModel };
    updateUsagePanel();

    // Start task polling + fetch agent configs (same as Live onOpen)
    pollInterval = setInterval(pollTasks, 2000);
    fetch("/api/agents").then(r => r.json()).then(agents => {
      agentConfigs = {};
      for (const a of agents) {
        agentConfigs[a.name] = a;
      }
    }).catch(() => {});

    // Render existing history
    renderHistory();

    // Hide camera/screen buttons for non-Live
    cameraBtn.style.display = "none";
    screenBtn.style.display = "none";
  }
};

// --- Model selector ---
modelSelect.onchange = () => {
  switchModel(modelSelect.value);
  updateAuthDescription();
};

// --- New Chat button ---
newChatBtn.onclick = () => {
  clearHistory();
  usageData.turns = 0;
  updateUsagePanel();
  if (!isLiveMode()) {
    // Nothing else to do, just cleared
  }
};

// --- UI Controls ---
disconnectBtn.onclick = () => {
  if (isLiveMode()) {
    geminiClient.disconnect();
  }
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  stopSpeaking();
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");
  statusDiv.textContent = "Disconnected";
  statusDiv.className = "status disconnected";
  connectBtn.disabled = false;
  cameraBtn.style.display = "";
  screenBtn.style.display = "";
};

micBtn.onclick = async () => {
  if (isLiveMode()) {
    // Original Gemini Live mic behavior
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
  } else {
    // Non-Live: Web Speech API push-to-talk
    if (isListening) {
      // Stop listening and send what we have
      if (speechRecognition) {
        speechRecognition.stop();
      }
      return;
    }

    if (!speechRecognition) {
      speechRecognition = initSpeechRecognition((transcript) => {
        if (transcript.trim()) {
          sendChatMessage(transcript.trim());
        }
      });
    }

    if (!speechRecognition) {
      alert("Speech recognition not supported in this browser");
      return;
    }

    try {
      isListening = true;
      micBtn.textContent = "Listening...";
      micBtn.classList.add("active");
      speechRecognition.start();
    } catch (e) {
      console.warn("SpeechRecognition start error:", e);
      isListening = false;
      micBtn.textContent = "Start Mic";
      micBtn.classList.remove("active");
    }
  }
};

voiceBtn.onclick = () => {
  voiceEnabled = !voiceEnabled;
  voiceBtn.textContent = voiceEnabled ? "Stop Voice" : "Start Voice";
  voiceBtn.classList.toggle("active", voiceEnabled);
  if (!voiceEnabled) {
    stopSpeaking();
  }
};

cameraBtn.onclick = async () => {
  if (cameraBtn.textContent === "Stop Camera") {
    mediaHandler.stopVideo(videoPreview);
    cameraBtn.textContent = "Start Camera";
    screenBtn.textContent = "Share Screen";
    videoPlaceholder.classList.remove("hidden");
  } else {
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
  if (!text) return;

  if (isLiveMode()) {
    if (geminiClient.isConnected()) {
      sendPendingFiles();
      geminiClient.sendText(text);
      appendMessage("user", text);
      textInput.value = "";
    }
  } else {
    sendChatMessage(text);
  }
}

// --- File Upload ---

const fileInput = document.getElementById("fileInput");
const attachBtn = document.getElementById("attachBtn");
const filePreviewBar = document.getElementById("filePreviewBar");
let pendingFiles = [];

attachBtn.onclick = () => fileInput.click();

fileInput.onchange = () => {
  for (const file of fileInput.files) {
    if (pendingFiles.length >= 5) break;
    const reader = new FileReader();
    reader.onload = () => {
      pendingFiles.push({
        file,
        dataUrl: reader.result,
        mimeType: file.type || "application/octet-stream",
        name: file.name,
      });
      renderFilePreview();
    };
    reader.readAsDataURL(file);
  }
  fileInput.value = "";
};

function renderFilePreview() {
  if (!pendingFiles.length) {
    filePreviewBar.classList.add("hidden");
    filePreviewBar.innerHTML = "";
    return;
  }
  filePreviewBar.classList.remove("hidden");
  filePreviewBar.innerHTML = pendingFiles.map((f, i) => {
    const isImage = f.mimeType.startsWith("image/");
    const thumb = isImage
      ? `<img src="${f.dataUrl}" class="file-thumb" />`
      : `<span class="file-icon">${f.name.split(".").pop()}</span>`;
    return `<div class="file-preview-item">
      ${thumb}
      <span class="file-name">${escapeHtml(f.name)}</span>
      <button class="file-remove" data-idx="${i}" title="Remove">&times;</button>
    </div>`;
  }).join("");

  filePreviewBar.querySelectorAll(".file-remove").forEach(btn => {
    btn.onclick = () => {
      pendingFiles.splice(parseInt(btn.dataset.idx), 1);
      renderFilePreview();
    };
  });
}

function sendPendingFiles() {
  if (!geminiClient.isConnected()) return;
  for (const f of pendingFiles) {
    const base64 = f.dataUrl.split(",")[1];
    geminiClient.sendFile(base64, f.mimeType, f.name);
    const isImage = f.mimeType.startsWith("image/");
    if (isImage) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "message user";
      const img = document.createElement("img");
      img.src = f.dataUrl;
      img.style.maxWidth = "200px";
      img.style.borderRadius = "6px";
      msgDiv.appendChild(img);
      chatLog.appendChild(msgDiv);
    } else {
      appendMessage("user", `📎 ${f.name}`);
    }
  }
  pendingFiles = [];
  renderFilePreview();
  chatLog.scrollTop = chatLog.scrollHeight;
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
  cameraBtn.style.display = "";
  screenBtn.style.display = "";
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

// Fullscreen toggle for chat pane
const centerPanel = document.getElementById("center-panel");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const fsExpand = fullscreenBtn.querySelector(".fs-expand");
const fsContract = fullscreenBtn.querySelector(".fs-contract");

fullscreenBtn.onclick = () => {
  const isFs = centerPanel.classList.toggle("fullscreen");
  fsExpand.style.display = isFs ? "none" : "";
  fsContract.style.display = isFs ? "" : "none";
  setTimeout(() => { chatLog.scrollTop = chatLog.scrollHeight; }, 100);
};

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && centerPanel.classList.contains("fullscreen")) {
    centerPanel.classList.remove("fullscreen");
    fsExpand.style.display = "";
    fsContract.style.display = "none";
  }
});

document.getElementById("resetUsageBtn").onclick = () => {
  accumulated = {
    totalCost: 0,
    totalInput: 0,
    totalOutput: 0,
    totalTurns: 0,
    since: new Date().toISOString(),
  };
  saveAccumulated();
  updateUsagePanel();
};

document.getElementById("clearAgentsBtn").onclick = () => {
  const clearedIds = new Set(Object.keys(agentTasks));
  agentList.innerHTML = "";
  agentTasks = {};
  window._clearedTaskIds = new Set([...(window._clearedTaskIds || []), ...clearedIds]);
};

// Mobile: handle virtual keyboard resizing
if (window.visualViewport) {
  const vv = window.visualViewport;
  const onResize = () => {
    document.documentElement.style.setProperty(
      "--vh", `${vv.height * 0.01}px`
    );
    if (chatLog) {
      setTimeout(() => { chatLog.scrollTop = chatLog.scrollHeight; }, 50);
    }
  };
  vv.addEventListener("resize", onResize);
  onResize();
}

// --- Init ---
updateAuthDescription();
