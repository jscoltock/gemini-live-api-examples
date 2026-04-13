/**
 * Agent Configuration UI — slide-over panel for managing agents.yaml
 *
 * Provides: list, create, edit, delete, hot-reload
 * All CRUD goes through /api/agents endpoints.
 */

const AgentConfigUI = (() => {
  // DOM refs
  const overlay = document.getElementById("agentConfigOverlay");
  const panel = document.getElementById("agentConfigPanel");
  const configBtn = document.getElementById("agentConfigBtn");
  const closeBtn = document.getElementById("configPanelClose");
  const addBtn = document.getElementById("addAgentBtn");
  const reloadBtn = document.getElementById("reloadAgentsBtn");
  const listEl = document.getElementById("configAgentList");

  const BACKENDS = ["ollama", "claude-code"];
  let OLLAMA_TOOLS = ["read_file", "write_file", "edit_file", "bash"]; // fallback, replaced by API
  let agents = [];
  let geminiConfig = null;
  let chatModels = [];  // non-Live models (GLM, Qwen)
  let editingName = null; // which agent is currently in form mode (null = none)
  let editingGemini = false; // is the gemini config in edit mode?
  let editingChatModel = null; // which chat model is in edit mode (null = none)

  // --- Panel open/close ---

  function open() {
    overlay.classList.remove("hidden");
    loadAgents();
  }

  function close() {
    overlay.classList.add("hidden");
    editingName = null;
    editingGemini = false;
  }

  configBtn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay && !editingName && !editingGemini) close();
  });

  // --- Data loading ---

  async function loadAgents() {
    try {
      const [agentsRes, geminiRes, toolsRes, chatModelsRes] = await Promise.all([
        fetch("/api/agents"),
        fetch("/api/gemini-config"),
        fetch("/api/ollama-tools"),
        fetch("/api/chat-models"),
      ]);
      if (!agentsRes.ok) throw new Error("Failed to load agents");
      agents = await agentsRes.json();
      if (geminiRes.ok) {
        geminiConfig = await geminiRes.json();
      }
      if (toolsRes.ok) {
        const toolsData = await toolsRes.json();
        if (toolsData.tools && toolsData.tools.length > 0) {
          OLLAMA_TOOLS = toolsData.tools;
        }
      }
      if (chatModelsRes.ok) {
        chatModels = await chatModelsRes.json();
      }
      // Don't wipe editing state on reload — only reset if not editing
      if (!editingName && !editingChatModel) render();
    } catch (e) {
      listEl.innerHTML = `<div class="config-card"><p style="color:#d93025">Error loading agents: ${e.message}</p></div>`;
    }
  }

  // --- Rendering ---

  function render() {
    listEl.innerHTML = "";

    // Gemini Live config section (read-only)
    if (geminiConfig) {
      listEl.appendChild(renderGeminiConfig());
    }

    // Chat model config sections (GLM, Qwen, etc.)
    for (const model of chatModels) {
      if (editingChatModel === model.id) {
        listEl.appendChild(renderChatModelForm(model));
      } else {
        listEl.appendChild(renderChatModelCard(model));
      }
    }

    if (agents.length === 0 && editingName !== "__new__") {
      listEl.innerHTML += `<div class="config-card"><p style="color:var(--text-secondary)">No agents configured. Click "+ New Agent" to add one.</p></div>`;
    }

    for (const agent of agents) {
      if (editingName === agent.name) {
        listEl.appendChild(renderForm(agent));
      } else {
        listEl.appendChild(renderCard(agent));
      }
    }

    // New agent form at the bottom
    if (editingName === "__new__") {
      const form = renderForm({
        name: "",
        description: "",
        backend: "ollama",
        model: "",
        timeout: 120,
        system_prompt: "",
        tools: [],
        options: {},
        fallbacks: [],
      });
      listEl.appendChild(form);
      form.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderCard(agent) {
    const card = document.createElement("div");
    card.className = "config-card";

    const fbCount = (agent.fallbacks || []).length;
    const fbText = fbCount > 0
      ? `<span class="badge">${fbCount} fallback${fbCount > 1 ? "s" : ""}</span>`
      : '<span style="color:var(--text-secondary)">no fallbacks</span>';

    const tools = agent.tools || [];
    const toolsText = tools.length > 0
      ? `<span class="badge badge-info">${tools.length} tool${tools.length > 1 ? "s" : ""}</span>`
      : "";

    const opts = agent.options || {};
    const optEntries = Object.entries(opts);
    const optsText = optEntries.length > 0
      ? optEntries.map(([k, v]) => `${k}=${v}`).join(", ")
      : "";

    card.innerHTML = `
      <div class="config-card-header">
        <span class="config-card-name">${esc(agent.name)}</span>
        <div class="config-card-actions">
          <button data-action="edit" data-name="${esc(agent.name)}">Edit</button>
          <button data-action="delete" data-name="${esc(agent.name)}" class="btn-delete">Delete</button>
        </div>
      </div>
      <div class="config-card-desc">${esc(agent.description || "No description")}</div>
      <div class="config-card-meta">
        <span>${esc(agent.backend)} / ${esc(agent.model)}</span>
        <span>${agent.timeout}s</span>
        ${fbText}
        ${toolsText}
      </div>
      ${tools.length > 0 ? `<div class="config-card-tools">Tools: ${tools.map(t => esc(t)).join(", ")}</div>` : ""}
      ${optsText ? `<div class="config-card-tools">Options: ${esc(optsText)}</div>` : ""}
    `;

    card.querySelector('[data-action="edit"]').addEventListener("click", () => {
      editingName = agent.name;
      render();
    });

    card.querySelector('[data-action="delete"]').addEventListener("click", () => {
      showDeleteConfirm(card, agent.name);
    });

    return card;
  }

  function renderGeminiConfig() {
    if (editingGemini) {
      return renderGeminiForm();
    }

    const card = document.createElement("div");
    card.className = "config-card gemini-config-card";

    // Show a short preview (first 3 lines)
    const lines = (geminiConfig.system_prompt || "").split("\n");
    const preview = lines.slice(0, 3).join("\n");
    const truncated = lines.length > 3;

    const tools = geminiConfig.tools || [];
    const toolsText = tools.length > 0
      ? `<span class="badge badge-info">${tools.length} tool${tools.length > 1 ? "s" : ""}</span>`
      : '<span style="color:var(--text-secondary)">no direct tools</span>';

    card.innerHTML = `
      <div class="config-card-header">
        <span class="config-card-name gemini-label">Gemini Live Session</span>
        <div class="config-card-actions">
          <button class="gemini-edit-btn">Edit</button>
        </div>
      </div>
      <div class="config-card-meta" style="margin-top:0.35rem">
        <span>Model: <strong>${esc(geminiConfig.model)}</strong></span>
        <span>Voice: <strong>${esc(geminiConfig.voice)}</strong></span>
      </div>
      <div class="gemini-prompt-preview">
        <span class="gemini-prompt-preview-label">System Prompt</span>
        <pre class="gemini-prompt-pre-collapsed">${esc(preview || "(none)")}${truncated ? "\n..." : ""}</pre>
      </div>
      <div class="config-card-meta" style="margin-top:0.35rem">
        ${toolsText}
        ${tools.length > 0 ? `<div class="config-card-tools" style="margin-top:0.25rem">Tools: ${tools.map(t => esc(t)).join(", ")}</div>` : ""}
      </div>
    `;

    card.querySelector(".gemini-edit-btn").addEventListener("click", () => {
      editingGemini = true;
      render();
    });

    return card;
  }

  function renderGeminiForm() {
    const card = document.createElement("div");
    card.className = "config-card gemini-config-card gemini-editing";

    const currentTools = geminiConfig.tools || [];

    card.innerHTML = `
      <div class="config-card-header">
        <span class="config-card-name gemini-label">Gemini Live Session</span>
        <span class="badge gemini-badge">editing</span>
      </div>
      <div class="config-card-meta" style="margin-top:0.35rem;margin-bottom:0.5rem">
        <span>Model: <strong>${esc(geminiConfig.model)}</strong></span>
        <span>Voice: <strong>${esc(geminiConfig.voice)}</strong></span>
      </div>
      <div class="gemini-edit-note">Changes take effect on the next session (reconnect).</div>
      <div class="form-group">
        <label>System Prompt</label>
        <textarea id="geminiPromptEdit" class="tall gemini-prompt-textarea">${esc(geminiConfig.system_prompt || "")}</textarea>
      </div>
      <div class="form-section-label">Direct Tools</div>
      <div style="margin-bottom:0.25rem;font-size:0.75rem;color:var(--text-secondary)">Tools the Gemini Live agent can use directly (without routing to a backend agent).</div>
      <div class="tools-checkboxes" id="geminiToolsCheckboxes">
        ${OLLAMA_TOOLS.map(t => {
          const checked = currentTools.includes(t) ? "checked" : "";
          return `<label class="tool-checkbox"><input type="checkbox" name="gtool_${t}" ${checked} /> ${t}</label>`;
        }).join("")}
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-cancel" id="geminiCancelBtn">Cancel</button>
        <button type="button" class="btn" id="geminiSaveBtn">Save</button>
      </div>
    `;

    card.querySelector("#geminiCancelBtn").addEventListener("click", () => {
      editingGemini = false;
      render();
    });

    card.querySelector("#geminiSaveBtn").addEventListener("click", async () => {
      const textarea = card.querySelector("#geminiPromptEdit");
      const prompt = textarea.value.trim();

      // Gather tools from checkboxes
      const tools = [];
      OLLAMA_TOOLS.forEach(t => {
        const cb = card.querySelector(`[name="gtool_${t}"]`);
        if (cb && cb.checked) tools.push(t);
      });

      try {
        const res = await fetch("/api/gemini-config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ system_prompt: prompt, tools }),
        });

        if (!res.ok) {
          const data = await res.json();
          toast(data.error || "Save failed", "error");
          return;
        }

        geminiConfig = await res.json();
        editingGemini = false;
        toast("Gemini config saved (reconnect to apply)", "success");
        render();
      } catch (e) {
        toast("Save failed: " + e.message, "error");
      }
    });

    return card;
  }

  function renderChatModelCard(model) {
    const card = document.createElement("div");
    card.className = "config-card gemini-config-card";

    const lines = (model.system_prompt || "").split("\n");
    const preview = lines.slice(0, 3).join("\n");
    const truncated = lines.length > 3;

    card.innerHTML = `
      <div class="config-card-header">
        <span class="config-card-name gemini-label">${esc(model.label || model.id)}</span>
        <div class="config-card-actions">
          <button class="chat-model-edit-btn">Edit</button>
        </div>
      </div>
      <div class="config-card-meta" style="margin-top:0.35rem">
        <span>Backend: <strong>${esc(model.backend)}</strong></span>
        <span>Model: <strong>${esc(model.model)}</strong></span>
        <span>Tools: <strong>${(model.tools || []).length}</strong></span>
      </div>
      <div class="gemini-prompt-preview">
        <span class="gemini-prompt-preview-label">System Prompt</span>
        <pre class="gemini-prompt-pre-collapsed">${esc(preview || "(none)")}${truncated ? "\n..." : ""}</pre>
      </div>
    `;

    card.querySelector(".chat-model-edit-btn").addEventListener("click", () => {
      editingChatModel = model.id;
      render();
    });

    return card;
  }

  function renderChatModelForm(model) {
    const card = document.createElement("div");
    card.className = "config-card gemini-config-card gemini-editing";

    const currentTools = model.tools || [];

    card.innerHTML = `
      <div class="config-card-header">
        <span class="config-card-name gemini-label">${esc(model.label || model.id)}</span>
        <span class="badge gemini-badge">editing</span>
      </div>
      <div class="config-card-meta" style="margin-top:0.35rem;margin-bottom:0.5rem">
        <span>Backend: <strong>${esc(model.backend)}</strong></span>
      </div>
      <div class="form-group">
        <label>Label</label>
        <input type="text" id="chatModelLabel" value="${esc(model.label || model.id)}" />
      </div>
      <div class="form-group">
        <label>Model</label>
        <input type="text" id="chatModelName" value="${esc(model.model || model.id)}" />
      </div>
      <div class="form-group">
        <label>System Prompt</label>
        <textarea id="chatModelPromptEdit" class="tall gemini-prompt-textarea">${esc(model.system_prompt || "")}</textarea>
      </div>
      <div class="form-section-label">Tools</div>
      <div style="margin-bottom:0.25rem;font-size:0.75rem;color:var(--text-secondary)">Tools this chat model can use.</div>
      <div class="tools-checkboxes" id="chatModelToolsCheckboxes">
        ${OLLAMA_TOOLS.map(t => {
          const checked = currentTools.includes(t) ? "checked" : "";
          return `<label class="tool-checkbox"><input type="checkbox" name="cmtool_${t}" ${checked} /> ${t}</label>`;
        }).join("")}
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-cancel" id="chatModelCancelBtn">Cancel</button>
        <button type="button" class="btn" id="chatModelSaveBtn">Save</button>
      </div>
    `;

    card.querySelector("#chatModelCancelBtn").addEventListener("click", () => {
      editingChatModel = null;
      render();
    });

    card.querySelector("#chatModelSaveBtn").addEventListener("click", async () => {
      const label = card.querySelector("#chatModelLabel").value.trim();
      const modelName = card.querySelector("#chatModelName").value.trim();
      const prompt = card.querySelector("#chatModelPromptEdit").value.trim();

      // Gather tools from checkboxes
      const tools = [];
      OLLAMA_TOOLS.forEach(t => {
        const cb = card.querySelector(`[name="cmtool_${t}"]`);
        if (cb && cb.checked) tools.push(t);
      });

      try {
        const res = await fetch(`/api/chat-models/${encodeURIComponent(model.id)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label, model: modelName, system_prompt: prompt, tools }),
        });

        if (!res.ok) {
          const data = await res.json();
          toast(data.error || "Save failed", "error");
          return;
        }

        editingChatModel = null;
        toast(`${model.label || model.id} config saved`, "success");
        await loadAgents();
      } catch (e) {
        toast("Save failed: " + e.message, "error");
      }
    });

    return card;
  }

  function showDeleteConfirm(card, name) {
    // Remove any existing confirm
    const existing = card.querySelector(".delete-confirm");
    if (existing) { existing.remove(); return; }

    const confirm = document.createElement("div");
    confirm.className = "delete-confirm";
    confirm.innerHTML = `
      <span>Delete "${esc(name)}"? This cannot be undone.</span>
      <div style="display:flex;gap:0.35rem">
        <button class="btn-cancel-delete">Cancel</button>
        <button class="btn-confirm-delete">Delete</button>
      </div>
    `;

    confirm.querySelector(".btn-cancel-delete").addEventListener("click", () => confirm.remove());
    confirm.querySelector(".btn-confirm-delete").addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/agents/${encodeURIComponent(name)}`, { method: "DELETE" });
        if (!res.ok) {
          const data = await res.json();
          toast(data.error || "Delete failed", "error");
          return;
        }
        toast(`Agent "${name}" deleted`, "success");
        await loadAgents();
      } catch (e) {
        toast("Delete failed: " + e.message, "error");
      }
    });

    card.appendChild(confirm);
  }

  function renderForm(agent) {
    const isNew = !agent.name;
    const form = document.createElement("div");
    form.className = "config-form";

    const fallbackRows = (agent.fallbacks || []).map((fb, i) => fallbackRowHTML(fb, i)).join("");

    form.innerHTML = `
      <h3>${isNew ? "New Agent" : `Edit: ${esc(agent.name)}`}</h3>

      <div class="form-group">
        <label>Name</label>
        <input type="text" name="name" value="${esc(agent.name)}"
               placeholder="e.g. my-agent"
               ${isNew ? "" : "readonly"} />
      </div>

      <div class="form-group">
        <label>Description</label>
        <textarea name="description" placeholder="What does this agent do? When should it be used?">${esc(agent.description)}</textarea>
      </div>

      <div class="form-section-label">Primary Config</div>

      <div class="form-row">
        <div class="form-group">
          <label>Backend</label>
          <select name="backend">
            ${BACKENDS.map(b => `<option value="${b}" ${b === agent.backend ? "selected" : ""}>${b}</option>`).join("")}
          </select>
        </div>
        <div class="form-group">
          <label>Model</label>
          <input type="text" name="model" value="${esc(agent.model)}" placeholder="e.g. glm-5.1" />
        </div>
        <div class="form-group" style="flex:0 0 90px">
          <label>Timeout</label>
          <input type="number" name="timeout" value="${agent.timeout}" min="1" />
        </div>
      </div>

      <div class="form-group">
        <label>System Prompt</label>
        <textarea name="system_prompt" class="tall" placeholder="Instructions for the agent...">${esc(agent.system_prompt || "")}</textarea>
      </div>

      <div class="form-section-label">Ollama Tools</div>
      <div class="tools-checkboxes" id="toolsCheckboxes">
        ${OLLAMA_TOOLS.map(t => {
          const checked = (agent.tools || []).includes(t) ? "checked" : "";
          return `<label class="tool-checkbox"><input type="checkbox" name="tool_${t}" ${checked} /> ${t}</label>`;
        }).join("")}
      </div>

      <div class="form-section-label">Generation Options</div>
      <div class="form-row" id="optionsFields">
        <div class="form-group">
          <label>Temperature</label>
          <input type="number" step="0.1" name="opt_temperature" value="${(agent.options || {}).temperature != null ? (agent.options || {}).temperature : ""}" placeholder="0" />
        </div>
        <div class="form-group">
          <label>Top P</label>
          <input type="number" step="0.05" name="opt_top_p" value="${(agent.options || {}).top_p != null ? (agent.options || {}).top_p : ""}" placeholder="0.9" />
        </div>
        <div class="form-group">
          <label>Repeat Penalty</label>
          <input type="number" step="0.05" name="opt_repeat_penalty" value="${(agent.options || {}).repeat_penalty != null ? (agent.options || {}).repeat_penalty : ""}" placeholder="1.1" />
        </div>
      </div>

      <div class="form-section-label">Fallbacks</div>
      <div class="fallback-list" id="fallbackList">
        ${fallbackRows}
      </div>
      <button type="button" class="add-fallback-btn" id="addFallbackBtn">+ Add Fallback</button>

      <div class="form-actions">
        <button type="button" class="btn btn-cancel" id="formCancel">Cancel</button>
        <button type="button" class="btn" id="formSave">Save Agent</button>
      </div>
    `;

    // Wire up events
    form.querySelector("#formCancel").addEventListener("click", () => {
      editingName = null;
      render();
    });

    form.querySelector("#addFallbackBtn").addEventListener("click", () => {
      const list = form.querySelector("#fallbackList");
      const idx = list.children.length;
      list.insertAdjacentHTML("beforeend", fallbackRowHTML({ backend: "ollama", model: "", timeout: 120 }, idx));
      wireFallbackRemove(list.lastElementChild);
    });

    // Wire remove buttons on existing fallbacks
    form.querySelectorAll(".fb-remove").forEach(btn => wireFallbackRemove(btn.closest(".fallback-row")));

    form.querySelector("#formSave").addEventListener("click", async () => {
      await saveForm(form, isNew ? "" : agent.name);
    });

    return form;
  }

  function fallbackRowHTML(fb, index) {
    return `
      <div class="fallback-row">
        <select name="fb_backend">
          ${BACKENDS.map(b => `<option value="${b}" ${b === fb.backend ? "selected" : ""}>${b}</option>`).join("")}
        </select>
        <input type="text" class="fb-model" name="fb_model" value="${esc(fb.model || "")}" placeholder="model" />
        <input type="number" class="fb-timeout" name="fb_timeout" value="${fb.timeout || 120}" min="1" />
        <button type="button" class="fb-remove" title="Remove">&times;</button>
      </div>
    `;
  }

  function wireFallbackRemove(row) {
    if (!row) return;
    const btn = row.querySelector(".fb-remove");
    if (btn) {
      btn.addEventListener("click", () => row.remove());
    }
  }

  async function saveForm(formEl, existingName) {
    const name = formEl.querySelector('[name="name"]').value.trim();
    const description = formEl.querySelector('[name="description"]').value.trim();
    const backend = formEl.querySelector('[name="backend"]').value;
    const model = formEl.querySelector('[name="model"]').value.trim();
    const timeout = parseInt(formEl.querySelector('[name="timeout"]').value, 10);
    const system_prompt = formEl.querySelector('[name="system_prompt"]').value.trim();

    // Gather tools from checkboxes
    const tools = [];
    OLLAMA_TOOLS.forEach(t => {
      const cb = formEl.querySelector(`[name="tool_${t}"]`);
      if (cb && cb.checked) tools.push(t);
    });

    // Gather options (only include non-empty values)
    const options = {};
    const tempVal = formEl.querySelector('[name="opt_temperature"]').value;
    const topPVal = formEl.querySelector('[name="opt_top_p"]').value;
    const rpVal = formEl.querySelector('[name="opt_repeat_penalty"]').value;
    if (tempVal !== "") options.temperature = parseFloat(tempVal);
    if (topPVal !== "") options.top_p = parseFloat(topPVal);
    if (rpVal !== "") options.repeat_penalty = parseFloat(rpVal);

    // Gather fallbacks
    const fallbacks = [];
    formEl.querySelectorAll(".fallback-row").forEach(row => {
      fallbacks.push({
        backend: row.querySelector('[name="fb_backend"]').value,
        model: row.querySelector('[name="fb_model"]').value.trim(),
        timeout: parseInt(row.querySelector('[name="fb_timeout"]').value, 10) || 120,
      });
    });
    // Filter out empty fallbacks
    const filteredFallbacks = fallbacks.filter(fb => fb.model.trim() !== "");

    const payload = {
      name,
      description,
      backend,
      model,
      timeout,
      system_prompt,
      tools,
      options: Object.keys(options).length > 0 ? options : undefined,
      fallbacks: filteredFallbacks,
    };

    try {
      const isNew = !existingName;
      const url = isNew
        ? "/api/agents"
        : `/api/agents/${encodeURIComponent(existingName)}`;
      const method = isNew ? "POST" : "PUT";

      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        // Show errors
        const errors = data.errors || [data.error || "Save failed"];
        showFormErrors(formEl, errors);
        return;
      }

      toast(`Agent "${data.name}" ${isNew ? "created" : "updated"}`, "success");
      editingName = null;
      await loadAgents();

    } catch (e) {
      toast("Save failed: " + e.message, "error");
    }
  }

  function showFormErrors(formEl, errors) {
    // Remove any existing error box
    const existing = formEl.querySelector(".form-errors");
    if (existing) existing.remove();

    const errorBox = document.createElement("div");
    errorBox.className = "form-errors";
    errorBox.innerHTML = `<ul>${errors.map(e => `<li>${esc(e)}</li>`).join("")}</ul>`;
    formEl.insertBefore(errorBox, formEl.firstChild);
    errorBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // --- Toolbar buttons ---

  addBtn.addEventListener("click", () => {
    editingName = "__new__";
    render();
  });

  reloadBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/agents/reload", { method: "POST" });
      const data = await res.json();
      toast(`Reloaded ${data.count} agent(s): ${data.agents.join(", ")}`, "success");
      await loadAgents();
    } catch (e) {
      toast("Reload failed: " + e.message, "error");
    }
  });

  // --- Toast ---

  function toast(message, type = "success") {
    // Remove existing toasts
    document.querySelectorAll(".config-toast").forEach(t => t.remove());

    const el = document.createElement("div");
    el.className = `config-toast ${type}`;
    el.textContent = message;
    document.body.appendChild(el);

    setTimeout(() => {
      el.style.transition = "opacity 0.3s";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 300);
    }, 3000);
  }

  // --- Utility ---

  function esc(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  // Public API (in case main.js or others need it)
  return { open, close, loadAgents };
})();
