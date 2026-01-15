# camera_webrtc_page.py
from fastapi.responses import HTMLResponse


def webrtc_page():
    """
    Returns a single-file HTML control panel for the CameraServer.
    Updated to support:
      - /get_camera_info returning range tuples: exposure_range, gain_range, width_range, height_range
      - SetCameraParamsRequest(exposure, gain, pixel_format, exposure_mode, custom_exposure_mode, width, height)
    Covers:
      - /webrtc/offer (WebRTC play-only)
      - /get_camera_info (populate controls)
      - /set_camera_params (quick controls + raw JSON advanced form)
      - /start_recording, /stop_recording
      - /get_saved_count (live stats)
      - /capture?element=...
      - /health
    """
    return HTMLResponse(content=r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Camera Control Panel (WebRTC)</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #151822;
    --muted: #8a92a6;
    --text: #e7ecf3;
    --accent: #6aa7ff;
    --good: #39d98a;
    --bad: #ff6a6a;
    --warn: #ffd36a;
    --border: #23283a;
  }
  html, body { margin:0; padding:0; background:var(--bg); color:var(--text); font: 14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
  h1 { font-size: 20px; margin: 16px 0 6px; }
  h2 { font-size: 16px; margin: 14px 0 6px; color: var(--muted); }
  a { color: var(--accent); }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 16px; }
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; grid-gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .row > * { margin: 4px 0; }
  .btn {
    background: #1e2435; border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 10px; cursor: pointer;
  }
  .btn:hover { border-color: #2c3550; }
  .btn.primary { background: #1f2a44; border-color: #2f4377; }
  .btn.danger { background: #3a1f22; border-color: #7a2c33; }
  .btn.good { background: #1f3a2c; border-color: #2f7751; }
  input[type="text"], input[type="number"], select, textarea {
    background: #111420; border: 1px solid var(--border); color: var(--text);
    padding: 8px 10px; border-radius: 8px; min-width: 120px;
  }
  input[type="range"] { width: 180px; }
  textarea { width: 100%; min-height: 100px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .label { color: var(--muted); font-size: 12px; }
  .kv { display: grid; grid-template-columns: 140px 1fr; gap: 6px 10px; align-items: center; }
  .pill { padding: 3px 8px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); }
  .state { font-weight: 600; }
  .state.good { color: var(--good); }
  .state.bad { color: var(--bad); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  #video { width: 100%; max-height: 60vh; background:#000; border-radius: 10px; }
  #snapshot { width: 100%; max-height: 40vh; background:#000; border-radius: 10px; object-fit: contain; }
  #log { max-height: 220px; overflow:auto; background:#0b0e16; padding:10px; border-radius: 8px; border:1px solid var(--border); }
  .muted { color: var(--muted); }
  .spacer { height: 8px; }
</style>
</head>
<body>
  <div class="wrap">
    <h1>🎥 Camera Control Panel</h1>
    <div class="grid">
      <!-- Left: Video & Snapshots -->
      <div class="card">
        <h2>Live Stream (WebRTC)</h2>
        <video id="video" autoplay playsinline></video>

        <div class="row" style="margin-top:10px;">
          <button id="connectBtn" class="btn primary">Connect</button>
          <button id="disconnectBtn" class="btn">Disconnect</button>
          <span class="pill">Conn: <span id="connState" class="state">-</span></span>
          <span class="pill">ICE: <span id="iceState" class="state">-</span></span>
          <span class="pill">Health: <span id="healthState" class="state">unknown</span></span>
        </div>

        <div class="spacer"></div>
        <h2>Snapshot (/capture)</h2>
        <div class="row">
          <label class="label">Data element</label>
          <select id="elementSelect"></select>
          <button id="snapshotBtn" class="btn">Grab snapshot</button>
        </div>
        <div class="spacer"></div>
        <img id="snapshot" alt="Snapshot" />

        <div class="spacer"></div>
        <h2>Logs</h2>
        <div id="log" class="mono"></div>
      </div>

      <!-- Right: Controls -->
      <div class="card">
        <h2>Camera Info</h2>
        <div class="kv">
          <div class="label">Exposure range</div><div><span id="expMin">-</span> … <span id="expMax">-</span></div>
          <div class="label">Gain range</div><div><span id="gainMin">-</span> … <span id="gainMax">-</span></div>
          <div class="label">Width range</div><div><span id="widthMin">-</span> … <span id="widthMax">-</span></div>
          <div class="label">Height range</div><div><span id="heightMin">-</span> … <span id="heightMax">-</span></div>
          <div class="label">Pixel formats</div><div id="pixFmts" class="muted">-</div>
          <div class="label">Exposure modes</div><div id="expModes" class="muted">-</div>
          <div class="label">Autoexposure</div><div id="aeOpts" class="muted">-</div>
          <div class="label">Data elements</div><div id="dataEls" class="muted">-</div>
        </div>
        <div class="row" style="margin-top:8px;">
          <button id="refreshInfoBtn" class="btn">Refresh info</button>
          <span class="pill">Saved frames: <span id="savedCount">-</span></span>
          <span class="pill">Free GB: <span id="freeGb">-</span></span>
          <span class="pill">Estimated FPS: <span id="estimatedFps">-</span></span>
          <button id="refreshStatsBtn" class="btn">Refresh stats</button>
        </div>

        <!-- Recording section -->
        <div class="spacer"></div>
        <h2>Recording</h2>
        <div class="kv">
          <div class="label">Record name</div>
          <div><input type="text" id="recordName" placeholder="session_001" /></div>
          <div class="label">Resume idx (optional)</div>
          <div><input type="number" id="beginIdx" min="0" step="1" placeholder="0" /></div>
        </div>
        <div class="row" style="margin-top:8px;">
          <button id="startRecBtn" class="btn primary">Start / Resume</button>
          <button id="stopRecBtn" class="btn danger">Stop</button>
        </div>

        <div class="spacer"></div>
        <h2>Quick Camera Controls (/set_camera_params)</h2>
        <div class="kv">
          <div class="label">Exposure</div>
          <div class="row">
            <input type="range" id="exposureRange" min="0" max="0" step="any" />
            <input type="number" id="exposureNum" class="mono" style="width:100px;" step="any" />
          </div>

          <div class="label">Gain</div>
          <div class="row">
            <input type="range" id="gainRange" min="0" max="0" step="any" />
            <input type="number" id="gainNum" class="mono" style="width:100px;" step="any" />
          </div>

          <div class="label">Width</div>
          <div class="row">
            <!-- step=4 so slider positions are multiples of 4 -->
            <input type="range" id="widthRange" min="0" max="0" step="4" />
            <input type="number" id="widthNum" class="mono" style="width:100px;" step="4" />
          </div>

          <div class="label">Height</div>
          <div class="row">
            <!-- step=4 so slider positions are multiples of 4 -->
            <input type="range" id="heightRange" min="0" max="0" step="4" />
            <input type="number" id="heightNum" class="mono" style="width:100px;" step="4" />
          </div>

          <div class="label">Exposure mode</div>
          <div><select id="expModeSel"></select></div>

          <div class="label">Pixel format</div>
          <div><select id="pixFmtSel"></select></div>

          <div class="label">AE option</div>
          <div><select id="aeOptSel"></select></div>
        </div>
        <div class="row" style="margin-top:8px;">
          <button id="applyQuickBtn" class="btn good">Apply</button>
          <span class="muted">Tip: use “Advanced JSON” below if your model expects other fields.</span>
        </div>

        <div class="spacer"></div>
        <h2>Advanced JSON (/set_camera_params)</h2>
        <p class="muted">Send an exact JSON body matching <code>SetCameraParamsRequest</code> (e.g. <code>{"exposure":1200,"gain":4,"width":1920,"height":1080}</code>).</p>
        <textarea id="advancedJson" placeholder='{"exposure": 1200, "gain": 4, "width": 1920, "height": 1080}'></textarea>
        <div class="row">
          <button id="applyAdvancedBtn" class="btn">Send JSON</button>
        </div>
      </div>
    </div>
  </div>

<script>
(function(){
  const $ = (id) => document.getElementById(id);
  const logEl = $("log");
  const videoEl = $("video");
  const healthState = $("healthState");
  const connState = $("connState");
  const iceState = $("iceState");
  const elementSelect = $("elementSelect");

  // Camera info / ranges
  const expMinEl = $("expMin"), expMaxEl = $("expMax");
  const gainMinEl = $("gainMin"), gainMaxEl = $("gainMax");
  const widthMinEl = $("widthMin"), widthMaxEl = $("widthMax");
  const heightMinEl = $("heightMin"), heightMaxEl = $("heightMax");
  const pixFmtsEl = $("pixFmts"), expModesEl = $("expModes"), aeOptsEl = $("aeOpts"), dataElsEl = $("dataEls");
  const exposureRange = $("exposureRange"), exposureNum = $("exposureNum");
  const gainRange = $("gainRange"), gainNum = $("gainNum");
  const widthRange = $("widthRange"), widthNum = $("widthNum");
  const heightRange = $("heightRange"), heightNum = $("heightNum");
  const expModeSel = $("expModeSel"), pixFmtSel = $("pixFmtSel"), aeOptSel = $("aeOptSel");

  // Stats
  const savedCount = $("savedCount"), freeGb = $("freeGb"), estimatedFps = $("estimatedFps"); 

  // Recording controls
  const recordName = $("recordName"), beginIdx = $("beginIdx");

  let pc = null;
  let statsTimer = null;

  // --- Helpers for multiples-of-4 ---
  const STEP4 = 4;
  const ceilTo4  = (n) => Math.ceil(Number(n) / STEP4) * STEP4;
  const floorTo4 = (n) => Math.floor(Number(n) / STEP4) * STEP4;

  // Adjust a [min,max] so both ends are divisible by 4
  function snapRangeTo4(min, max) {
    const mn = ceilTo4(min);
    const mx = floorTo4(max);
    // If reported range doesn't include any /4 value, clamp to the reported max
    return mn <= mx ? { min: mn, max: mx } : { min: Number(max), max: Number(max) };
  }

  // Snap an arbitrary value to the nearest /4 within [min,max]
  function snapTo4Within(v, min, max) {
    let n = Math.round(Number(v) / STEP4) * STEP4;
    n = Math.max(Number(min), Math.min(Number(max), n));
    return n;
  }

  function log(msg, kind="info") {
    const t = new Date().toLocaleTimeString();
    const color = kind === "error" ? "#ff8a8a" : (kind === "warn" ? "#ffd36a" : "#9bb5ff");
    const div = document.createElement("div");
    div.innerHTML = `<span class="muted">[${t}]</span> <span style="color:${color}">${msg}</span>`;
    logEl.prepend(div);
  }

  function setState(el, value) {
    el.textContent = value || "-";
    el.classList.remove("good","bad");
    if (value === "connected" || value === "completed" || value === "ok") el.classList.add("good");
    if (value === "failed" || value === "disconnected" || value === "closed" || value === "error") el.classList.add("bad");
  }

  async function healthCheck() {
    try {
      const r = await fetch("/health");
      const ok = r.ok;
      setState(healthState, ok ? "ok" : "error");
      log(ok ? "Health check OK" : "Health check FAILED", ok ? "info" : "error");
    } catch (e) {
      setState(healthState, "error");
      log("Health check error: " + e, "error");
    }
  }

  // --- WebRTC ---
  async function connect() {
    if (pc) { log("Already connected"); return; }

    pc = new RTCPeerConnection();
    const remoteStream = new MediaStream();
    videoEl.srcObject = remoteStream;

    pc.addEventListener("track", ev => {
      remoteStream.addTrack(ev.track);
      log("Track added: " + ev.track.kind);
    });

    pc.addEventListener("connectionstatechange", () => {
      setState(connState, pc.connectionState);
      log("Connection state: " + pc.connectionState);
      if (pc.connectionState === "failed" || pc.connectionState === "closed" || pc.connectionState === "disconnected") {
        cleanupPC();
      }
    });

    pc.addEventListener("iceconnectionstatechange", () => {
      setState(iceState, pc.iceConnectionState);
      log("ICE state: " + pc.iceConnectionState);
    });

    // receive-only
    pc.addTransceiver("video", { direction: "recvonly" });

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const resp = await fetch("/webrtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
      });
      if (!resp.ok) {
        const msg = await resp.text();
        throw new Error("Offer failed: " + msg);
      }
      const answer = await resp.json();
      await pc.setRemoteDescription(answer);
      log("Connected to WebRTC stream ✅");
    } catch (e) {
      log("WebRTC error: " + e.message, "error");
      cleanupPC();
    }
  }

  function cleanupPC() {
    if (pc) {
      try { pc.close(); } catch {}
      pc = null;
    }
    setState(connState, "-");
    setState(iceState, "-");
    log("PeerConnection cleaned up");
  }

  function disconnect() {
    cleanupPC();
  }

  // --- Camera Info ---
  async function refreshCameraInfo() {
    try {
      const r = await fetch("/get_camera_info");
      if (!r.ok) throw new Error("Failed to fetch camera info");
      const info = await r.json();

      // Ranges from tuples/arrays
      const [expMin, expMax] = Array.isArray(info.exposure_range) && info.exposure_range.length === 2 ? info.exposure_range : [0, 0];
      const [gainMin, gainMax] = Array.isArray(info.gain_range) && info.gain_range.length === 2 ? info.gain_range : [0, 0];
      const [wMin, wMax] = Array.isArray(info.width_range) && info.width_range.length === 2 ? info.width_range : [0, 0];
      const [hMin, hMax] = Array.isArray(info.height_range) && info.height_range.length === 2 ? info.height_range : [0, 0];

      expMinEl.textContent = expMin; expMaxEl.textContent = expMax;
      gainMinEl.textContent = gainMin; gainMaxEl.textContent = gainMax;
      widthMinEl.textContent = wMin; widthMaxEl.textContent = wMax;
      heightMinEl.textContent = hMin; heightMaxEl.textContent = hMax;

      // Set exposure/gain as before (default to min)
      exposureRange.min = expMin; exposureRange.max = expMax; exposureRange.value = expMin;
      gainRange.min = gainMin; gainRange.max = gainMax; gainRange.value = gainMin;
      exposureNum.value = expMin;
      gainNum.value = gainMin;

      // WIDTH / HEIGHT: restrict to multiples of 4 and default to MAX
      const w = snapRangeTo4(wMin, wMax);
      widthRange.min = w.min; widthRange.max = w.max; widthRange.step = STEP4;
      widthRange.value = w.max;
      widthNum.step = STEP4; widthNum.value = w.max;

      const h = snapRangeTo4(hMin, hMax);
      heightRange.min = h.min; heightRange.max = h.max; heightRange.step = STEP4;
      heightRange.value = h.max;
      heightNum.step = STEP4; heightNum.value = h.max;

      // Populate selects from arrays if present
      function fillSelect(sel, arr) {
        sel.innerHTML = "";
        (arr || []).forEach(v => {
          const opt = document.createElement("option");
          opt.value = v; opt.textContent = v;
          sel.appendChild(opt);
        });
      }

      fillSelect(expModeSel, info.available_exposure_modes || []);
      fillSelect(pixFmtSel, info.available_pixel_format || []);
      fillSelect(aeOptSel, info.available_custom_autoexposure_options || []);

      // Human text
      pixFmtsEl.textContent = (info.available_pixel_format || []).join(", ") || "-";
      expModesEl.textContent = (info.available_exposure_modes || []).join(", ") || "-";
      aeOptsEl.textContent = (info.available_custom_autoexposure_options || []).join(", ") || "-";
      dataElsEl.textContent = (info.data_elements || []).join(", ") || "-";

      // Elements -> dropdown for /capture
      elementSelect.innerHTML = "";
      (info.data_elements || ["raw"]).forEach(v => {
        const opt = document.createElement("option"); opt.value = v; opt.textContent = v;
        elementSelect.appendChild(opt);
      });

      log("Camera info refreshed ✅");
    } catch (e) {
      log("Error fetching camera info: " + e.message, "error");
    }
  }

  // Keep range/number in sync (exposure/gain: freeform)
  exposureRange.addEventListener("input", () => exposureNum.value = exposureRange.value);
  exposureNum.addEventListener("input", () => exposureRange.value = exposureNum.value);
  gainRange.addEventListener("input", () => gainNum.value = gainRange.value);
  gainNum.addEventListener("input", () => gainRange.value = gainNum.value);

  // Width/Height: enforce divisible-by-4 behavior
  widthRange.addEventListener("input", () => { widthNum.value = widthRange.value; });
  widthNum.addEventListener("change", () => {
    const n = snapTo4Within(widthNum.value, widthRange.min, widthRange.max);
    widthNum.value = n; widthRange.value = n;
  });

  heightRange.addEventListener("input", () => { heightNum.value = heightRange.value; });
  heightNum.addEventListener("change", () => {
    const n = snapTo4Within(heightNum.value, heightRange.min, heightRange.max);
    heightNum.value = n; heightRange.value = n;
  });

  // --- Apply camera params ---
  async function applyQuickParams() {
    // Conform to SetCameraParamsRequest — include only provided values
    const body = {};
    if (exposureNum.value !== "") body["exposure"] = Number(exposureNum.value);
    if (gainNum.value !== "") body["gain"] = Number(gainNum.value);

    // Ensure width/height are multiples of 4 and within range
    if (widthNum.value !== "")
      body["width"] = snapTo4Within(widthNum.value, widthRange.min, widthRange.max);
    if (heightNum.value !== "")
      body["height"] = snapTo4Within(heightNum.value, heightRange.min, heightRange.max);

    if (expModeSel.value) body["exposure_mode"] = expModeSel.value;
    if (pixFmtSel.value) body["pixel_format"] = pixFmtSel.value;
    if (aeOptSel.value) body["custom_exposure_mode"] = aeOptSel.value;

    await postSetCameraParams(body);
  }

  async function applyAdvancedJson() {
    const raw = $("advancedJson").value.trim();
    if (!raw) { log("Advanced JSON is empty", "warn"); return; }
    try {
      const body = JSON.parse(raw);
      await postSetCameraParams(body);
    } catch (e) {
      log("Invalid JSON: " + e.message, "error");
    }
  }

  async function postSetCameraParams(body) {
    try {
      const r = await fetch("/set_camera_params", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const txt = await r.text();
      if (!r.ok) throw new Error(txt || "Failed to set params");
      log("Camera params applied: " + txt.slice(0, 200) + " ✅");
    } catch (e) {
      log("Set params error: " + e.message, "error");
    }
  }

  // --- Recording ---
  async function startRecording() {
    const name = (recordName.value || "").trim();
    if (!name) { log("Please enter a record name", "warn"); return; }
    const idxVal = beginIdx.value;
    const body = { record_name: name };
    if (idxVal !== "" && Number(idxVal) >= 0) {
      body["begin_frame_idx"] = Number(idxVal);
    }
    try {
      const r = await fetch("/start_recording", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const txt = await r.text();
      if (!r.ok) throw new Error(txt || "Failed to start recording");
      log("Recording started: " + txt.slice(0, 200) + " ▶️");
      pokeStats();
    } catch (e) {
      log("Start recording error: " + e.message, "error");
    }
  }

  async function stopRecording() {
    try {
      const r = await fetch("/stop_recording", { method: "POST" });
      const txt = await r.text();
      if (!r.ok) throw new Error(txt || "Failed to stop recording");
      log("Recording stopped: " + txt.slice(0, 200) + " ⏹");
      pokeStats();
    } catch (e) {
      log("Stop recording error: " + e.message, "error");
    }
  }

  // --- Stats + Snapshot ---
  async function refreshStats() {
    try {
      const r = await fetch("/get_saved_count");
      if (!r.ok) throw new Error("Failed to fetch stats");
      const s = await r.json();
      savedCount.textContent = String(s.n_records ?? "-");
      freeGb.textContent = String(s.free_gb ?? "-");
      estimatedFps.textContent = String(s.estimated_fps ?? "-");
                        
    } catch (e) {
      log("Stats error: " + e.message, "error");
    }
  }

  function startStatsTimer() {
    if (statsTimer) return;
    statsTimer = setInterval(refreshStats, 2000);
  }

  function stopStatsTimer() {
    if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
  }

  function pokeStats() {
    refreshStats();
    startStatsTimer();
  }

  async function grabSnapshot() {
    const el = elementSelect.value || "raw";
    const url = `/capture?element=${encodeURIComponent(el)}&_=${Date.now()}`;
    $("snapshot").src = url;
    log(`Snapshot requested for element '${el}'`);
  }

  // --- Wire up UI ---
  $("connectBtn").addEventListener("click", connect);
  $("disconnectBtn").addEventListener("click", disconnect);
  $("refreshInfoBtn").addEventListener("click", refreshCameraInfo);
  $("refreshStatsBtn").addEventListener("click", refreshStats);
  $("applyQuickBtn").addEventListener("click", applyQuickParams);
  $("applyAdvancedBtn").addEventListener("click", applyAdvancedJson);
  $("snapshotBtn").addEventListener("click", grabSnapshot);
  $("startRecBtn").addEventListener("click", startRecording);
  $("stopRecBtn").addEventListener("click", stopRecording);

  // --- Init ---
  (async function init(){
    await healthCheck();
    await refreshCameraInfo(); // sets width/height to MAX and enforces /4
    await refreshStats();
    startStatsTimer();
    log("UI initialized");
  })();

  // Cleanup on unload
  window.addEventListener("beforeunload", () => {
    disconnect();
    stopStatsTimer();
  });
})();
</script>
</body>
</html>
    """)
