#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import threading
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from flask import Flask, jsonify, request, send_file, Response, render_template_string

APP_TITLE = "TwoChat (via EVA) — Marrtino ↔ Peppino"

# ✅ Config spostato dentro "config/"
CONFIG_DIR = "config"
CONFIG_PATH = os.path.join(CONFIG_DIR, "twochat.json")


# -------------------- JSON helpers (atomic) --------------------
def _write_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    tmp = None
    d = json.dumps(obj, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(os.path.abspath(path)) or ".") as tf:
        tf.write(d)
        tmp = tf.name
    os.replace(tmp, path)


# -------------------- DEFAULT CONFIG --------------------
# Qui la parte "prompt separati" la metti su EVA creando due profili:
#   - twochat_marrtino
#   - twochat_peppino
DEFAULT_CONFIG: Dict[str, Any] = {
    "eva_base": "http://127.0.0.1:5000",

    # Profili EVA dedicati alla chat "per persone normali"
    "marrtino": {"profile": "twochat_marrtino", "model": ""},
    "peppino": {"profile": "twochat_peppino", "model": ""},

    "topics": [
        "chiacchiere casuali",
        "cibo e ricette semplici",
        "piccole cose strane della giornata",
        "musica mentre si lavora",
        "tecnologia che fa i capricci"
    ],
    "topic_mode": "cycle",          # "cycle" o "random"
    "max_history_lines": 12,
    "sleep_s": 0.35,
    "timeout_s": 30,
    "auto_turns": 40,

    "log": {"enabled": True, "path": "twochat.log"},
    "ui": {"host": "127.0.0.1", "port": 5001},
}


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> Dict[str, Any]:
    # ✅ crea cartella config/ se manca
    os.makedirs(CONFIG_DIR, exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        _write_json_atomic(CONFIG_PATH, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return deep_merge(DEFAULT_CONFIG, cfg if isinstance(cfg, dict) else {})
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    _write_json_atomic(CONFIG_PATH, cfg)


def validate_config(cfg: Dict[str, Any]) -> Optional[str]:
    if not isinstance(cfg.get("eva_base"), str) or not cfg["eva_base"].startswith(("http://", "https://")):
        return "eva_base deve essere una URL valida (es: http://127.0.0.1:5000)"
    for who in ("marrtino", "peppino"):
        if not isinstance(cfg.get(who), dict):
            return f"Manca sezione {who}"
        if not isinstance(cfg[who].get("profile"), str) or not cfg[who]["profile"].strip():
            return f"{who}.profile non valido"
        if not isinstance(cfg[who].get("model", ""), str):
            return f"{who}.model deve essere stringa (può essere vuota)"
    topics = cfg.get("topics")
    if not isinstance(topics, list) or not topics or not all(isinstance(x, str) and x.strip() for x in topics):
        return "topics deve essere una lista non vuota di stringhe"
    if cfg.get("topic_mode") not in ("cycle", "random"):
        return "topic_mode deve essere 'cycle' o 'random'"
    ui = cfg.get("ui", {})
    if not isinstance(ui, dict):
        return "ui deve essere un oggetto"
    if not isinstance(ui.get("host", ""), str) or not ui.get("host", "").strip():
        return "ui.host non valido"
    try:
        p = int(ui.get("port", 5001))
        if p < 1 or p > 65535:
            return "ui.port fuori range"
    except Exception:
        return "ui.port non valido"
    return None


# -------------------- EVA call --------------------
def eva_ask(eva_base: str, query: str, profile: str, model: str, timeout_s: int) -> str:
    """
    Chiama EVA /json con:
      - no_handlers: True  (disabilita handlers locali tipo present_it.py)
      - no_commands: True  (disabilita trigger comandi)
    """
    url = eva_base.rstrip("/") + "/json"
    payload = {
        "query": query,
        "profile": profile,
        "no_handlers": True,
        "no_commands": True
    }
    if model and model.strip():
        payload["model"] = model.strip()

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"EVA HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Errore rete verso EVA ({url}): {e}") from e

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Risposta non-JSON da EVA: {raw[:300]}") from e

    ans = obj.get("response", "")
    if not isinstance(ans, str):
        ans = str(ans)
    return ans.strip()


# -------------------- Phases (UI: clessidra/spinner) --------------------
PHASE_IDLE = "IDLE"
PHASE_CALL_M = "CALLING_MARRTINO"
PHASE_PAUSE_AFTER_M = "PAUSE_AFTER_MARRTINO"
PHASE_CALL_P = "CALLING_PEPPINO"
PHASE_PAUSE_AFTER_P = "PAUSE_AFTER_PEPPINO"
PHASE_STOPPING = "STOPPING"


class TwoChatState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = load_config()

        self.running = False
        self.stop_evt = threading.Event()
        self.worker: Optional[threading.Thread] = None

        self.topic_idx = 0
        self.current_topic = ""
        self.turn_count = 0
        self.last_error = ""

        self.phase: str = PHASE_IDLE
        self.pause_until_epoch: float = 0.0
        self.last_phase_change: str = self._now()

        self.transcript: List[Dict[str, str]] = []
        self._reset_locked(new_topic=True)

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _set_phase_locked(self, phase: str, pause_for_s: float = 0.0) -> None:
        self.phase = phase
        self.last_phase_change = self._now()
        if pause_for_s and pause_for_s > 0:
            self.pause_until_epoch = time.time() + float(pause_for_s)
        else:
            self.pause_until_epoch = 0.0

    def _pick_topic_locked(self) -> str:
        topics = self.cfg.get("topics", ["chiacchiere casuali"])
        mode = self.cfg.get("topic_mode", "cycle")
        if mode == "random":
            i = int(time.time()) % len(topics)
            self.topic_idx = i
        else:
            i = self.topic_idx % len(topics)
            self.topic_idx = (self.topic_idx + 1) % len(topics)
        return topics[i].strip()

    def _append_line_locked(self, who: str, text: str) -> None:
        ts = self._now()
        self.transcript.append({"ts": ts, "who": who, "text": text})

        log_cfg = (self.cfg.get("log") or {})
        if log_cfg.get("enabled"):
            path = log_cfg.get("path", "twochat.log")
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {who}: {text}\n")
            except Exception as e:
                self.last_error = f"Errore scrittura log: {e}"

    def _context_tail_locked(self) -> List[Dict[str, str]]:
        max_lines = int(self.cfg.get("max_history_lines", 12))
        usable = [m for m in self.transcript if m.get("who") in ("MARRTINO", "PEPPINO", "SYSTEM")]
        return usable[-max_lines:] if len(usable) > max_lines else usable

    def _last_speaker_locked(self) -> str:
        for m in reversed(self.transcript):
            if m.get("who") in ("MARRTINO", "PEPPINO"):
                return m["who"]
        return ""

    def _last_text_by_locked(self, who: str) -> str:
        for m in reversed(self.transcript):
            if m.get("who") == who:
                return (m.get("text") or "").strip()
        return ""

    def _build_prompt_locked(self, target: str, is_first: bool) -> str:
        tail = self._context_tail_locked()
        last_speaker = self._last_speaker_locked()

        lines = []
        lines.append(f"TEMA: {self.current_topic}")
        lines.append("")
        lines.append("REGOLE:")
        lines.append("- Risposta breve: massimo 2–3 frasi.")
        lines.append("- Non ripetere la trascrizione.")
        lines.append("- Mantieni la conversazione leggera e concreta.")
        lines.append("- Fai UNA domanda semplice alla fine.")
        lines.append("")
        lines.append("CONTESTO (ultimo scambio):")

        if tail:
            for m in tail:
                who = m.get("who", "SYSTEM")
                txt = (m.get("text") or "").strip()
                if not txt:
                    continue
                if who == "SYSTEM":
                    continue
                lines.append(f"{who}: {txt}")
        else:
            lines.append("(vuoto)")

        lines.append("")
        if is_first:
            lines.append(f"INIZIA tu come {target}. Una frase + una domanda semplice.")
        else:
            if last_speaker and last_speaker != target:
                lines.append(f"ORA parla tu come {target}. Rispondi a {last_speaker} e fai una domanda semplice.")
            else:
                lines.append(f"ORA parla tu come {target}. Continua e fai una domanda semplice.")

        last_me = self._last_text_by_locked(target)
        last_other = self._last_text_by_locked("PEPPINO" if target == "MARRTINO" else "MARRTINO")
        if last_me and last_other and last_me == last_other:
            lines.append("IMPORTANTE: Non copiare parola per parola. Cambia frase e vai avanti.")

        return "\n".join(lines).strip()

    def _reset_locked(self, new_topic: bool = True) -> None:
        self.turn_count = 0
        self.last_error = ""
        self.transcript = []
        if new_topic or not self.current_topic:
            self.current_topic = self._pick_topic_locked()
        self._append_line_locked("SYSTEM", f"Nuova conversazione. Topic: {self.current_topic}")
        self._set_phase_locked(PHASE_IDLE, 0)

    def reset(self) -> None:
        with self.lock:
            if self.running:
                self.stop_evt.set()
                self.running = False
                self._set_phase_locked(PHASE_STOPPING, 0)
        time.sleep(0.05)
        with self.lock:
            self._reset_locked(new_topic=True)

    def stop(self) -> None:
        with self.lock:
            self.stop_evt.set()
            self.running = False
            self._set_phase_locked(PHASE_STOPPING, 0)

    def _pause_without_blocking_ui(self, phase: str, seconds: float) -> None:
        with self.lock:
            if self.stop_evt.is_set():
                return
            self._set_phase_locked(phase, pause_for_s=seconds)

        end = time.time() + max(0.0, float(seconds))
        while time.time() < end:
            if self.stop_evt.is_set():
                return
            time.sleep(0.05)

        with self.lock:
            self.pause_until_epoch = 0.0

    def step_one_turn(self) -> None:
        with self.lock:
            if self.stop_evt.is_set():
                return

            cfg = self.cfg
            eva_base = cfg["eva_base"]
            timeout_s = int(cfg.get("timeout_s", 30))
            sleep_s = float(cfg.get("sleep_s", 0.35))

            m_prof = cfg["marrtino"]["profile"]
            m_model = cfg["marrtino"].get("model", "")
            p_prof = cfg["peppino"]["profile"]
            p_model = cfg["peppino"].get("model", "")

            is_first_m = (self.turn_count == 0 and not any(m["who"] == "MARRTINO" for m in self.transcript))
            m_prompt = self._build_prompt_locked("MARRTINO", is_first=is_first_m)

            self.last_error = ""
            self._set_phase_locked(PHASE_CALL_M, 0)

        try:
            m_text = eva_ask(eva_base, m_prompt, m_prof, m_model, timeout_s) or "(silenzio)"
        except Exception as e:
            with self.lock:
                self.last_error = str(e)
                self._append_line_locked("SYSTEM", f"ERRORE (MARRTINO): {e}")
                self._set_phase_locked(PHASE_IDLE, 0)
            return

        with self.lock:
            self._append_line_locked("MARRTINO", m_text)

        self._pause_without_blocking_ui(PHASE_PAUSE_AFTER_M, sleep_s)
        if self.stop_evt.is_set():
            with self.lock:
                self._set_phase_locked(PHASE_IDLE, 0)
            return

        with self.lock:
            p_prompt = self._build_prompt_locked("PEPPINO", is_first=False)
            self._set_phase_locked(PHASE_CALL_P, 0)

        try:
            p_text = eva_ask(eva_base, p_prompt, p_prof, p_model, timeout_s) or "(silenzio)"
        except Exception as e:
            with self.lock:
                self.last_error = str(e)
                self._append_line_locked("SYSTEM", f"ERRORE (PEPPINO): {e}")
                self._set_phase_locked(PHASE_IDLE, 0)
            return

        with self.lock:
            self._append_line_locked("PEPPINO", p_text)
            self.turn_count += 1

        self._pause_without_blocking_ui(PHASE_PAUSE_AFTER_P, sleep_s)

        with self.lock:
            self._set_phase_locked(PHASE_IDLE, 0)

    def start(self, turns: Optional[int] = None) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.stop_evt.clear()
            self.last_error = ""
            self._set_phase_locked(PHASE_IDLE, 0)
            tmax = int(turns if turns is not None else self.cfg.get("auto_turns", 40))

        def worker():
            try:
                for _ in range(tmax):
                    if self.stop_evt.is_set():
                        break
                    self.step_one_turn()
                    if self.stop_evt.is_set():
                        break
            finally:
                with self.lock:
                    self.running = False
                    self._set_phase_locked(PHASE_IDLE, 0)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def update_config(self, new_cfg: Dict[str, Any]) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("Ferma la chat (Stop) prima di salvare la config.")
            self.cfg = deep_merge(DEFAULT_CONFIG, new_cfg)
            save_config(self.cfg)

    def reload_config(self) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("Ferma la chat (Stop) prima di ricaricare la config.")
            self.cfg = load_config()

    def public_state(self) -> Dict[str, Any]:
        with self.lock:
            log_cfg = (self.cfg.get("log") or {})
            pause_rem = 0.0
            if self.pause_until_epoch and self.pause_until_epoch > time.time():
                pause_rem = self.pause_until_epoch - time.time()
            return {
                "running": self.running,
                "turn_count": self.turn_count,
                "current_topic": self.current_topic,
                "last_error": self.last_error,
                "phase": self.phase,
                "pause_remaining_s": round(pause_rem, 2),
                "last_phase_change": self.last_phase_change,
                "transcript": self.transcript[-600:],
                "config": self.cfg,
                "log_enabled": bool(log_cfg.get("enabled")),
            }


# -------------------- Flask UI --------------------
app = Flask(__name__)
state = TwoChatState()

PAGE_HTML = """
<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:16px;background:#0b0f14;color:#e6edf3}
    .wrap{display:grid;grid-template-columns:1.1fr 0.9fr;gap:14px}
    .card{background:#111824;border:1px solid #1f2a3a;border-radius:12px;padding:14px}
    h1{font-size:18px;margin:0 0 10px}
    h2{font-size:14px;margin:0 0 10px;color:#b6c2cf}
    button{background:#1f6feb;border:0;color:#fff;padding:10px 12px;border-radius:10px;cursor:pointer}
    button.secondary{background:#30363d}
    button.danger{background:#da3633}
    button:disabled{opacity:.5;cursor:not-allowed}
    .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
    .small{font-size:12px;color:#9fb0c0}
    textarea{width:100%;min-height:320px;background:#0b1220;color:#e6edf3;border:1px solid #1f2a3a;border-radius:10px;padding:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    #chat{height:60vh;overflow:auto;padding:10px;background:#0b1220;border-radius:10px;border:1px solid #1f2a3a}
    .msg{margin:0 0 10px}
    .meta{font-size:12px;color:#9fb0c0;margin-bottom:2px}
    .who{font-weight:700}
    .text{white-space:pre-wrap;line-height:1.35}
    .err{color:#ff7b72}
    a{color:#7ee787;text-decoration:none}
    a:hover{text-decoration:underline}
    input[type="number"]{width:92px;background:#0b1220;color:#e6edf3;border:1px solid #1f2a3a;border-radius:10px;padding:8px}

    .statusbar{display:flex;align-items:center;gap:10px;margin-top:6px}
    .pill{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #1f2a3a;background:#0b1220}
    .dot{width:8px;height:8px;border-radius:50%}
    .dot.idle{background:#7ee787}
    .dot.busy{background:#ffa657}
    .dot.err{background:#ff7b72}
    .spin{
      width:14px;height:14px;border-radius:50%;
      border:2px solid rgba(230,237,243,.25);
      border-top-color: rgba(230,237,243,.95);
      animation: sp 0.9s linear infinite;
    }
    @keyframes sp { to { transform: rotate(360deg); } }
    .progress{height:6px;background:#0b1220;border:1px solid #1f2a3a;border-radius:999px;overflow:hidden;width:260px}
    .bar{height:100%;background:#1f6feb;width:0%}
  </style>
</head>
<body>
  <div class="row" style="justify-content:space-between;">
    <div>
      <h1>{{ title }}</h1>
      <div class="small">Topic: <span id="topic"></span> • Turni: <span id="turns"></span> • Stato: <span id="running"></span></div>
      <div class="small">EVA: <span id="eva_base"></span> • Profili: Marrtino=<span id="pm"></span> Peppino=<span id="pp"></span></div>

      <div class="statusbar">
        <div class="pill">
          <span class="dot idle" id="dot"></span>
          <span id="phaseText">In attesa.</span>
          <span id="spinnerSlot"></span>
          <span class="small" id="phaseSince"></span>
        </div>
        <div class="pill" id="pausePill" style="display:none;">
          <span>⌛</span>
          <span id="pauseText">Pausa</span>
          <div class="progress"><div class="bar" id="pauseBar"></div></div>
        </div>
      </div>

      <div class="small err" id="last_error"></div>
    </div>

    <div class="row">
      <button id="btnStep" onclick="stepOnce()">Step (1 turno)</button>
      <input id="turnCount" type="number" min="1" value="40" title="Turni auto">
      <button id="btnStart" onclick="startAuto()">Start</button>
      <button class="danger" id="btnStop" onclick="stopAuto()">Stop</button>
      <button class="secondary" onclick="resetChat()">Reset</button>
      <a id="downloadLog" href="/download_log" target="_blank">Scarica log</a>
    </div>
  </div>

  <div class="wrap" style="margin-top:14px;">
    <div class="card">
      <h2>Dialogo</h2>
      <div id="chat"></div>
    </div>

    <div class="card">
      <h2>Config ({{ config_path }})</h2>
      <div class="small">Modifica il JSON e premi “Salva config”. (Serve Stop se la chat è in RUNNING.)</div>
      <textarea id="cfg"></textarea>
      <div class="row" style="margin-top:10px;">
        <button onclick="saveConfig()">Salva config</button>
        <button class="secondary" onclick="reloadConfig()">Ricarica config</button>
        <button class="secondary" onclick="viewLog()">Vedi log (ultime righe)</button>
      </div>
      <pre class="small" id="logview" style="white-space:pre-wrap;margin-top:10px;"></pre>
    </div>
  </div>

<script>
let lastLen = 0;
let lastPauseTotal = 0;

function escapeHtml(s){
  return (s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function phaseLabel(phase){
  switch(phase){
    case "CALLING_MARRTINO": return "Sto chiamando EVA per Marrtino…";
    case "PAUSE_AFTER_MARRTINO": return "Pausa dopo Marrtino…";
    case "CALLING_PEPPINO": return "Sto chiamando EVA per Peppino…";
    case "PAUSE_AFTER_PEPPINO": return "Pausa dopo Peppino…";
    case "STOPPING": return "Stop in corso…";
    default: return "In attesa.";
  }
}

function isPause(phase){
  return phase === "PAUSE_AFTER_MARRTINO" || phase === "PAUSE_AFTER_PEPPINO";
}
function isCalling(phase){
  return phase === "CALLING_MARRTINO" || phase === "CALLING_PEPPINO";
}

function render(st){
  document.getElementById("topic").textContent = st.current_topic || "";
  document.getElementById("turns").textContent = st.turn_count ?? 0;
  document.getElementById("running").textContent = st.running ? "RUNNING" : "IDLE";
  document.getElementById("last_error").textContent = st.last_error || "";

  document.getElementById("eva_base").textContent = st.config?.eva_base || "";
  document.getElementById("pm").textContent = st.config?.marrtino?.profile || "";
  document.getElementById("pp").textContent = st.config?.peppino?.profile || "";

  document.getElementById("btnStart").disabled = st.running;
  document.getElementById("btnStop").disabled = !st.running;
  document.getElementById("btnStep").disabled = st.running;

  const phase = st.phase || "IDLE";
  document.getElementById("phaseText").textContent = phaseLabel(phase);
  document.getElementById("phaseSince").textContent = st.last_phase_change ? ("• " + st.last_phase_change) : "";

  const dot = document.getElementById("dot");
  dot.className = "dot " + (st.last_error ? "err" : (isCalling(phase) || isPause(phase) ? "busy" : "idle"));

  const spinnerSlot = document.getElementById("spinnerSlot");
  spinnerSlot.innerHTML = isCalling(phase) ? '<span class="spin" title="in elaborazione"></span>' : '';

  const pausePill = document.getElementById("pausePill");
  const pauseRem = Number(st.pause_remaining_s || 0);
  if (isPause(phase) && pauseRem > 0){
    pausePill.style.display = "inline-flex";
    document.getElementById("pauseText").textContent = `Pausa: ${pauseRem.toFixed(1)}s`;

    if (!lastPauseTotal || pauseRem > lastPauseTotal) lastPauseTotal = pauseRem;
    const pct = Math.max(0, Math.min(100, 100 * (1 - (pauseRem / lastPauseTotal))));
    document.getElementById("pauseBar").style.width = pct.toFixed(1) + "%";
  } else {
    pausePill.style.display = "none";
    lastPauseTotal = 0;
    document.getElementById("pauseBar").style.width = "0%";
  }

  const chat = document.getElementById("chat");
  if ((st.transcript?.length || 0) !== lastLen){
    lastLen = st.transcript.length || 0;
    chat.innerHTML = "";
    for (const m of st.transcript){
      const div = document.createElement("div");
      div.className = "msg";
      div.innerHTML = `
        <div class="meta">[${escapeHtml(m.ts)}] <span class="who">${escapeHtml(m.who)}</span></div>
        <div class="text">${escapeHtml(m.text)}</div>
      `;
      chat.appendChild(div);
    }
    chat.scrollTop = chat.scrollHeight;
  }

  if (!document.getElementById("cfg").dataset.touched){
    document.getElementById("cfg").value = JSON.stringify(st.config, null, 2);
  }

  document.getElementById("downloadLog").style.display = st.log_enabled ? "inline" : "none";
}

async function poll(){
  try{
    const r = await fetch("/api/state");
    const s = await r.json();
    render(s);
  }catch(e){}
  setTimeout(poll, 350);
}

document.getElementById("cfg").addEventListener("input", () => {
  document.getElementById("cfg").dataset.touched = "1";
});

async function stepOnce(){ await fetch("/api/step", {method:"POST"}); }
async function startAuto(){
  const n = parseInt(document.getElementById("turnCount").value || "40", 10);
  await fetch("/api/start", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({turns:n})});
}
async function stopAuto(){ await fetch("/api/stop", {method:"POST"}); }
async function resetChat(){
  await fetch("/api/reset", {method:"POST"});
  document.getElementById("cfg").dataset.touched = "";
}

async function saveConfig(){
  try{
    const obj = JSON.parse(document.getElementById("cfg").value);
    const r = await fetch("/api/save_config", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(obj)});
    const out = await r.json();
    if (!out.ok) alert("Errore: " + out.error);
    else document.getElementById("cfg").dataset.touched = "";
  }catch(e){
    alert("JSON non valido: " + e);
  }
}

async function reloadConfig(){
  document.getElementById("cfg").dataset.touched = "";
  const r = await fetch("/api/reload_config", {method:"POST"});
  const s = await r.json();
  if (!s.ok) alert("Errore: " + s.error);
}

async function viewLog(){
  const r = await fetch("/api/log_tail");
  const s = await r.json();
  document.getElementById("logview").textContent = s.tail || "";
}

poll();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(PAGE_HTML, title=APP_TITLE, config_path=CONFIG_PATH)

@app.get("/api/state")
def api_state():
    return jsonify(state.public_state())

@app.post("/api/save_config")
def api_save_config():
    if not request.is_json:
        return jsonify({"ok": False, "error": "Body JSON richiesto"}), 400
    cfg = request.get_json()
    if not isinstance(cfg, dict):
        return jsonify({"ok": False, "error": "Config deve essere un oggetto JSON"}), 400

    err = validate_config(cfg)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    try:
        state.update_config(cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 409

@app.post("/api/reload_config")
def api_reload_config():
    try:
        state.reload_config()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 409

@app.post("/api/reset")
def api_reset():
    state.reset()
    return jsonify({"ok": True})

@app.post("/api/step")
def api_step():
    st = state.public_state()
    if st.get("running"):
        return jsonify({"ok": False, "error": "Stop prima di usare Step."}), 409

    def _do():
        state.step_one_turn()

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok": True})

@app.post("/api/start")
def api_start():
    turns = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if isinstance(body, dict) and "turns" in body:
            try:
                turns = int(body["turns"])
            except Exception:
                turns = None
    state.start(turns=turns)
    return jsonify({"ok": True})

@app.post("/api/stop")
def api_stop():
    state.stop()
    return jsonify({"ok": True})

@app.get("/api/log_tail")
def api_log_tail():
    st = state.public_state()
    cfg = st.get("config") or {}
    log_cfg = (cfg.get("log") or {})
    enabled = bool(log_cfg.get("enabled"))
    path = log_cfg.get("path", "twochat.log")

    if not enabled:
        return jsonify({"tail": "(log disabilitato in config)"}), 200
    if not os.path.exists(path):
        return jsonify({"tail": "(log non trovato)"}), 200

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return jsonify({"tail": "".join(lines[-160:])})
    except Exception as e:
        return jsonify({"tail": f"(errore lettura log: {e})"}), 200

@app.get("/download_log")
def download_log():
    st = state.public_state()
    cfg = st.get("config") or {}
    log_cfg = (cfg.get("log") or {})
    enabled = bool(log_cfg.get("enabled"))
    path = log_cfg.get("path", "twochat.log")

    if not enabled:
        return Response("Log disabilitato in config.\n", mimetype="text/plain")
    if not os.path.exists(path):
        return Response("Log non trovato.\n", mimetype="text/plain")
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

def main():
    cfg = load_config()
    ui = cfg.get("ui", {})
    host = ui.get("host", "127.0.0.1")
    port = int(ui.get("port", 5001))
    print(f"{APP_TITLE} → http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
