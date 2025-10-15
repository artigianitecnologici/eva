#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import importlib.util
from time import time
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from ollama import Client
import requests

# ========= Paths & Config =========
BASE_PATH = os.path.abspath("./")
CONFIG_PATH = os.path.join(BASE_PATH, "config", "config.json")
# --- PATCH: command mode paths ---
COMMANDS_PATH = os.path.join(BASE_PATH, "config", "comandi.json")
LOG_PATH = os.path.join(BASE_PATH, "log")
HANDLERS_PATH = os.path.join(BASE_PATH, "handlers")
STATE_FILE = os.path.join(LOG_PATH, "command_mode.state")
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(HANDLERS_PATH, exist_ok=True)  # in caso non esista

def _now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def log_error(msg: str):
    line = f"[{_now()}] [ERROR] {msg}"
    print(line, file=sys.stderr)
    try:
        with open(os.path.join(LOG_PATH, "error.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def log_info(msg: str):
    print(f"[{_now()}] [INFO] {msg}", file=sys.stderr)

# Caricamento config
if not os.path.exists(CONFIG_PATH):
    log_error(f"File di configurazione mancante: {CONFIG_PATH}")
    sys.exit(1)
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except Exception as e:
    log_error(f"Impossibile leggere/parsare il config: {CONFIG_PATH} - {e}")
    sys.exit(1)

OLLAMA_BASE = CONFIG.get("ollama_host", "http://127.0.0.1:11434")
DEFAULT_MODEL = CONFIG.get("default_model", "llama3:latest")
PROMPT_SYSTEM = CONFIG.get("prompt_system",
                           "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.")

# ========= Client Ollama =========
ollama_client = Client(host=OLLAMA_BASE)

# ========= Utility =========
def log_to_file(question, bot_answer):
    try:
        with open(os.path.join(LOG_PATH, "log.txt"), "a", encoding="utf-8") as log_file:
            log_file.write(f"{_now()}\n[QUESTION]: {question};[OLLAMA]: {bot_answer}\n")
        if bot_answer:
            with open(os.path.join(LOG_PATH, "user.txt"), "a", encoding="utf-8") as bot_file:
                bot_file.write("user: " + str(question) + "\n")
                bot_file.write("bot: " + str(bot_answer) + "\n")
    except Exception as e:
        log_error(f"Errore scrivendo i log conversazione: {e}")

def split_string(msg):
    print(f"[DEBUG] Risposta grezza del modello: {msg}", file=sys.stderr)
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict) and "content" in msg:
        return str(msg["content"])
    return "(errore nella risposta del modello: contenuto non leggibile)"

def sanitize_response(text: str) -> str:
    """Rimuove i doppi asterischi ** (markdown bold)."""
    if not isinstance(text, str):
        return text
    text = text.replace("**", "")
    # opzionale: rimuovere anche __ e backtick
    # text = text.replace("__", "").replace("`", "")
    return text

def check_ollama_connectivity(raise_on_fail=False):
    url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        _ = r.json()
        log_info(f"Connessione a Ollama OK su {OLLAMA_BASE}")
        return True
    except Exception as e:
        log_error(
            f"Impossibile connettersi a Ollama su {OLLAMA_BASE} - {e}\n"
            "Verifica 'ollama serve', porta 11434 su 127.0.0.1 e valore 'ollama_host' nel config."
        )
        if raise_on_fail:
            raise
        return False

def get_response(messages, model_name: str = DEFAULT_MODEL):
    print(f"[DEBUG] Messaggi inviati al modello ({model_name}): {messages}", file=sys.stderr)
    try:
        start = time()
        response = ollama_client.chat(model=model_name, messages=messages)
        print(f"[DEBUG] Risposta completa ricevuta: {response}", file=sys.stderr)
        print(f"[DEBUG] Tempo di risposta del modello: {time()-start:.2f} secondi", file=sys.stderr)

        # Normalizzazione: supporta oggetto o dict
        content = None
        try:
            msg_obj = getattr(response, "message", None)
            if msg_obj is not None:
                content = getattr(msg_obj, "content", None)
        except Exception:
            pass
        if content is None and isinstance(response, dict):
            msg_dict = response.get("message")
            if isinstance(msg_dict, dict):
                content = msg_dict.get("content")
        if content is None and isinstance(response, dict) and "content" in response:
            content = response["content"]

        if not isinstance(content, str) or not content.strip():
            log_error("Formato risposta inatteso da Ollama: impossibile estrarre 'message.content'.")
            return {"content": "(errore: formato risposta inatteso da Ollama)"}
        return {"content": content}
    except Exception as e:
        log_error(f"Chiamata a Ollama fallita (host: {OLLAMA_BASE}, model: {model_name}) - {e}")
        return {"content": f"(errore: impossibile contattare Ollama su {OLLAMA_BASE} - {e})"}

# ========= Handler Loader (plugin locali) =========
# Ogni handler e un file .py in ./handlers/ con due funzioni:
#   can_handle(text:str, context:dict) -> bool
#   handle(text:str, context:dict) -> str
# Vengono caricati dinamicamente all'avvio.
_LOADED_HANDLERS = []

def _load_handlers():
    global _LOADED_HANDLERS
    _LOADED_HANDLERS = []
    if not os.path.isdir(HANDLERS_PATH):
        return

    for fname in os.listdir(HANDLERS_PATH):
        if not fname.endswith(".py"):
            continue
        if fname.startswith("_"):
            continue
        fpath = os.path.join(HANDLERS_PATH, fname)
        mod_name = f"handlers.{fname[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            # controlla che abbia le due funzioni richieste
            if hasattr(module, "can_handle") and hasattr(module, "handle"):
                _LOADED_HANDLERS.append(module)
                log_info(f"Handler caricato: {fname}")
            else:
                log_error(f"Handler {fname} ignorato: mancano funzioni can_handle/handle")
        except Exception as e:
            log_error(f"Errore caricando handler {fname}: {e}")

def try_local_handlers(text: str):
    """Passa il testo a tutti gli handler caricati, restituisce la prima risposta che matcha o None."""
    ctx = {"config": CONFIG}
    for mod in _LOADED_HANDLERS:
        try:
            if mod.can_handle(text, ctx):
                reply = mod.handle(text, ctx)
                if isinstance(reply, str) and reply.strip():
                    return sanitize_response(reply)
        except Exception as e:
            log_error(f"Errore in handler {getattr(mod, '__name__', mod)}: {e}")
    return None

# ========= PATCH: Modalita comandi =========

def _load_commands():
    """Legge config/comandi.json. Se assente o invalido, usa fallback minimal."""
    if not os.path.exists(COMMANDS_PATH):
        log_error(f"File comandi mancante: {COMMANDS_PATH}")
        return {
            "prefix": "#@#",
            "start": [r"avvia\s+programmazione"],
            "stop": [r"fine\s+programmazione", r"\bstop\b"],
            "status": [r"\bstato\s+programmazione\b"]
        }
    try:
        with open(COMMANDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("prefix", "#@#")
        data.setdefault("start", [])
        data.setdefault("stop", [])
        data.setdefault("status", [])
        return data
    except Exception as e:
        log_error(f"Impossibile leggere/parsare {COMMANDS_PATH} - {e}")
        return {"prefix": "#@#", "start": [], "stop": [], "status": []}

COMANDI = _load_commands()
CMD_PREFIX = COMANDI.get("prefix", "#@#")

def _read_command_mode() -> bool:
    try:
        if not os.path.exists(STATE_FILE):
            return False
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() == "1"
    except Exception:
        return False

def _write_command_mode(on: bool) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write("1" if on else "0")
    except Exception:
        pass

def _match_any(patterns, text: str) -> bool:
    t = (text or "").strip()
    for p in patterns or []:
        try:
            if re.search(p, t, flags=re.IGNORECASE):
                return True
        except re.error as e:
            log_error(f"Regex non valida in comandi.json ('{p}'): {e}")
    return False

# ========= Flask =========
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        model_names = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
        if not model_names:
            model_names = [DEFAULT_MODEL]
        print(f"[DEBUG] Modelli disponibili: {model_names}", file=sys.stderr)
    except Exception as e:
        log_error(f"Errore durante il recupero modelli da {OLLAMA_BASE} - {e}")
        model_names = [DEFAULT_MODEL]
    return render_template("indexollama.html", models=model_names)

def _answer_pipeline(user_text: str, model: str):
    t = (user_text or "").strip()

    # --- PRIORITA: comandi start/stop/status ---
    if _match_any(COMANDI.get("start"), t):
        _write_command_mode(True)
        return ("Modalita comandi ATTIVATA")
    if _match_any(COMANDI.get("stop"), t):
        _write_command_mode(False)
        return "Modalita comandi DISATTIVATA. Torno a usare il modello."
    if _match_any(COMANDI.get("status"), t):
        return "Modalita comandi: ON" if _read_command_mode() else "Modalita comandi: OFF"

    # --- Se la modalita comandi e attiva: niente handler, niente modello ---
    if _read_command_mode():
        return f"{CMD_PREFIX}{t if t else '(vuoto)'}"

    # 1) prova handler locali
    local = try_local_handlers(t)
    if local is not None:
        return local

    # 2) fallback -> LLM
    messages = [{"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": t}]
    new_msg = get_response(messages, model)
    msgout = split_string(new_msg.get('content', new_msg))
    msgout = sanitize_response(msgout)
    return msgout

@app.route("/get")
def get_bot_response():
    q = (request.args.get('msg') or '').strip()
    model = (request.args.get('model') or DEFAULT_MODEL).strip()
    msgout = _answer_pipeline(q, model)
    log_to_file(q, msgout)
    return msgout

@app.route('/bot')
def bot():
    q = (request.args.get('query') or '').strip()
    model = (request.args.get('model') or DEFAULT_MODEL).strip()
    msgout = _answer_pipeline(q, model)
    log_to_file(q, msgout)
    return msgout

@app.route('/json', methods=['GET', 'POST'])
def json_response():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        q = (data.get('query') or '').strip()
        model = (data.get('model') or DEFAULT_MODEL).strip()
    else:
        q = (request.args.get('query') or '').strip()
        model = (request.args.get('model') or DEFAULT_MODEL).strip()

    msgout = _answer_pipeline(q, model)
    log_to_file(q, msgout)
    return jsonify({"response": msgout, "action": "ok"})

@app.route('/healthz')
def healthz():
    ok = check_ollama_connectivity(False)
    return ("ok", 200) if ok else ("ollama unreachable", 503)

if __name__ == '__main__':
    log_info(f"Avvio app-ollama | OLLAMA_BASE={OLLAMA_BASE} | DEFAULT_MODEL={DEFAULT_MODEL}")
    _load_handlers()  # <=== carica gli handler locali all'avvio
    check_ollama_connectivity(False)
    app.run(host='0.0.0.0', debug=True, port=5000)
