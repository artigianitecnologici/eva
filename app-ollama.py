#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import importlib.util
from time import time
from datetime import datetime
from tempfile import NamedTemporaryFile

from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, flash
from ollama import Client
import requests

# ========= Paths & Config =========
BASE_PATH = os.path.abspath("./")
CONFIG_DIR = os.path.join(BASE_PATH, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
COMMANDS_PATH = os.path.join(CONFIG_DIR, "comandi.json")
LOG_PATH = os.path.join(BASE_PATH, "log")
HANDLERS_PATH = os.path.join(BASE_PATH, "handlers")
STATE_FILE = os.path.join(LOG_PATH, "command_mode.state")

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(HANDLERS_PATH, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

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

# ---- helpers: read/write JSON atomico
def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Impossibile leggere/parsare {path}: {e}")
        return default

def _write_json_atomic(path, data_obj):
    tmp = None
    try:
        d = json.dumps(data_obj, ensure_ascii=False, indent=2)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tf:
            tf.write(d)
            tmp = tf.name
        os.replace(tmp, path)  # atomico su POSIX
        return True
    except Exception as e:
        log_error(f"Scrittura atomica fallita per {path}: {e}")
        try:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False

# ======= Caricamento config iniziale =======
CONFIG = _read_json(CONFIG_PATH, default={
    "ollama_host": "http://127.0.0.1:11434",
    "default_model": "llama3:latest",
    "prompt_system": "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano."
})
if CONFIG is None:
    log_error(f"File di configurazione mancante o invalido: {CONFIG_PATH}")
    sys.exit(1)

# Riferimenti "vivi" aggiornabili a runtime
OLLAMA_BASE = CONFIG.get("ollama_host", "http://127.0.0.1:11434")
DEFAULT_MODEL = CONFIG.get("default_model", "llama3:latest")
PROMPT_SYSTEM = CONFIG.get("prompt_system", "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.")

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

def sanitize_chunk(text: str) -> str:
    # rimuove markdown '*', '**' al volo, chunk-safe
    if not isinstance(text, str):
        return text
    return text.replace("**", "").replace("*", "")

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

def stream_response(messages, model_name: str = DEFAULT_MODEL):
    """Generatore chunk-by-chunk con ollama_client.chat(..., stream=True)."""
    def _generator():
        print(f"[DEBUG] STREAM avviato verso modello {model_name}", file=sys.stderr)
        try:
            for part in ollama_client.chat(model=model_name, messages=messages, stream=True):
                # formati tipici: {'message': {'role':'assistant','content':'...'}, 'done':False}
                content = ""
                try:
                    if isinstance(part, dict):
                        msg = part.get("message") or {}
                        content = msg.get("content") or ""
                except Exception:
                    content = ""
                if content:
                    yield sanitize_chunk(content)
            # opzionale: marcatore fine
            # yield "\n"
        except Exception as e:
            err = f"\n[errore stream: {e}]"
            log_error(err)
            yield err
    return _generator

# ========= Handler Loader (plugin locali) =========
_LOADED_HANDLERS = []

def _load_handlers():
    global _LOADED_HANDLERS
    _LOADED_HANDLERS = []
    if not os.path.isdir(HANDLERS_PATH):
        return
    for fname in os.listdir(HANDLERS_PATH):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(HANDLERS_PATH, fname)
        mod_name = f"handlers.{fname[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            if hasattr(module, "can_handle") and hasattr(module, "handle"):
                _LOADED_HANDLERS.append(module)
                log_info(f"Handler caricato: {fname}")
            else:
                log_error(f"Handler {fname} ignorato: mancano funzioni can_handle/handle")
        except Exception as e:
            log_error(f"Errore caricando handler {fname}: {e}")

def try_local_handlers(text: str):
    ctx = {"config": CONFIG}
    for mod in _LOADED_HANDLERS:
        try:
            if mod.can_handle(text, ctx):
                reply = mod.handle(text, ctx)
                if isinstance(reply, str) and reply.strip():
                    return sanitize_chunk(reply)
        except Exception as e:
            log_error(f"Errore in handler {getattr(mod, '__name__', mod)}: {e}")
    return None

# ========= Modalità comandi =========
def _load_commands():
    data = _read_json(COMMANDS_PATH, default={
        "prefix": "#@#",
        "start": [r"avvia\s+programmazione"],
        "stop": [r"fine\s+programmazione", r"\bstop\b"],
        "status": [r"\bstato\s+programmazione\b"]
    })
    data.setdefault("prefix", "#@#")
    data.setdefault("start", [])
    data.setdefault("stop", [])
    data.setdefault("status", [])
    return data

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
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")  # per flash messaggi

@app.route("/")
def home():
    # elenco modelli per select
    model_names = []
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        model_names = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
    except Exception as e:
        log_error(f"Errore durante il recupero modelli da {OLLAMA_BASE} - {e}")

    if not model_names:
        model_names = [DEFAULT_MODEL]
    return render_template("indexollama.html", models=model_names, default_model=DEFAULT_MODEL)

# ---- API modelli (opzionale per AJAX)
@app.route("/models")
def models():
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        names = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
        if not names:
            names = [DEFAULT_MODEL]
        return jsonify({"models": names})
    except Exception as e:
        log_error(f"/models error: {e}")
        return jsonify({"models": [DEFAULT_MODEL], "error": str(e)}), 200

# ---- PIPELINE sincrona (legacy)
def _answer_pipeline(user_text: str, model: str):
    t = (user_text or "").strip()

    # comandi
    if _match_any(COMANDI.get("start"), t):
        _write_command_mode(True)
        return "Modalita comandi ATTIVATA"
    if _match_any(COMANDI.get("stop"), t):
        _write_command_mode(False)
        return "Modalita comandi DISATTIVATA. Torno a usare il modello."
    if _match_any(COMANDI.get("status"), t):
        return "Modalita comandi: ON" if _read_command_mode() else "Modalita comandi: OFF"

    # se attiva -> echo prefissato
    if _read_command_mode():
        return f"{CMD_PREFIX}{t if t else '(vuoto)'}"

    # handler locali
    local = try_local_handlers(t)
    if local is not None:
        return local

    # modello
    messages = [{"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": t}]
    new_msg = get_response(messages, model)
    msgout = split_string(new_msg.get('content', new_msg))
    msgout = sanitize_chunk(msgout)
    return msgout

# ---- PIPELINE streaming
def _answer_pipeline_stream(user_text: str, model: str):
    t = (user_text or "").strip()

    # comandi (risposta immediata non-stream)
    if _match_any(COMANDI.get("start"), t):
        _write_command_mode(True)
        return "text", ("Modalita comandi ATTIVATA",)

    if _match_any(COMANDI.get("stop"), t):
        _write_command_mode(False)
        return "text", ("Modalita comandi DISATTIVATA. Torno a usare il modello.",)

    if _match_any(COMANDI.get("status"), t):
        status = "Modalita comandi: ON" if _read_command_mode() else "Modalita comandi: OFF"
        return "text", (status,)

    if _read_command_mode():
        return "text", (f"{CMD_PREFIX}{t if t else '(vuoto)'}",)

    # handler locali (non stream)
    local = try_local_handlers(t)
    if local is not None:
        return "text", (local,)

    # stream modello
    messages = [{"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": t}]
    gen = stream_response(messages, model)
    return "stream", (gen,)

# ---- Endpoint classici
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

# ---- Endpoint streaming (chunked transfer, text/plain)
@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    model = (data.get("model") or DEFAULT_MODEL).strip()

    mode, payload = _answer_pipeline_stream(q, model)
    if mode == "text":
        # risposta non-stream (comandi/handler)
        text = payload[0]
        log_to_file(q, text)
        return Response(text, mimetype="text/plain")

    # mode == "stream"
    gen = payload[0]

    @stream_with_context
    def _wrapped():
        buf = []
        for chunk in gen():
            buf.append(chunk)
            yield chunk
        # log intero alla fine
        try:
            log_to_file(q, "".join(buf))
        except Exception as e:
            log_error(f"log stream fallito: {e}")

    return Response(_wrapped(), mimetype="text/plain")

# ---- Healthcheck
@app.route('/healthz')
def healthz():
    ok = check_ollama_connectivity(False)
    return ("ok", 200) if ok else ("ollama unreachable", 503)

# ---- Pannello Config
@app.route("/config", methods=["GET", "POST"])
def config_page():
    global CONFIG, OLLAMA_BASE, DEFAULT_MODEL, PROMPT_SYSTEM, ollama_client, COMANDI, CMD_PREFIX
    if request.method == "POST":
        form = request.form
        # Selezione: quale file stiamo salvando
        target = form.get("target", "config")
        if target == "config":
            new_conf = {
                "ollama_host": form.get("ollama_host", OLLAMA_BASE).strip(),
                "default_model": form.get("default_model", DEFAULT_MODEL).strip(),
                "prompt_system": form.get("prompt_system", PROMPT_SYSTEM),
            }
            if _write_json_atomic(CONFIG_PATH, new_conf):
                CONFIG = new_conf
                OLLAMA_BASE = CONFIG.get("ollama_host", OLLAMA_BASE)
                DEFAULT_MODEL = CONFIG.get("default_model", DEFAULT_MODEL)
                PROMPT_SYSTEM = CONFIG.get("prompt_system", PROMPT_SYSTEM)
                ollama_client = Client(host=OLLAMA_BASE)  # ricollega client
                flash("config.json aggiornato correttamente.", "success")
            else:
                flash("Errore nel salvataggio di config.json", "error")
        elif target == "comandi":
            # L’editor invia il JSON grezzo del file comandi
            raw = form.get("comandi_raw", "").strip()
            try:
                data = json.loads(raw) if raw else {}
                if _write_json_atomic(COMMANDS_PATH, data):
                    COMANDI = _load_commands()
                    CMD_PREFIX = COMANDI.get("prefix", "#@#")
                    flash("comandi.json aggiornato correttamente.", "success")
                else:
                    flash("Errore nel salvataggio di comandi.json", "error")
            except Exception as e:
                flash(f"comandi.json non valido: {e}", "error")
        return redirect(url_for("config_page"))

    # GET -> mostra valori correnti
    cfg = _read_json(CONFIG_PATH, default=CONFIG) or CONFIG
    cmd = _read_json(COMMANDS_PATH, default=COMANDI) or COMANDI
    # modelli per select
    models = []
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        models = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
    except Exception as e:
        log_error(f"/config fetch models error: {e}")
    if not models:
        models = [DEFAULT_MODEL]

    return render_template("config.html",
                           cfg=cfg,
                           comandi_json=json.dumps(cmd, ensure_ascii=False, indent=2),
                           models=models)

# ---- Ricarica handler a caldo
@app.route("/reload-handlers", methods=["POST"])
def reload_handlers():
    _load_handlers()
    return jsonify({"status": "ok", "loaded": [getattr(m, "__name__", "handler") for m in _LOADED_HANDLERS]})

if __name__ == '__main__':
    log_info(f"Avvio app-ollama | OLLAMA_BASE={OLLAMA_BASE} | DEFAULT_MODEL={DEFAULT_MODEL}")
    _load_handlers()
    check_ollama_connectivity(False)
    app.run(host='0.0.0.0', debug=True, port=5000)
