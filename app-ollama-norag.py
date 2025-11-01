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

from flask import (
    Flask, render_template, request, jsonify, Response, stream_with_context,
    redirect, url_for, flash
)
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
BACKUP_DIR = os.path.join(CONFIG_DIR, "backups")

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(HANDLERS_PATH, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

def _now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def _stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

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

# ---- helpers: read/write JSON atomico + backup
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

def _safe_write_config(new_cfg: dict) -> bool:
    """Crea backup e scrive atomico il config.
       Se fallisce, lascia il vecchio intatto."""
    try:
        current = _read_json(CONFIG_PATH, default=None)
        if current is not None:
            # backup time-stamped + ultimo backup "config.backup.json"
            ts = _stamp()
            _write_json_atomic(os.path.join(BACKUP_DIR, f"config.{ts}.bak.json"), current)
            _write_json_atomic(os.path.join(CONFIG_DIR, "config.backup.json"), current)
    except Exception as e:
        log_error(f"Backup config fallito: {e}")
    return _write_json_atomic(CONFIG_PATH, new_cfg)

# ======= Config iniziale (compatibile con versioni vecchie) =======
CONFIG = _read_json(CONFIG_PATH, default={
    "ollama_host": "http://127.0.0.1:11434",
    "default_model": "llama3:latest",
    "prompt_system": "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.",
    "default_profile": "default",
    "profiles": {
        "default": {
            "label": "Default",
            "model": "llama3:latest",
            "system": "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.",
            "options": {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "num_ctx": 4096,
                "num_predict": 512,
                "repeat_penalty": 1.1
            }
        }
    }
})
if CONFIG is None:
    log_error(f"File di configurazione mancante o invalido: {CONFIG_PATH}")
    sys.exit(1)

def _normalize_config(cfg: dict) -> dict:
    """Garantisce chiavi minime e coerenza del default_profile."""
    cfg = dict(cfg or {})
    cfg.setdefault("ollama_host", "http://127.0.0.1:11434")
    cfg.setdefault("default_model", "llama3:latest")
    cfg.setdefault("prompt_system", "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.")
    cfg.setdefault("default_profile", "default")
    cfg.setdefault("profiles", {})
    if not isinstance(cfg["profiles"], dict):
        cfg["profiles"] = {}
    # se il default non esiste, scegline uno
    if cfg["default_profile"] not in cfg["profiles"]:
        cfg["default_profile"] = "default" if "default" in cfg["profiles"] else (next(iter(cfg["profiles"]), ""))
    return cfg

CONFIG = _normalize_config(CONFIG)

# Riferimenti runtime
OLLAMA_BASE = CONFIG.get("ollama_host", "http://127.0.0.1:11434")
DEFAULT_MODEL = CONFIG.get("default_model", "llama3:latest")
PROMPT_SYSTEM = CONFIG.get("prompt_system", "Sei E.V.A. Enhanced Virtual Assistant, rispondi in italiano.")
DEFAULT_PROFILE = CONFIG.get("default_profile", "default")
PROFILES = CONFIG.get("profiles", {})

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
            "Verifica 'ollama serve', porta 11434 e valore 'ollama_host' nel config."
        )
        if raise_on_fail:
            raise
        return False

# ---- Profili e opzioni
def _get_profile(name: str):
    if not name:
        name = CONFIG.get("default_profile") or "default"
    prof = CONFIG.get("profiles", {}).get(name)
    if prof:
        return name, prof
    # fallback compatibilità
    fallback = {
        "label": "Compat",
        "model": CONFIG.get("default_model", "llama3:latest"),
        "system": CONFIG.get("prompt_system", PROMPT_SYSTEM),
        "options": {}
    }
    return "compat", fallback

def _merge_options(base: dict, extra: dict) -> dict:
    out = dict(base or {})
    for k, v in (extra or {}).items():
        out[k] = v
    return out

def get_response(messages, model_name: str, options: dict):
    print(f"[DEBUG] Messaggi inviati al modello ({model_name}) options={options}: {messages}", file=sys.stderr)
    try:
        start = time()
        response = ollama_client.chat(model=model_name, messages=messages, options=options or {})
        print(f"[DEBUG] Risposta completa: {response}", file=sys.stderr)
        print(f"[DEBUG] Tempo risposta: {time()-start:.2f}s", file=sys.stderr)

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

def stream_response(messages, model_name: str, options: dict):
    def _generator():
        print(f"[DEBUG] STREAM → {model_name} options={options}", file=sys.stderr)
        try:
            for part in ollama_client.chat(model=model_name, messages=messages, options=options or {}, stream=True):
                content = ""
                try:
                    if isinstance(part, dict):
                        msg = part.get("message") or {}
                        content = msg.get("content") or ""
                except Exception:
                    content = ""
                if content:
                    yield sanitize_chunk(content)
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
                log_error(f"Handler {fname} ignorato: mancano can_handle/handle")
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
def _read_commands():
    data = _read_json(COMMANDS_PATH, default={
        "prefix": "#@#",
        "start": [r"avvia\s+programmazione"],
        "stop": [r"fine\s+programmazione", r"\bstop\b"],
        "status": [r"\bstato\s+programmazione\b"]
    }) or {}
    data.setdefault("prefix", "#@#")
    data.setdefault("start", [])
    data.setdefault("stop", [])
    data.setdefault("status", [])
    return data

COMANDI = _read_commands()
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

# ========= Risoluzione esecuzione (modello, system, options)
def _resolve_run_settings(model_from_req: str, profile_name: str):
    prof_name, prof = _get_profile(profile_name)
    model = (prof.get("model") or model_from_req or DEFAULT_MODEL).strip()
    system = (prof.get("system") or PROMPT_SYSTEM).strip()
    options = prof.get("options") or {}
    if model_from_req:
        model = model_from_req.strip()
    return model, system, options

# ========= Flask =========
app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

# ---------- CHAT ----------
@app.route("/")
def home():
    model_names = []
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        model_names = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
    except Exception as e:
        log_error(f"Errore recupero modelli da {OLLAMA_BASE} - {e}")
    if not model_names:
        model_names = [DEFAULT_MODEL]
    return render_template("indexollama.html",
                           models=model_names,
                           default_model=DEFAULT_MODEL,
                           profiles=CONFIG.get("profiles", {}),
                           default_profile=CONFIG.get("default_profile", "default"))

def _answer_pipeline(user_text: str, model: str, profile: str):
    t = (user_text or "").strip()
    if _match_any(COMANDI.get("start"), t):
        _write_command_mode(True)
        return "Modalita comandi ATTIVATA"
    if _match_any(COMANDI.get("stop"), t):
        _write_command_mode(False)
        return "Modalita comandi DISATTIVATA. Torno a usare il modello."
    if _match_any(COMANDI.get("status"), t):
        return "Modalita comandi: ON" if _read_command_mode() else "Modalita comandi: OFF"
    if _read_command_mode():
        return f"{CMD_PREFIX}{t if t else '(vuoto)'}"

    local = try_local_handlers(t)
    if local is not None:
        return local

    model_res, system_prompt, options = _resolve_run_settings(model, profile)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": t}]
    new_msg = get_response(messages, model_res, options)
    msgout = split_string(new_msg.get('content', new_msg))
    msgout = sanitize_chunk(msgout)
    return msgout

def _answer_pipeline_stream(user_text: str, model: str, profile: str):
    t = (user_text or "").strip()
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

    local = try_local_handlers(t)
    if local is not None:
        return "text", (local,)

    model_res, system_prompt, options = _resolve_run_settings(model, profile)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": t}]
    gen = stream_response(messages, model_res, options)
    return "stream", (gen,)

@app.route("/get")
def get_bot_response():
    q = (request.args.get('msg') or '').strip()
    model = (request.args.get('model') or DEFAULT_MODEL).strip()
    profile = (request.args.get('profile') or CONFIG.get("default_profile", "default")).strip()
    msgout = _answer_pipeline(q, model, profile)
    log_to_file(q, msgout)
    return msgout

@app.route('/bot')
def bot():
    q = (request.args.get('query') or '').strip()
    model = (request.args.get('model') or DEFAULT_MODEL).strip()
    profile = (request.args.get('profile') or CONFIG.get("default_profile", "default")).strip()
    msgout = _answer_pipeline(q, model, profile)
    log_to_file(q, msgout)
    return msgout

@app.route('/json', methods=['GET', 'POST'])
def json_response():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        q = (data.get('query') or '').strip()
        model = (data.get('model') or DEFAULT_MODEL).strip()
        profile = (data.get('profile') or CONFIG.get("default_profile", "default")).strip()
    else:
        q = (request.args.get('query') or '').strip()
        model = (request.args.get('model') or DEFAULT_MODEL).strip()
        profile = (request.args.get('profile') or CONFIG.get("default_profile", "default")).strip()

    msgout = _answer_pipeline(q, model, profile)
    log_to_file(q, msgout)
    return jsonify({"response": msgout, "action": "ok"})

@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    model = (data.get("model") or DEFAULT_MODEL).strip()
    profile = (data.get("profile") or CONFIG.get("default_profile", "default")).strip()

    mode, payload = _answer_pipeline_stream(q, model, profile)
    if mode == "text":
        text = payload[0]
        log_to_file(q, text)
        return Response(text, mimetype="text/plain")

    gen = payload[0]
    @stream_with_context
    def _wrapped():
        buf = []
        for chunk in gen():
            buf.append(chunk)
            yield chunk
        try:
            log_to_file(q, "".join(buf))
        except Exception as e:
            log_error(f"log stream fallito: {e}")
    return Response(_wrapped(), mimetype="text/plain")

@app.route('/healthz')
def healthz():
    ok = check_ollama_connectivity(False)
    return ("ok", 200) if ok else ("ollama unreachable", 503)

# ---------- FORMS UTILS ----------
def _to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None

def _to_int(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(v))
    except Exception:
        return None

def _to_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    arr = [s.strip() for s in str(v).split(",")]
    return [s for s in arr if s]

def _options_from_form(form, allow_raw_merge=True):
    opts = {}
    # Campi più usati in Ollama
    m = {
        "temperature": _to_float(form.get("temperature")),
        "top_p": _to_float(form.get("top_p")),
        "top_k": _to_int(form.get("top_k")),
        "num_ctx": _to_int(form.get("num_ctx")),
        "num_predict": _to_int(form.get("num_predict")),
        "repeat_penalty": _to_float(form.get("repeat_penalty")),
        "seed": _to_int(form.get("seed")),
    }
    for k, v in m.items():
        if v is not None:
            opts[k] = v

    st = _to_list(form.get("stop"))
    if st:
        opts["stop"] = st

    if allow_raw_merge:
        options_raw = (form.get("options_raw") or "").strip()
        if options_raw:
            try:
                raw = json.loads(options_raw)
                if isinstance(raw, dict):
                    opts = _merge_options(opts, raw)
                else:
                    flash("options_raw non è un oggetto JSON (verrà ignorato).", "warning")
            except Exception as e:
                flash(f"options_raw non valido: {e}", "error")
                # non interrompo, ma non salvo nulla di options_raw
    return opts

# ---------- CONFIG GENERALE ----------
@app.route("/config", methods=["GET", "POST"])
def config_page():
    global CONFIG, OLLAMA_BASE, DEFAULT_MODEL, PROMPT_SYSTEM, DEFAULT_PROFILE, PROFILES, ollama_client
    if request.method == "POST":
        new_conf = dict(CONFIG)  # merge safe
        new_conf["ollama_host"]   = (request.form.get("ollama_host") or OLLAMA_BASE).strip()
        new_conf["default_model"] = (request.form.get("default_model") or DEFAULT_MODEL).strip()
        new_conf["prompt_system"] = request.form.get("prompt_system", PROMPT_SYSTEM)
        dp = (request.form.get("default_profile") or DEFAULT_PROFILE).strip()
        if dp and dp in CONFIG.get("profiles", {}):
            new_conf["default_profile"] = dp
        else:
            flash("Profilo predefinito non valido: lascio quello precedente.", "warning")

        new_conf = _normalize_config(new_conf)
        if _safe_write_config(new_conf):
            CONFIG = new_conf
            OLLAMA_BASE = CONFIG.get("ollama_host", OLLAMA_BASE)
            DEFAULT_MODEL = CONFIG.get("default_model", DEFAULT_MODEL)
            PROMPT_SYSTEM = CONFIG.get("prompt_system", PROMPT_SYSTEM)
            DEFAULT_PROFILE = CONFIG.get("default_profile", DEFAULT_PROFILE)
            PROFILES = CONFIG.get("profiles", {})
            ollama_client = Client(host=OLLAMA_BASE)
            flash("config.json aggiornato correttamente.", "success")
        else:
            flash("Errore nel salvataggio di config.json (backup preservato).", "error")
        return redirect(url_for("config_page"))

    # GET
    # elenco modelli per select
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

    return render_template("config_general.html",
                           cfg=CONFIG,
                           models=models,
                           profiles=CONFIG.get("profiles", {}),
                           default_profile=CONFIG.get("default_profile", "default"))

# ---------- PROFILES: LISTA ----------
@app.route("/profiles")
def profiles_list():
    # elenco modelli per info tabella (facoltativo)
    models = []
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        models = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
    except Exception as e:
        log_error(f"/profiles fetch models error: {e}")
    return render_template("profiles_list.html",
                           profiles=CONFIG.get("profiles", {}),
                           default_profile=CONFIG.get("default_profile", "default"),
                           models=models)

# ---------- PROFILES: NUOVO ----------
@app.route("/profiles/new", methods=["GET", "POST"])
def profile_new():
    global CONFIG, PROFILES
    if request.method == "POST":
        pname = (request.form.get("profile_name") or "").strip()
        if not pname:
            flash("Nome profilo mancante.", "error")
            return redirect(url_for("profile_new"))
        if pname in CONFIG.get("profiles", {}):
            flash("Esiste già un profilo con questo nome.", "error")
            return redirect(url_for("profile_new"))

        label = (request.form.get("label") or pname).strip()
        model = (request.form.get("model") or DEFAULT_MODEL).strip()
        system = (request.form.get("system") or PROMPT_SYSTEM).strip()
        options = _options_from_form(request.form, allow_raw_merge=True)

        new_conf = dict(CONFIG)
        profiles = dict(new_conf.get("profiles", {}))
        profiles[pname] = {
            "label": label,
            "model": model,
            "system": system,
            "options": options
        }
        new_conf["profiles"] = profiles
        new_conf = _normalize_config(new_conf)

        if _safe_write_config(new_conf):
            CONFIG = new_conf
            PROFILES = CONFIG.get("profiles", {})
            flash(f"Profilo '{pname}' creato.", "success")
            return redirect(url_for("profiles_list"))
        else:
            flash("Errore nel salvataggio del profilo (backup preservato).", "error")
            return redirect(url_for("profile_new"))

    # GET
    models = []
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        models = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
    except Exception as e:
        log_error(f"/profiles/new fetch models error: {e}")
    if not models:
        models = [DEFAULT_MODEL]
    return render_template("profile_new.html", models=models, default_system=PROMPT_SYSTEM)

# ---------- PROFILES: EDIT ----------
@app.route("/profiles/<name>/edit", methods=["GET", "POST"])
def profile_edit(name):
    global CONFIG, PROFILES
    name = (name or "").strip()
    if name not in CONFIG.get("profiles", {}):
        flash("Profilo inesistente.", "error")
        return redirect(url_for("profiles_list"))

    if request.method == "POST":
        label = (request.form.get("label") or name).strip()
        model = (request.form.get("model") or DEFAULT_MODEL).strip()
        system = (request.form.get("system") or PROMPT_SYSTEM).strip()
        options = _options_from_form(request.form, allow_raw_merge=True)

        new_conf = dict(CONFIG)
        profiles = dict(new_conf.get("profiles", {}))
        profiles[name] = {
            "label": label,
            "model": model,
            "system": system,
            "options": options
        }
        new_conf["profiles"] = profiles
        new_conf = _normalize_config(new_conf)

        if _safe_write_config(new_conf):
            CONFIG = new_conf
            PROFILES = CONFIG.get("profiles", {})
            flash(f"Profilo '{name}' aggiornato.", "success")
            return redirect(url_for("profiles_list"))
        else:
            flash("Errore nell'aggiornamento del profilo (backup preservato).", "error")
            return redirect(url_for("profile_edit", name=name))

    # GET
    prof = CONFIG["profiles"][name]
    models = []
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        model_list = r.json().get("models", [])
        models = [(m.get("name") or m.get("model")) for m in model_list if (m.get("name") or m.get("model"))]
    except Exception as e:
        log_error(f"/profiles/<name>/edit fetch models error: {e}")
    if not models:
        models = [DEFAULT_MODEL]
    return render_template("profile_edit.html", pname=name, p=prof, models=models)

# ---------- PROFILES: DELETE ----------
@app.route("/profiles/<name>/delete", methods=["POST"])
def profile_delete(name):
    global CONFIG, PROFILES
    name = (name or "").strip()
    if name not in CONFIG.get("profiles", {}):
        flash("Profilo inesistente.", "error")
        return redirect(url_for("profiles_list"))

    # non permettere di cancellare l'ultimo profilo
    if len(CONFIG.get("profiles", {})) <= 1:
        flash("Impossibile eliminare: serve almeno un profilo.", "error")
        return redirect(url_for("profiles_list"))

    new_conf = dict(CONFIG)
    profiles = dict(new_conf.get("profiles", {}))
    del profiles[name]
    new_conf["profiles"] = profiles
    # se era default, riassegna
    if new_conf.get("default_profile") == name:
        new_conf["default_profile"] = "default" if "default" in profiles else next(iter(profiles), "")
    new_conf = _normalize_config(new_conf)

    if _safe_write_config(new_conf):
        CONFIG = new_conf
        PROFILES = CONFIG.get("profiles", {})
        flash(f"Profilo '{name}' eliminato.", "success")
    else:
        flash("Errore nell'eliminazione del profilo (backup preservato).", "error")
    return redirect(url_for("profiles_list"))

# ---------- COMANDI ----------
@app.route("/comandi", methods=["GET", "POST"])
def comandi_page():
    global COMANDI, CMD_PREFIX
    if request.method == "POST":
        raw = request.form.get("comandi_raw", "").strip()
        try:
            data = json.loads(raw) if raw else {}
            if _write_json_atomic(COMMANDS_PATH, data):
                COMANDI = _read_commands()
                CMD_PREFIX = COMANDI.get("prefix", "#@#")
                flash("comandi.json aggiornato correttamente.", "success")
            else:
                flash("Errore nel salvataggio di comandi.json", "error")
        except Exception as e:
            flash(f"comandi.json non valido: {e}", "error")
        return redirect(url_for("comandi_page"))
    # GET
    cmd = _read_json(COMMANDS_PATH, default=COMANDI) or COMANDI
    return render_template("comandi.html",
                           comandi_json=json.dumps(cmd, ensure_ascii=False, indent=2))

# ---------- Ricarica handler ----------
@app.route("/reload-handlers", methods=["POST"])
def reload_handlers():
    _load_handlers()
    return jsonify({"status": "ok", "loaded": [getattr(m, "__name__", "handler") for m in _LOADED_HANDLERS]})

# ---------- Avvio ----------
if __name__ == '__main__':
    log_info(f"Avvio app-ollama | OLLAMA_BASE={OLLAMA_BASE} | DEFAULT_MODEL={DEFAULT_MODEL} | DEFAULT_PROFILE={DEFAULT_PROFILE}")
    _load_handlers()
    check_ollama_connectivity(False)
    app.run(host='0.0.0.0', debug=True, port=5000)
