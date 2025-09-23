#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import html
import asyncio
import tempfile
import subprocess
from typing import Dict
from io import BytesIO

import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, CallbackQueryHandler
)

# =============== Config load ===============
BASE_PATH = os.path.abspath("./")
CFG_PATH = os.path.join(BASE_PATH, "config", "telegram.json")

if not os.path.exists(CFG_PATH):
    print(f"[ERRORE] Config Telegram mancante: {CFG_PATH}", file=sys.stderr)
    sys.exit(1)

try:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        CFG = json.load(f)
except Exception as e:
    print(f"[ERRORE] Impossibile leggere/parsare {CFG_PATH}: {e}", file=sys.stderr)
    sys.exit(1)

BOT_TOKEN: str = (CFG.get("bot_token") or "").strip()
APP_BASE_URL: str = (CFG.get("app_base_url") or "http://127.0.0.1:5000/json").rstrip("/")
DEFAULT_MODEL: str = CFG.get("default_model", "gemma2:2b")
ALLOWED_CHAT_IDS = set(CFG.get("allowed_chat_ids", []))
TIMEOUT_SEC: int = int(CFG.get("timeout_sec", 60))
ASR_CFG = CFG.get("asr", {}) or {}
ASR_BACKEND = (ASR_CFG.get("backend") or "faster-whisper").lower()
TTS_CFG = CFG.get("tts", {}) or {}
TTS_ENABLED = bool(TTS_CFG.get("enabled", False))
TTS_ENGINE = (TTS_CFG.get("engine") or "piper").lower()
PIPER_CFG = TTS_CFG.get("piper", {}) or {}

# Elenco di voci per il TTS. Aggiungi qui le voci supportate dal tuo setup.
TTS_VOICES = ["Paola", "Giovanni", "Lisa"]

if not BOT_TOKEN:
    print("[ERRORE] bot_token non impostato in config/telegram.json", file=sys.stderr)
    sys.exit(1)

# =============== Stato per chat ===============
CHAT_MODEL: Dict[int, str] = {}
CHAT_VOICE: Dict[int, str] = {}

def _is_allowed(chat_id: int) -> bool:
    return (not ALLOWED_CHAT_IDS) or (chat_id in ALLOWED_CHAT_IDS)

def _get_model_for_chat(chat_id: int) -> str:
    return CHAT_MODEL.get(chat_id, DEFAULT_MODEL)

def _set_model_for_chat(chat_id: int, model: str):
    CHAT_MODEL[chat_id] = (model or "").strip()

def _get_voice_for_chat(chat_id: int) -> str:
    return CHAT_VOICE.get(chat_id, "Paola")

def _set_voice_for_chat(chat_id: int, voice: str):
    CHAT_VOICE[chat_id] = (voice or "").strip()

def chunk_text(text: str, max_len: int = 3800):
    text = text or ""
    for i in range(0, len(text), max_len):
        yield text[i:i+max_len]

def _html(msg: str) -> str:
    return html.escape(msg or "")

def sanitize_response(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace("**", "")
    return text

# =============== Client verso app-ollama.py ===============
async def query_app_ollama(session: aiohttp.ClientSession, text: str, model: str) -> str:
    url = APP_BASE_URL
    try:
        async with session.post(url, json={"query": text, "model": model}, timeout=TIMEOUT_SEC) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, dict):
                    return str(data.get("response", "")) or "(risposta vuota)"
    except Exception:
        pass

    # Fallback GET
    try:
        params = {"query": text, "model": model}
        async with session.get(url, params=params, timeout=TIMEOUT_SEC) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, dict):
                    return str(data.get("response", "")) or "(risposta vuota)"
            return f"(errore: server ha risposto {resp.status} su GET {url})"
    except Exception as e:
        return f"(errore: impossibile contattare l'app Ollama {url} — {e})"

# =============== ASR: Faster-Whisper ===============
_whisper_model = None

def _load_faster_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(f"faster-whisper non installato: {e}. Esegui: pip install faster-whisper") from e

    model_name = ASR_CFG.get("model", "small")
    compute_type = ASR_CFG.get("compute_type", "int8")
    _whisper_model = WhisperModel(model_name, compute_type=compute_type)
    return _whisper_model

def _ffmpeg_convert_to_wav16k_mono(in_path: str, out_path: str):
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg errore: {result.stderr.decode(errors='ignore')}")

async def transcribe_voice_ogg_to_text(ogg_bytes: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmpd:
        ogg_path = os.path.join(tmpd, "voice.ogg")
        wav_path = os.path.join(tmpd, "voice.wav")
        with open(ogg_path, "wb") as f:
            f.write(ogg_bytes)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _ffmpeg_convert_to_wav16k_mono, ogg_path, wav_path)

        def _do_transcribe():
            model = _load_faster_whisper_model()
            language = ASR_CFG.get("language") or None
            beam_size = ASR_CFG.get("beam_size", 5)
            segments, info = model.transcribe(wav_path, language=language, beam_size=beam_size)
            return "".join(seg.text for seg in segments).strip()

        text = await loop.run_in_executor(None, _do_transcribe)
        return text or ""

# =============== TTS: Piper ===============
def _piper_check():
    bin_path = PIPER_CFG.get("binary")
    model_path = PIPER_CFG.get("model")
    if not bin_path or not os.path.exists(bin_path):
        raise RuntimeError("Percorso 'tts.piper.binary' non valido o mancante.")
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError("Percorso 'tts.piper.model' non valido o mancante.")
    return bin_path, model_path

def _piper_tts_to_wav(text: str, wav_path: str):
    bin_path, model_path = _piper_check()
    speaker = str(PIPER_CFG.get("speaker", 0))
    length_scale = str(PIPER_CFG.get("length_scale", 1.0))
    noise_scale = str(PIPER_CFG.get("noise_scale", 0.667))
    noise_w = str(PIPER_CFG.get("noise_w", 0.8))
    sentence_silence = str(PIPER_CFG.get("sentence_silence", 0.3))

    cmd = [
        bin_path,
        "--model", model_path,
        "--output_file", wav_path,
        "--speaker", speaker,
        "--length_scale", length_scale,
        "--noise_scale", noise_scale,
        "--noise_w", noise_w,
        "--sentence_silence", sentence_silence,
    ]
    result = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"piper errore: {result.stderr.decode(errors='ignore')}")

def _wav_to_ogg_opus(wav_path: str, ogg_path: str):
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "48k", ogg_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg errore (wav->ogg): {result.stderr.decode(errors='ignore')}")

async def tts_reply_and_send_voice(update: Update, context: ContextTypes.DEFAULT_TYPE, reply_text: str):
    if not TTS_ENABLED or TTS_ENGINE != "piper":
        return
    try:
        _piper_check()
    except Exception as e:
        try:
            await update.message.reply_text(f"🔇 TTS disabilitato: {e}")
        except Exception:
            pass
        return

    max_chars = int(PIPER_CFG.get("max_chars", 600))
    tts_text = (reply_text or "").strip()
    if len(tts_text) > max_chars:
        tts_text = tts_text[:max_chars] + "…"

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)

    with tempfile.TemporaryDirectory() as tmpd:
        wav_path = os.path.join(tmpd, "out.wav")
        ogg_path = os.path.join(tmpd, "out.ogg")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _piper_tts_to_wav, tts_text, wav_path)
            await loop.run_in_executor(None, _wav_to_ogg_opus, wav_path, ogg_path)
        except Exception as e:
            try:
                await update.message.reply_text(f"🔇 Errore TTS: {e}")
            except Exception:
                pass
            return

        try:
            with open(ogg_path, "rb") as f:
                caption = None
                if len(reply_text) <= 120:
                    caption = reply_text
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=f, caption=caption)
        except Exception as e:
            try:
                await update.message.reply_text(f"🔇 Errore invio voce: {e}")
            except Exception:
                pass

# =============== Bot Handlers ===============
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not _is_allowed(update.effective_chat.id):
        return
    model = _get_model_for_chat(update.effective_chat.id)
    voice = _get_voice_for_chat(update.effective_chat.id)
    tts_status = "attivo" if TTS_ENABLED else "spento"
    body = (
        "👋 Ciao! Questo bot inoltra i tuoi messaggi a <b>app-ollama.py</b> e ti restituisce la risposta.<br><br>"
        f"• Endpoint: <code>{_html(APP_BASE_URL)}</code><br>"
        f"• Modello attuale: <code>{_html(model)}</code><br>"
        f"• Voce TTS: <code>{_html(voice)}</code><br>"
        f"• TTS: <b>{_html(tts_status)}</b><br><br>"
        "<b>Comandi</b><br>"
        "• <code>/model &lt;nome_modello&gt;</code><br>"
        "• <code>/voice &lt;nome_voce&gt;</code><br>"
        "• <code>/health</code><br>"
        "• invia <b>messaggi vocali</b> per trascrizione e risposta"
    )
    try:
        await update.message.reply_text(body, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except BadRequest:
        await update.message.reply_text(f"Endpoint: {APP_BASE_URL}\nModello: {model}\nVoce: {voice}\nTTS: {tts_status}")

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not _is_allowed(update.effective_chat.id):
        return
    if not context.args:
        current = _get_model_for_chat(update.effective_chat.id)
        await update.message.reply_text(
            f"Modello attuale: <code>{_html(current)}</code>\nUsa: <code>/model &lt;nome&gt;</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    model = " ".join(context.args).strip()
    _set_model_for_chat(update.effective_chat.id, model)
    await update.message.reply_text(
        f"✅ Modello impostato su: <code>{_html(model)}</code>",
        parse_mode=ParseMode.HTML,
    )

async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not _is_allowed(update.effective_chat.id):
        return
    if not context.args:
        current_voice = _get_voice_for_chat(update.effective_chat.id)
        await update.message.reply_text(
            f"Voce TTS attuale: <code>{_html(current_voice)}</code>\nUsa: <code>/voice &lt;nome_voce&gt;</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    voice = " ".join(context.args).strip()
    if voice not in TTS_VOICES:
        await update.message.reply_text(f"⚠️ Voce non valida. Scegli tra: {', '.join(TTS_VOICES)}.")
        return
    _set_voice_for_chat(update.effective_chat.id, voice)
    await update.message.reply_text(
        f"✅ Voce TTS impostata su: <code>{_html(voice)}</code>",
        parse_mode=ParseMode.HTML,
    )

async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not _is_allowed(update.effective_chat.id):
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    health_url = APP_BASE_URL.rsplit("/", 1)[0] + "/healthz"
    try:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(health_url, timeout=10) as resp:
                    if resp.status == 200:
                        await update.message.reply_text("💚 app-ollama è raggiungibile.")
                        return
            except Exception:
                pass
            ans = await query_app_ollama(session, "ping", _get_model_for_chat(update.effective_chat.id))
    except Exception as e:
        ans = f"(errore: {e})"
    if ans.startswith("(errore"):
        await update.message.reply_text(f"💥 app-ollama non raggiungibile:\n{ans}")
    else:
        await update.message.reply_text("💚 app-ollama risponde correttamente.")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not _is_allowed(update.effective_chat.id):
        return
    text_in = (update.message.text or "").strip()
    if not text_in:
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    model = _get_model_for_chat(update.effective_chat.id)
    async with aiohttp.ClientSession() as session:
        reply = await query_app_ollama(session, text_in, model)
    reply = sanitize_response(reply)
    for chunk in chunk_text(reply):
        try:
            await update.message.reply_text(chunk)
        except TelegramError:
            await update.message.reply_text(chunk)
    await tts_reply_and_send_voice(update, context, reply)

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce i messaggi vocali: li trascrive, interroga l'app Ollama e risponde."""
    if not update.message or not update.message.voice or not _is_allowed(update.effective_chat.id):
        return

    # Inizializza la variabile 'reply' per evitare l'errore UnboundLocalError
    reply = ""

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        # Scarica il file audio del messaggio vocale
        ogg_file = await update.message.voice.get_file()
        ogg_buffer = BytesIO()
        await ogg_file.download_to_memory(ogg_buffer)
        ogg_bytes = ogg_buffer.getvalue()

        # Trascrivi l'audio in testo
        text_in = await transcribe_voice_ogg_to_text(ogg_bytes)

        if not text_in:
            await update.message.reply_text("Non sono riuscito a trascrivere il messaggio vocale.")
            return

        # Interroga l'app Ollama con il testo trascritto
        model = _get_model_for_chat(update.effective_chat.id)
        async with aiohttp.ClientSession() as session:
            reply = await query_app_ollama(session, text_in, model)
        
        reply = sanitize_response(reply)

        # Rispondi all'utente con il testo
        for chunk in chunk_text(reply):
            try:
                await update.message.reply_text(chunk)
            except TelegramError:
                await update.message.reply_text(chunk)

    except Exception as e:
        print(f"[ERRORE] Errore durante la gestione del messaggio vocale: {e}", file=sys.stderr)
        await update.message.reply_text("Si è verificato un errore durante l'elaborazione del messaggio vocale.")
        reply = f"Si è verificato un errore: {e}"

    # Invia la risposta anche come voce, se il TTS è abilitato
    await tts_reply_and_send_voice(update, context, reply)


# =============== Main ===============
def main():
    print(f"[INFO] Avvio Telegram bridge v.1.00 → {APP_BASE_URL} | default model: {DEFAULT_MODEL}", file=sys.stderr)
    if ALLOWED_CHAT_IDS:
        print(f"[INFO] Chat autorizzate: {sorted(ALLOWED_CHAT_IDS)}", file=sys.stderr)
    if TTS_ENABLED:
        print("[INFO] TTS Piper abilitato", file=sys.stderr)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)

if __name__ == "__main__":
    main()
