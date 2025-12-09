#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stt_vosk.py
-----------
Acquisizione microfono + STT con Vosk + wake word + TTS Piper (senza ROS2).

Dipendenze di sistema (Raspberry Pi / Debian-like):
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv python3-dev
    sudo apt install -y libportaudio2 portaudio19-dev libportaudiocpp0
    sudo apt install -y libasound2 libasound2-dev
    sudo apt install -y sox piper

Dipendenze Python (dentro una venv consigliata):
    pip install sounddevice vosk numpy
"""

import argparse
import json
import os
import queue
import sys
import unicodedata
import zipfile
import urllib.request
import subprocess

import sounddevice as sd
from vosk import Model, KaldiRecognizer

DEFAULT_CFG = {
    "lang": "it",
    "samplerate": 16000,
    "audio_device": "auto",
    "show_partial": False,
    "models_dir": "./models",
    "models": {
        "it": {
            "name": "vosk-model-it-0.22",
            "url": "https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip"
        },
        "en": {
            "name": "vosk-model-small-en-us-0.15",
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        }
    },
    "wake": {
        "enabled": True,
        "wake_words": ["martino"],
        "match": "anywhere",
        "case_sensitive": False,
        "normalize_accents": True
    },
    "beep": {
        "enabled": True,
        "frequency": 1000,
        "duration_ms": 180,
        "volume": 0.2
    },
    "tts": {
        "enabled": True,
        "models_dir": "./piper_models",
        "default_voice": "paola",
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "ack_text": "Ho capito, un attimo e ti rispondo."
    },
    "nlp": {
        "corrections": True,
        "commands_vocab_file": "commands_vocab.it.json"
    }
}

CONFIG_DIR = os.path.join(".", "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "vosk.json")


def ensure_config(path: str) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CFG, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] Creato file di configurazione: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except UnicodeDecodeError:
        # Se il file e in una codifica strana, lo rinominiamo e lo ricreiamo
        backup = path + ".bak"
        os.rename(path, backup)
        print(f"[CONFIG] File non UTF-8, spostato in {backup}. Ricreo config di default.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CFG, f, indent=2, ensure_ascii=False)
        cfg = DEFAULT_CFG.copy()

    changed = False

    def merge_defaults(dst, src):
        nonlocal changed
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
                changed = True
            else:
                if isinstance(v, dict) and isinstance(dst[k], dict):
                    merge_defaults(dst[k], v)

    merge_defaults(cfg, DEFAULT_CFG)

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("[CONFIG] Aggiornata config con chiavi mancanti.")

    return cfg


def download_vosk_model(model_url: str, models_root: str) -> bool:
    os.makedirs(models_root, exist_ok=True)
    zip_path = os.path.join(models_root, "vosk_model.zip")
    print(f"[VOSK] Scarico modello da: {model_url}")
    try:
        urllib.request.urlretrieve(model_url, zip_path)
    except Exception as e:
        print(f"[VOSK] Errore download: {e}")
        return False

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(models_root)
        os.remove(zip_path)
        print("[VOSK] Modello scaricato ed estratto.")
        return True
    except Exception as e:
        print(f"[VOSK] Errore estrazione: {e}")
        return False


def ensure_model(lang: str, models_dir: str, models_map: dict) -> str:
    if lang not in models_map:
        raise ValueError(f"Lingua '{lang}' non supportata. Disponibili: {', '.join(models_map.keys())}")
    model_name = models_map[lang]["name"]
    model_url = models_map[lang]["url"]

    model_dir = os.path.join(models_dir, model_name)
    expected_file = os.path.join(model_dir, "am", "final.mdl")

    if not os.path.exists(expected_file):
        print(f"[VOSK] Modello mancante per '{lang}'. Avvio download...")
        ok = download_vosk_model(model_url, models_dir)
        if not ok:
            raise RuntimeError("Impossibile scaricare o estrarre il modello Vosk.")
        if not os.path.exists(expected_file):
            candidates = [
                d for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d)) and d.startswith(f"vosk-model-{lang}")
            ]
            if candidates:
                return os.path.join(models_dir, candidates[0])
            if os.path.exists(model_dir):
                return model_dir
            raise RuntimeError("Modello Vosk scaricato ma file attesi non trovati.")
    return model_dir


class PiperTTS:
    SUPPORTED_VOICES = {
        "paola": "it_IT-paola-medium.onnx",
        "riccardo": "it_IT-riccardo-medium.onnx",
    }

    def __init__(
        self,
        models_dir: str,
        default_voice: str = "paola",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.models_dir = models_dir
        self.voice = default_voice if default_voice in self.SUPPORTED_VOICES else "paola"
        self.length_scale = str(length_scale)
        self.noise_scale = str(noise_scale)
        self.noise_w = str(noise_w)

        if subprocess.call(['which', 'piper'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            raise RuntimeError("piper non e installato. sudo apt-get install -y piper")
        if subprocess.call(['which', 'play'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            raise RuntimeError("SoX non e installato. sudo apt-get install -y sox")

    def _model_path(self) -> str:
        return os.path.join(self.models_dir, self.SUPPORTED_VOICES[self.voice])

    def get_params(self):
        return {
            "voice": self.voice,
            "model_path": self._model_path(),
            "length_scale": self.length_scale,
            "noise_scale": self.noise_scale,
            "noise_w": self.noise_w
        }

    def speak(self, text: str, wav_path: str = "/tmp/robot_speech.wav"):
        if not text:
            return
        model = self._model_path()
        if not os.path.exists(model):
            raise FileNotFoundError(f"Modello piper non trovato: {model}")
        subprocess.run(
            [
                "piper",
                "--model", model,
                "--output_file", wav_path,
                "--length_scale", self.length_scale,
                "--noise_scale", self.noise_scale,
                "--noise_w", self.noise_w,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run([
            "play", wav_path, "--norm", "-q",
            "pitch", "400",
            "tempo", "0.9",
            "treble", "+3",
            "highpass", "120"
        ], check=True)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    v0 = list(range(len(b) + 1))
    v1 = [0] * (len(b) + 1)
    for i in range(len(a)):
        v1[0] = i + 1
        for j in range(len(b)):
            cost = 0 if a[i] == b[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0
    return v0[len(b)]


def norm_basic(text: str, normalize_accents: bool = True) -> str:
    t = text.strip().lower()
    if normalize_accents:
        t = unicodedata.normalize('NFD', t)
        t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
    import re
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_commands_vocab(path: str):
    try:
        if not os.path.exists(path):
            print(f"[NLP] File vocabolario comandi non trovato: {path}")
            return []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[NLP] Formato non valido in {path}: attesa una lista di oggetti")
            return []
        cleaned = []
        for it in data:
            if isinstance(it, dict) and "phrase" in it:
                cleaned.append({
                    "phrase": str(it["phrase"]),
                    "intent": str(it.get("intent", it["phrase"]))
                })
        print(f"[NLP] Vocabolario caricato: {len(cleaned)} frasi.")
        return cleaned
    except Exception as e:
        print(f"[NLP] Impossibile caricare vocabolario {path}: {e}")
        return []


def snap_to_vocab(text: str, vocab: list) -> str:
    if not vocab:
        return text
    best = None
    best_score = -1.0
    for item in vocab:
        phrase = item.get("phrase", "")
        intent = item.get("intent", phrase)
        if not phrase:
            continue
        dist = levenshtein(text, phrase)
        maxlen = max(len(text), len(phrase), 1)
        score = 1.0 - (dist / maxlen)
        if score > best_score:
            best_score = score
            best = intent
    return best if best_score >= 0.72 else text


def postprocess_text(text: str, cfg_nlp: dict, vocab: list, normalize_accents: bool = True) -> str:
    t = norm_basic(text, normalize_accents=normalize_accents)
    if cfg_nlp.get("corrections", True):
        t = snap_to_vocab(t, vocab)
    return t


def check_wake(raw_text: str, wake_cfg: dict):
    if not wake_cfg.get("enabled", True):
        return False, "", ""

    wake_words = wake_cfg.get("wake_words") or []
    if not wake_words:
        single = wake_cfg.get("wake_word", "martino")
        wake_words = [single]

    case_sensitive = wake_cfg.get("case_sensitive", False)
    normalize_accents = wake_cfg.get("normalize_accents", True)
    match_mode = wake_cfg.get("match", "anywhere").lower()

    text_for_match = raw_text if case_sensitive else raw_text.lower()
    if normalize_accents:
        text_for_match = unicodedata.normalize('NFD', text_for_match)
        text_for_match = ''.join(ch for ch in text_for_match if unicodedata.category(ch) != 'Mn')

    for kw in wake_words:
        kw_match = kw if case_sensitive else kw.lower()
        if normalize_accents:
            kw_match = unicodedata.normalize('NFD', kw_match)
            kw_match = ''.join(ch for ch in kw_match if unicodedata.category(ch) != 'Mn')

        if match_mode == "prefix":
            if text_for_match.startswith(kw_match):
                remainder = raw_text[len(kw):].lstrip()
                return True, kw, remainder
        else:
            idx = text_for_match.find(kw_match)
            if idx != -1:
                start = idx + len(kw)
                remainder = raw_text[start:].lstrip()
                return True, kw, remainder

    return False, "", ""


def do_beep(beep_cfg: dict):
    if not beep_cfg.get("enabled", True):
        return
    try:
        dur = max(0.05, min(1.0, beep_cfg.get("duration_ms", 180) / 1000.0))
        vol = str(max(0.0, min(1.0, float(beep_cfg.get("volume", 0.2)))))
        freq = str(int(beep_cfg.get("frequency", 1000)))
        subprocess.Popen(
            ["play", "-nq", "synth", str(dur), "sin", freq, "vol", vol],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"[BEEP] Beep fallito: {e}")


def list_devices_and_exit():
    print("Dispositivi audio disponibili:")
    for idx, dev in enumerate(sd.query_devices()):
        print(f"[{idx}] {dev.get('name')}  |  max input ch: {dev.get('max_input_channels')}, "
              f"default sr: {dev.get('default_samplerate')}")
    sys.exit(0)


def pick_input_device(device_arg) -> int | None:
    devices = sd.query_devices()
    if device_arg is None or str(device_arg).lower() == "auto":
        for idx, dev in enumerate(devices):
            if "respeaker" in str(dev.get("name", "")).lower():
                return idx
        return sd.default.device[0] if sd.default.device else None

    try:
        return int(device_arg)
    except (TypeError, ValueError):
        pass

    needle = str(device_arg).lower()
    for idx, dev in enumerate(devices):
        if needle in str(dev.get("name", "")).lower():
            return idx

    raise ValueError(f"Dispositivo non trovato per chiave: {device_arg}")


def main():
    ap = argparse.ArgumentParser(description="Ascolto microfono + STT con Vosk + wake-word + Piper TTS.")
    ap.add_argument("--config", default=CONFIG_PATH, help=f"Percorso file config JSON. Default: {CONFIG_PATH}")
    ap.add_argument("--lang", help="Override lingua (es. it, en).")
    ap.add_argument("--models-dir", help="Override cartella modelli Vosk.")
    ap.add_argument("--samplerate", type=int, help="Override sample rate (Hz).")
    ap.add_argument("--device", help="Override input: 'auto', indice, parte del nome.")
    ap.add_argument("--show-partial", action="store_true", help="Abilita stampa parziali.")
    ap.add_argument("--no-partial", action="store_true", help="Disabilita stampa parziali.")
    ap.add_argument("--list-devices", action="store_true", help="Elenca i dispositivi e termina.")
    ap.add_argument("--no-wake", action="store_true", help="Disabilita wake-word.")
    ap.add_argument("--no-tts", action="store_true", help="Disabilita Piper TTS.")
    args = ap.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    cfg = ensure_config(args.config)

    if args.lang:
        cfg["lang"] = args.lang
    if args.models_dir:
        cfg["models_dir"] = args.models_dir
    if args.samplerate:
        cfg["samplerate"] = args.samplerate
    if args.device:
        cfg["audio_device"] = args.device
    if args.show_partial:
        cfg["show_partial"] = True
    if args.no_partial:
        cfg["show_partial"] = False
    if args.no_wake:
        cfg.setdefault("wake", {})["enabled"] = False
    if args.no_tts:
        cfg.setdefault("tts", {})["enabled"] = False

    model_path = ensure_model(cfg["lang"], cfg["models_dir"], cfg["models"])
    print(f"[VOSK] Modello: {model_path}")

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, cfg["samplerate"])

    wake_cfg = cfg.get("wake", {})
    beep_cfg = cfg.get("beep", {})
    nlp_cfg = cfg.get("nlp", {})
    tts_cfg = cfg.get("tts", {})

    vocab = []
    vocab_file = nlp_cfg.get("commands_vocab_file")
    if vocab_file:
        if not os.path.isabs(vocab_file):
            vocab_path = os.path.join(CONFIG_DIR, vocab_file)
        else:
            vocab_path = vocab_file
        vocab = load_commands_vocab(vocab_path)

    piper = None
    if tts_cfg.get("enabled", True):
        try:
            piper = PiperTTS(
                models_dir=tts_cfg.get("models_dir", "./piper_models"),
                default_voice=tts_cfg.get("default_voice", "paola"),
                length_scale=tts_cfg.get("length_scale", 1.0),
                noise_scale=tts_cfg.get("noise_scale", 0.667),
                noise_w=tts_cfg.get("noise_w", 0.8),
            )
            params = piper.get_params()
            print(f"[PIPER] TTS attivo. Voce={params['voice']} Modello={params['model_path']}")
        except Exception as e:
            print(f"[PIPER] TTS disabilitato: {e}")
            piper = None

    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}", flush=True)
        audio_q.put(bytes(indata))

    try:
        input_dev_idx = pick_input_device(cfg.get("audio_device"))
        if input_dev_idx is None:
            print("[AUDIO] Nessun dispositivo input predefinito. Usa --device o --list-devices.")
            sys.exit(1)
    except Exception as e:
        print(f"[AUDIO] Errore selezione device: {e}")
        sys.exit(1)

    sd.default.device = (input_dev_idx, None)
    print(f"[AUDIO] Uso dispositivo input index: {input_dev_idx}")
    print(f"[INFO] Ascolto... lang={cfg['lang']}  sr={cfg['samplerate']}  device={cfg['audio_device']} (Ctrl+C per uscire)")

    last_partial = ""
    try:
        with sd.RawInputStream(
            samplerate=cfg["samplerate"],
            dtype="int16",
            channels=1,
            callback=audio_callback,
            device=input_dev_idx
        ):
            while True:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    res = json.loads(recognizer.Result())
                    text = (res.get("text") or "").strip()
                    if not text:
                        last_partial = ""
                        continue

                    print(f"[ASR] Hai detto: {text}")

                    if wake_cfg.get("enabled", True):
                        triggered, trig_word, remainder = check_wake(text, wake_cfg)
                        if not triggered:
                            continue

                        print(f"[WAKE] Rilevata wake-word: '{trig_word}'")
                        do_beep(beep_cfg)

                        if remainder:
                            corrected = postprocess_text(
                                remainder,
                                cfg_nlp=nlp_cfg,
                                vocab=vocab,
                                normalize_accents=wake_cfg.get("normalize_accents", True)
                            )
                            print(f"[CMD] Comando: {corrected!r}")
                            if piper is not None:
                                ack = tts_cfg.get("ack_text") or ""
                                if ack:
                                    try:
                                        piper.speak(ack)
                                    except Exception as e:
                                        print(f"[PIPER] Errore speak: {e}")
                    else:
                        do_beep(beep_cfg)
                        processed = postprocess_text(
                            text,
                            cfg_nlp=nlp_cfg,
                            vocab=vocab,
                            normalize_accents=wake_cfg.get("normalize_accents", True)
                        )
                        print(f"[TEXT] {processed}")

                    last_partial = ""
                else:
                    if cfg.get("show_partial", False):
                        pres = json.loads(recognizer.PartialResult())
                        ptext = (pres.get("partial") or "").strip()
                        if ptext and ptext != last_partial:
                            print(f".. {ptext}", end="\r", flush=True)
                            last_partial = ptext

    except KeyboardInterrupt:
        print("\n[INFO] Uscita richiesta dall'utente.")
    except Exception as e:
        print(f"[ERRORE] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
