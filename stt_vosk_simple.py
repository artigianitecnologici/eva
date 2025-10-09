#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acquisizione microfono + STT con Vosk (senza ROS2/TTS)
- Legge config da ./config/vosk.json (autogenerata se mancante)
- Permette override da CLI
"""

import argparse
import json
import os
import queue
import sys
import zipfile
import urllib.request

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---- Default config ----
DEFAULT_CFG = {
    "lang": "it",                           # it | en (puoi aggiungerne altre tu)
    "samplerate": 16000,                    # Hz
    "audio_device": "auto",                 # "auto" | indice | parte del nome (es. "ReSpeaker")
    "show_partial": False,                  # mostra risultati parziali
    "models_dir": "./models",               # dove salvare i modelli Vosk
    "models": {
        "it": {
            "name": "vosk-model-it-0.22",
            "url": "https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip"
        },
        "en": {
            "name": "vosk-model-small-en-us-0.15",
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        }
    }
}

CONFIG_DIR = os.path.join(".", "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "vosk.json")


def ensure_config(path: str) -> dict:
    """Crea il file di config con i default se non esiste. Restituisce il dict caricato (con fallback alle nuove chiavi)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CFG, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] Creato file di configurazione: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # fallback su eventuali nuove chiavi
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
    """Scarica ed estrae un modello Vosk se non presente."""
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
    """Verifica/Scarica il modello per la lingua e ritorna il path della cartella modello."""
    if lang not in models_map:
        raise ValueError(f"Lingua '{lang}' non supportata in config. Disponibili: {', '.join(models_map.keys())}")
    model_name = models_map[lang]["name"]
    model_url  = models_map[lang]["url"]

    model_dir = os.path.join(models_dir, model_name)
    expected_file = os.path.join(model_dir, "am", "final.mdl")

    if not os.path.exists(expected_file):
        print(f"[VOSK] Modello mancante per '{lang}'. Avvio download…")
        ok = download_vosk_model(model_url, models_dir)
        if not ok:
            raise RuntimeError("Impossibile scaricare/estrarre il modello Vosk.")
        # in caso lo zip estragga con nome cartella diverso, prova ad auto-detect
        if not os.path.exists(expected_file):
            candidates = [d for d in os.listdir(models_dir)
                          if os.path.isdir(os.path.join(models_dir, d)) and d.startswith(f"vosk-model-{lang}")]
            if candidates:
                return os.path.join(models_dir, candidates[0])
            # ultimo tentativo: magari ha estratto proprio 'model_name'
            if os.path.exists(model_dir):
                return model_dir
            raise RuntimeError("Modello Vosk scaricato ma file attesi non trovati.")
    return model_dir


def list_devices_and_exit():
    print("Dispositivi audio disponibili:")
    for idx, dev in enumerate(sd.query_devices()):
        print(f"[{idx}] {dev.get('name')}  |  max input ch: {dev.get('max_input_channels')}, "
              f"default sr: {dev.get('default_samplerate')}")
    sys.exit(0)


def pick_input_device(device_arg) -> int | None:
    """
    Restituisce l'indice del dispositivo input.
    - None/'auto': prova 'ReSpeaker', altrimenti default di sistema
    - numero: usa come indice
    - stringa: match case-insensitive nel nome del device
    """
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
    # ---- CLI ----
    ap = argparse.ArgumentParser(description="Ascolto microfono + STT con Vosk (config + CLI override).")
    ap.add_argument("--config", default=CONFIG_PATH, help=f"Percorso file config JSON. Default: {CONFIG_PATH}")
    ap.add_argument("--lang", help="Override lingua (es. it, en).")
    ap.add_argument("--models-dir", help="Override cartella modelli.")
    ap.add_argument("--samplerate", type=int, help="Override sample rate (Hz).")
    ap.add_argument("--device", help="Override input: 'auto', indice, parte del nome (es. 'ReSpeaker').")
    ap.add_argument("--show-partial", action="store_true", help="Abilita stampa parziali.")
    ap.add_argument("--no-partial", action="store_true", help="Disabilita stampa parziali.")
    ap.add_argument("--list-devices", action="store_true", help="Elenca i dispositivi e termina.")
    args = ap.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    # ---- Config ----
    cfg = ensure_config(args.config)

    # Override da CLI
    if args.lang:            cfg["lang"] = args.lang
    if args.models_dir:      cfg["models_dir"] = args.models_dir
    if args.samplerate:      cfg["samplerate"] = args.samplerate
    if args.device:          cfg["audio_device"] = args.device
    if args.show_partial:    cfg["show_partial"] = True
    if args.no_partial:      cfg["show_partial"] = False

    # ---- Modello ----
    model_path = ensure_model(cfg["lang"], cfg["models_dir"], cfg["models"])
    print(f"[VOSK] Modello: {model_path}")

    # ---- Inizializza Vosk ----
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, cfg["samplerate"])

    # ---- Audio ----
    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}", flush=True)
        audio_q.put(bytes(indata))

    # Selezione dispositivo
    try:
        input_dev_idx = pick_input_device(cfg["audio_device"])
        if input_dev_idx is None:
            print("[AUDIO] Nessun dispositivo input predefinito. Usa --device o --list-devices.")
            sys.exit(1)
    except Exception as e:
        print(f"[AUDIO] Errore selezione device: {e}")
        sys.exit(1)

    sd.default.device = (input_dev_idx, None)
    print(f"[AUDIO] Uso dispositivo input index: {input_dev_idx}")
    print(f"[INFO] Ascolto… lang={cfg['lang']}  sr={cfg['samplerate']}  device={cfg['audio_device']} "
          f"(Ctrl+C per uscire)")

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
                    if text:
                        print(f">> {text}")
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
