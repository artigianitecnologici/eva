#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microfono → STT con faster-whisper.
- Config da file: ./config/whisper.json (autogenerata se mancante)
- Modello di default: 'small'
- Download automatico nella cache locale (download_root)
- Scelta compute-device da CLI: --compute-device cpu|auto|cuda
"""

import argparse
import json
import os
import queue
import sys
import time
from typing import Any, Dict

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --------- Default config ----------
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "small",                 # tiny | base | small | medium | large-v2 | large-v3
    "lang": "it",                     # lingua forzata (es. it, en)
    "audio_device": "auto",           # "auto" | indice (int) | parte del nome (str, es. "ReSpeaker")
    "samplerate": 16000,              # Hz
    "window_sec": 5.0,                # dimensione blocco trascrizione
    "use_vad": False,                 # filtra silenzi con VAD interno
    "compute_type": "int8",           # int8 | int8_float16 | float16 | float32
    "compute_device": "cpu",          # cpu | auto | cuda
    "download_dir": "./models_whisper" # cartella per i modelli scaricati
}

CONFIG_DIR = os.path.join(".", "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "whisper.json")


# ---------- Helpers ----------
def ensure_config(path: str) -> Dict[str, Any]:
    """Crea una config di default se non esiste, poi la carica."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] File di configurazione creato con i default: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # fallback su eventuali nuove chiavi
    changed = False
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("[CONFIG] Aggiornata config con chiavi mancanti.")

    return cfg


def list_devices_and_exit():
    print("Dispositivi audio disponibili:")
    for idx, dev in enumerate(sd.query_devices()):
        print(f"[{idx}] {dev.get('name')}  |  max input ch: {dev.get('max_input_channels')}, "
              f"default sr: {dev.get('default_samplerate')}")
    sys.exit(0)


def pick_input_device(device_arg) -> int | None:
    """
    Restituisce indice del dispositivo input.
    - None/'auto': prova 'ReSpeaker', altrimenti default di sistema
    - numero (int o str numerica): usa come indice
    - stringa: match case-insensitive nel nome device
    """
    devices = sd.query_devices()

    if device_arg is None or str(device_arg).lower() == "auto":
        for idx, dev in enumerate(devices):
            if "respeaker" in str(dev.get("name", "")).lower():
                return idx
        return sd.default.device[0] if sd.default.device else None

    # indice
    try:
        return int(device_arg)
    except (TypeError, ValueError):
        pass

    # match per nome
    needle = str(device_arg).lower()
    for idx, dev in enumerate(devices):
        if needle in str(dev.get("name", "")).lower():
            return idx

    raise ValueError(f"Dispositivo non trovato per chiave: {device_arg}")


def ensure_model_ready(model_name: str, download_dir: str, compute_type: str, compute_device: str):
    """Istanzia WhisperModel una volta per forzare il download in download_root, se manca."""
    os.makedirs(download_dir, exist_ok=True)
    print(f"[MODEL] Verifica modello '{model_name}' (compute_device={compute_device}, compute_type={compute_type})…")
    _ = WhisperModel(model_name, device=compute_device, compute_type=compute_type, download_root=download_dir)
    print("[MODEL] OK: modello pronto.")


def main():
    # --------- CLI ----------
    ap = argparse.ArgumentParser(description="Microfono → STT con faster-whisper (config + CLI).")
    ap.add_argument("--config", default=CONFIG_PATH, help=f"Percorso file config JSON. Default: {CONFIG_PATH}")
    ap.add_argument("--model", help="Override modello (tiny/base/small/medium/large-v2/large-v3).")
    ap.add_argument("--lang", help="Override lingua (es. it, en).")
    ap.add_argument("--device", help="Override input audio: 'auto', indice, o parte del nome (es. 'ReSpeaker').")
    ap.add_argument("--samplerate", type=int, help="Override sample rate microfono (Hz).")
    ap.add_argument("--window-sec", type=float, help="Override durata finestra (s).")
    ap.add_argument("--use-vad", action="store_true", help="Abilita VAD (filtra silenzi).")
    ap.add_argument("--no-vad", action="store_true", help="Disabilita VAD.")
    ap.add_argument("--compute-type", help="Override precisione: int8|int8_float16|float16|float32.")
    ap.add_argument("--compute-device", choices=["cpu", "auto", "cuda"],
                    help="Override device di calcolo per l'inferenza.")
    ap.add_argument("--download-dir", help="Override cartella locale per i modelli scaricati.")
    ap.add_argument("--list-devices", action="store_true", help="Elenca i dispositivi audio e termina.")
    args = ap.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    # --------- Config ----------
    cfg = ensure_config(args.config)

    # CLI override → config
    if args.model:          cfg["model"] = args.model
    if args.lang:           cfg["lang"] = args.lang
    if args.device:         cfg["audio_device"] = args.device
    if args.samplerate:     cfg["samplerate"] = args.samplerate
    if args.window_sec:     cfg["window_sec"] = args.window_sec
    if args.compute_type:   cfg["compute_type"] = args.compute_type
    if args.compute_device: cfg["compute_device"] = args.compute_device
    if args.download_dir:   cfg["download_dir"] = args.download_dir
    if args.use_vad:        cfg["use_vad"] = True
    if args.no_vad:         cfg["use_vad"] = False

    # --------- Modello (download se assente) ----------
    try:
        ensure_model_ready(cfg["model"], cfg["download_dir"], cfg["compute_type"], cfg["compute_device"])
    except Exception as e:
        print(f"[MODEL] Errore durante la preparazione del modello: {e}")
        # piano B: se fallisce con CUDA/cuDNN, suggerisci CPU
        if cfg.get("compute_device") != "cpu":
            print("[MODEL] Riprova forzando la CPU: --compute-device cpu  (o setta 'compute_device':'cpu' nel JSON)")
        sys.exit(1)

    # --------- Dispositivo audio ----------
    try:
        input_dev_idx = pick_input_device(cfg["audio_device"])
        if input_dev_idx is None:
            print("[AUDIO] Nessun dispositivo input predefinito. Usa --device o --list-devices.")
            sys.exit(1)
    except Exception as e:
        print(f"[AUDIO] Errore selezione device: {e}")
        sys.exit(1)

    # --------- Inizializza modello reale ----------
    model = WhisperModel(
        cfg["model"],
        device=cfg["compute_device"],
        compute_type=cfg["compute_type"],
        download_root=cfg["download_dir"]
    )

    # --------- Audio loop ----------
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}", flush=True)
        audio_q.put(indata.copy())

    sd.default.device = (input_dev_idx, None)
    print(f"[AUDIO] Uso input index: {input_dev_idx}")
    print(f"[INFO] Ascolto… lang={cfg['lang']}  model={cfg['model']}  device={cfg['compute_device']}  "
          f"ctype={cfg['compute_type']}  (Ctrl+C per uscire)")

    target_samples = int(cfg["samplerate"] * float(cfg["window_sec"]))
    ring = np.zeros((0, 1), dtype=np.int16)
    last_print_ts = 0.0

    try:
        with sd.InputStream(
            samplerate=cfg["samplerate"],
            dtype="int16",
            channels=1,
            callback=audio_callback,
            device=input_dev_idx,
            blocksize=0
        ):
            while True:
                chunk = audio_q.get()
                ring = np.concatenate([ring, chunk], axis=0)

                if ring.shape[0] >= target_samples:
                    window = ring[:target_samples]
                    ring = ring[target_samples:]

                    audio_float32 = (window.astype(np.float32) / 32768.0).flatten()

                    segments, _ = model.transcribe(
                        audio=audio_float32,
                        language=cfg["lang"],
                        vad_filter=bool(cfg["use_vad"]),
                        vad_parameters=dict(min_silence_duration_ms=300),
                        beam_size=1,
                        best_of=1,
                        condition_on_previous_text=False,
                        temperature=0.0,
                    )

                    text = "".join(seg.text for seg in segments).strip()
                    now = time.time()
                    if text and (now - last_print_ts > 0.5):
                        print(f">> {text}")
                        last_print_ts = now

    except KeyboardInterrupt:
        print("\n[INFO] Uscita richiesta dall'utente.")
    except Exception as e:
        print(f"[ERRORE] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
