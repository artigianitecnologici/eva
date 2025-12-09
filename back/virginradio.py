# -*- coding: utf-8 -*-
"""
Handler 'radio_virgin' per controllare lo streaming di Virgin Radio:
 - accendi / avvia / metti la radio  -> avvia lo streaming (senza risposta vocale)
 - spegni / ferma la radio           -> ferma lo streaming (senza risposta vocale)
 - stato radio                       -> risposta parlata/testuale

Richiede:
    sudo apt install mpg123

Esempi di comandi riconosciuti:
    "accendi la radio"
    "eva accendi la radio"
    "metti virgin radio"
    "spegni la radio"
    "ferma la radio"
    "stato radio"
"""

import re
import subprocess
import threading

# URL streaming Virgin Radio
VIRGIN_URL = "http://icecast.unitedradio.it/Virgin.mp3"

# Prefisso opzionale (#@#) e "eva " come nel handler sistema
RX_PREFIX = r"^\s*(?:#@#\s*)?"

# Comandi ON
RX_ON = re.compile(
    RX_PREFIX +
    r"(?:eva\s+)?(?:(?:accendi|avvia|metti|met|riproduci|play)\s+(?:la\s+)?radio|"
    r"(?:metti|riproduci|play)\s+virgin(?:\s+radio)?)\s*$",
    re.I
)

# Comandi OFF
RX_OFF = re.compile(
    RX_PREFIX +
    r"(?:eva\s+)?(?:(?:spegni|ferma|stop|chiudi)\s+(?:la\s+)?radio)\s*$",
    re.I
)

# Stato radio
RX_STATUS = re.compile(
    RX_PREFIX +
    r"(?:stato\s+radio|radio\s+on\??|radio\s+attiva\??)\s*$",
    re.I
)

# Stato globale del processo radio
_radio_proc = None
_radio_lock = threading.Lock()


def _start_radio() -> bool:
    """Avvia mpg123 in un thread separato. Ritorna True se ha avviato, False se era gia attiva."""
    global _radio_proc
    with _radio_lock:
        # Se c'  gi  un processo vivo, non fare nulla
        if _radio_proc is not None and _radio_proc.poll() is None:
            return False

        def run():
            global _radio_proc
            try:
                _radio_proc = subprocess.Popen(
                    ["mpg123", "-q", VIRGIN_URL],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                _radio_proc.wait()
            except Exception:
                pass
            finally:
                with _radio_lock:
                    _radio_proc = None

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return True


def _stop_radio() -> bool:
    """Termina il processo mpg123 se attivo. Ritorna True se l'ha fermato, False se era gi  spenta."""
    global _radio_proc
    with _radio_lock:
        proc = _radio_proc
        if proc is None or proc.poll() is not None:
            _radio_proc = None
            return False
        try:
            proc.terminate()
        except Exception:
            pass
        _radio_proc = None
        return True


def _is_radio_on() -> bool:
    with _radio_lock:
        return _radio_proc is not None and _radio_proc.poll() is None


def can_handle(text: str, context: dict) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(RX_ON.search(t) or RX_OFF.search(t) or RX_STATUS.search(t))


def handle(text: str, context: dict) -> str:
    t = (text or "").strip()

    # Accendi / avvia radio
    if RX_ON.search(t):
        started = _start_radio()
        if started:
            # NON restituiamo testo: cos  stt_vosk non chiama TTS
            # (l'utente "capisce" perch  parte la radio)
            return ""
        else:
            # Radio gi  attiva: qui puoi decidere se parlare o no
            # Io metto una breve risposta (TTS), ma puoi anche restituire "".
            return "La radio   gi  in riproduzione."

    # Spegni / ferma radio
    if RX_OFF.search(t):
        stopped = _stop_radio()
        if stopped:
            # Anche qui, nessuna risposta vocale per evitare conflitti,
            # ma puoi cambiare in una frase parlata se vuoi.
            return ""
        else:
            return "La radio risulta gi  spenta."

    # Stato radio: qui invece ha senso che parli
    if RX_STATUS.search(t):
        if _is_radio_on():
            return "La radio   attiva e sta riproducendo Virgin Radio."
        else:
            return "La radio   spenta."

    # Fallback (in teoria non dovrebbe arrivarci)
    return ""
