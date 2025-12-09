# -*- coding: utf-8 -*-
"""
Handler 'radio_virgin' per controllare lo streaming di Virgin Radio:
 - accendi / avvia / metti la radio  -> avvia lo streaming (senza risposta vocale)
 - spegni / ferma la radio           -> ferma lo streaming (senza risposta vocale)
 - stato radio                       -> risposta parlata/testuale

Richiede:
    sudo apt install mpg123

Esempi di comandi riconosciuti per ON:
    "virgin radio"
    "avvia virgin radio"
    "metti virgin radio"
    "eva metti virgin radio"
    "riproduci virgin radio"

Comandi OFF:
    "spegni la radio"
    "ferma la radio"
    "eva spegni la radio"

Stato:
    "stato radio"
    "radio on?"
    "radio attiva?"
"""

import re
import subprocess
import threading

print("[radio_virgin] modulo importato", flush=True)

# URL streaming Virgin Radio
VIRGIN_URL = "http://icecast.unitedradio.it/Virgin.mp3"

# Prefisso opzionale (#@#) e "eva " come nel handler sistema
RX_PREFIX = r"^\s*(?:#@#\s*)?"

# Comandi ON: qualsiasi frase che contenga "virgin radio"
RX_ON = re.compile(
    RX_PREFIX +
    r"(?:eva\s+)?(?:.*\bvirgin\b\s+\bradio\b.*)$",
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
    print("[radio_virgin] _start_radio chiamata", flush=True)
    with _radio_lock:
        # Se c'e gia un processo vivo, non fare nulla
        if _radio_proc is not None and _radio_proc.poll() is None:
            print("[radio_virgin] _start_radio: radio gia attiva (processo vivo)", flush=True)
            return False

        def run():
            global _radio_proc
            print("[radio_virgin] thread radio: avvio mpg123...", flush=True)
            try:
                _radio_proc = subprocess.Popen(
                    ["mpg123", "-q", VIRGIN_URL],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("[radio_virgin] thread radio: processo mpg123 avviato con pid =", _radio_proc.pid, flush=True)
                _radio_proc.wait()
                print("[radio_virgin] thread radio: mpg123 terminato", flush=True)
            except Exception as e:
                print("[radio_virgin] thread radio: eccezione in esecuzione mpg123:", repr(e), flush=True)
            finally:
                with _radio_lock:
                    print("[radio_virgin] thread radio: azzero _radio_proc", flush=True)
                    _radio_proc = None

        t = threading.Thread(target=run, daemon=True)
        t.start()
        print("[radio_virgin] _start_radio: thread radio avviato", flush=True)
        return True


def _stop_radio() -> bool:
    """Termina il processo mpg123 se attivo. Ritorna True se l'ha fermato, False se era gia spenta."""
    global _radio_proc
    print("[radio_virgin] _stop_radio chiamata", flush=True)
    with _radio_lock:
        proc = _radio_proc
        if proc is None or proc.poll() is not None:
            print("[radio_virgin] _stop_radio: nessun processo attivo", flush=True)
            _radio_proc = None
            return False
        try:
            print("[radio_virgin] _stop_radio: termino processo pid =", proc.pid, flush=True)
            proc.terminate()
        except Exception as e:
            print("[radio_virgin] _stop_radio: eccezione su terminate:", repr(e), flush=True)
        _radio_proc = None
        return True


def _is_radio_on() -> bool:
    with _radio_lock:
        on = _radio_proc is not None and _radio_proc.poll() is None
        print("[radio_virgin] _is_radio_on:", on, flush=True)
        return on


def can_handle(text: str, context: dict) -> bool:
    t = (text or "").strip()
    if not t:
        print("[radio_virgin] can_handle: testo vuoto o None", flush=True)
        return False
    on_match = bool(RX_ON.search(t))
    off_match = bool(RX_OFF.search(t))
    status_match = bool(RX_STATUS.search(t))
    result = on_match or off_match or status_match
    print(
        "[radio_virgin] can_handle: text=%r on=%s off=%s status=%s -> %s"
        % (t, on_match, off_match, status_match, result),
        flush=True,
    )
    return result


def handle(text: str, context: dict) -> str:
    t = (text or "").strip()
    print("[radio_virgin] handle: text=%r" % t, flush=True)

    # Accendi / avvia radio (riconosciuta via presenza di 'virgin radio')
    if RX_ON.search(t):
        print("[radio_virgin] handle: comando ON riconosciuto", flush=True)
        started = _start_radio()
        if started:
            print("[radio_virgin] handle: radio avviata (nessuna risposta TTS)", flush=True)
            # Non restituiamo testo: cosi stt_vosk non chiama TTS
            return ""
        else:
            print("[radio_virgin] handle: radio gia in riproduzione", flush=True)
            return "La radio e gia in riproduzione."

    # Spegni / ferma radio
    if RX_OFF.search(t):
        print("[radio_virgin] handle: comando OFF riconosciuto", flush=True)
        stopped = _stop_radio()
        if stopped:
            print("[radio_virgin] handle: radio fermata (nessuna risposta TTS)", flush=True)
            # Nessuna risposta vocale per evitare conflitti
            return ""
        else:
            print("[radio_virgin] handle: radio gia spenta", flush=True)
            return "La radio risulta gia spenta."

    # Stato radio: qui invece ha senso che parli
    if RX_STATUS.search(t):
        print("[radio_virgin] handle: comando STATUS riconosciuto", flush=True)
        if _is_radio_on():
            return "La radio e attiva e sta riproducendo Virgin Radio."
        else:
            return "La radio e spenta."

    # Fallback (in teoria non dovrebbe arrivarci)
    print("[radio_virgin] handle: nessun comando riconosciuto (fallback)", flush=True)
    return ""


if __name__ == "__main__":
    # Test rapido da riga di comando:
    # python3 radio_virgin.py
    print("[radio_virgin] eseguito come script, test handle('metti virgin radio')", flush=True)
    resp = handle("metti virgin radio", {})
    print("[radio_virgin] risposta handle:", repr(resp), flush=True)
