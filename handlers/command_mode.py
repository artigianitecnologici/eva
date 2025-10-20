# handlers/command_mode.py
# -*- coding: utf-8 -*-
"""
Handler 'modalità comandi' per app-ollama.py

Funzione:
- Attiva/Disattiva una modalità in cui l'app NON chiama Ollama
  e restituisce semplicemente l'input dell'utente anteponendo '#@#'.
- Comandi supportati (sinonimi):
  - START:  "avvia programmazione", "avvia coding", "esegui comandi",
            "attiva modalità comando", "attiva modalita comando",
            "entra in programmazione", "abilita comandi",
            "start coding", "start command", "start commands"
  - STOP:   "fine programmazione", "termina programmazione", "stop",
            "disattiva modalità comando", "esci", "ferma comandi",
            "end coding", "exit coding", "exit"
  - STATO:  "stato programmazione", "stato comandi", "status coding"

Persistenza stato:
- Scrive un piccolo file 'command_mode.state' dentro ./log/
"""

import os
import re

# ====== Config base path & state file ======
# Questo file si trova in: ./handlers/command_mode.py
# La cartella base del progetto è una su (..)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
STATE_FILE = os.path.join(LOG_DIR, "command_mode.state")

# Prefisso da anteporre ai comandi quando la modalità è attiva
CMD_PREFIX = "#@#"

# ====== Pattern di attivazione/disattivazione/stato ======
# Usiamo regex robuste con vari sinonimi in italiano/inglese
START_PATTERNS = [
    r"\bavvia\s+(programmazione|coding)\b",
    r"\besegui\s+comandi\b",
    r"\b(attiva|abilita)\s+(modalit[aà]\s+comando|modalita\s+comando|comandi)\b",
    r"\bentra\s+in\s+(programmazione|modalit[aà]\s+comando|modalita\s+comando)\b",
    r"\bstart\s+(coding|command|commands?)\b",
]

STOP_PATTERNS = [
    r"\b(fine|termina|stop|ferma|disattiva|esci)\b(\s+(programmazione|coding|comandi|modalit[aà]\s+comando|modalita\s+comando))?\b",
    r"\b(end|exit)\b(\s+(coding|command|commands?))?\b",
]

STATUS_PATTERNS = [
    r"\bstato\s+(programmazione|comandi)\b",
    r"\bstatus\s+(coding|command|commands?)\b",
    r"\bstato\b\s*(modalit[aà]\s+comando|modalita\s+comando)?",
]

# ====== Util ======
def _read_mode() -> bool:
    try:
        if not os.path.exists(STATE_FILE):
            return False
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() == "1"
    except Exception:
        return False

def _write_mode(on: bool) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write("1" if on else "0")
    except Exception:
        # In caso di problemi di I/O, non esplodere: semplicemente non persiste
        pass

def _match_any(patterns, text: str) -> bool:
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False

def _normalize(text: str) -> str:
    return (text or "").strip()

# ====== API richieste da app-ollama ======
def can_handle(text: str, context: dict) -> bool:
    """
    Torna True se:
      - il testo attiva/disattiva/controlla lo stato, oppure
      - la modalità comandi è già attiva (così intercettiamo tutti i messaggi successivi)
    """
    t = _normalize(text)
    if not t:
        # se vuoto, lo gestiamo solo se la modalità è attiva (per evitare di bloccare altri handler)
        return _read_mode()

    if _match_any(START_PATTERNS, t):
        return True
    if _match_any(STOP_PATTERNS, t):
        return True
    if _match_any(STATUS_PATTERNS, t):
        return True

    # Se siamo in modalità comandi, cattura tutto
    if _read_mode():
        return True

    return False

def handle(text: str, context: dict) -> str:
    """
    - Se comando START: attiva modalità e conferma.
    - Se comando STOP:  disattiva modalità e conferma.
    - Se comando STATO: ritorna stato attuale.
    - Se modalità attiva e non è comando di controllo: restituisce '#@#' + testo.
    """
    t = _normalize(text)

    # Comandi di controllo
    if _match_any(START_PATTERNS, t):
        _write_mode(True)
        return ("Modalità comandi ATTIVATA.\n"
                "Da ora ti restituisco qualsiasi testo con prefisso '#@#'.\n"
                "Per uscire: 'fine programmazione' / 'stop' / 'esci'.")
    if _match_any(STOP_PATTERNS, t):
        _write_mode(False)
        return "Modalità comandi DISATTIVATA. Torno a usare il modello."
    if _match_any(STATUS_PATTERNS, t):
        return "Modalità comandi: ON" if _read_mode() else "Modalità comandi: OFF"

    # Pass-through in modalità attiva
    if _read_mode():
        if not t:
            return f"{CMD_PREFIX}(vuoto)"
        # Non modificare il testo, aggiungi solo il prefisso richiesto
        return f"{CMD_PREFIX}{t}"

    # Se non è né comando né modalità attiva, lascia che altri gestiscano
    return ""
