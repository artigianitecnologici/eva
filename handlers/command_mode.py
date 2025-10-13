# Caricamento comandi modalità
import os
def _load_commands():
    if not os.path.exists(COMMANDS_PATH):
        log_error(f"File comandi mancante: {COMMANDS_PATH}")
        # fallback minimal
        return {
            "prefix": "#@#",
            "start": [r"avvia\s+programmazione"],
            "stop": [r"fine\s+programmazione", r"\bstop\b"],
            "status": [r"\bstato\s+programmazione\b"]
        }
    try:
        with open(COMMANDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalizza chiavi assenti
        data.setdefault("prefix", "#@#")
        data.setdefault("start", [])
        data.setdefault("stop", [])
        data.setdefault("status", [])
        return data
    except Exception as e:
        log_error(f"Impossibile leggere/parsare {COMMANDS_PATH} — {e}")
        return {"prefix": "#@#", "start": [], "stop": [], "status": []}

COMANDI = _load_commands()
CMD_PREFIX = COMANDI.get("prefix", "#@#")
