# handlers/time_it.py
import re
from datetime import datetime

# Pattern italiani comuni per chiedere l'ora
_PATTERNS = [
    r"\bche\s+ora\s+è\??\b",
    r"\bche\s+ore\s+sono\??\b",
    r"\bdimmi\s+che\s+ore\s+sono\??\b",
    r"\bsai\s+l['’]?ora\??\b",
    r"\bmi\s+puoi\s+dire\s+che\s+ora\s+è\??\b",
    r"\bche\s+ora\s+fa\??\b",
]

def can_handle(text: str, context: dict) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    for pat in _PATTERNS:
        if re.search(pat, t):
            return True
    return False

def handle(text: str, context: dict) -> str:
    # Usa l'ora locale della macchina (WSL/Ubuntu). Se vuoi forzare un fuso,
    # puoi leggere context["config"].get("timezone") e usare zoneinfo.
    now = datetime.now()
    return f"Sono le {now.strftime('%H:%M')}."