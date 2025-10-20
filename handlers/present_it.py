# -*- coding: utf-8 -*-
"""
Handler 'presentazione' per risposte di presentazione del robot.

Riconosce richieste come:
- "come ti chiami?"
- "chi sei?"
- "come ti presenti?"
- "parlami di te"
- "presentati"
- "ti vuoi presentare"
- "puoi presentarti"
- "fai una presentazione"
- "presentazione"
"""

import re

# Compiliamo i pattern una volta sola (case-insensitive)
_PATTERNS = [
    r"\bcome\s+ti\s+chiami\??\b",
    r"\bchi\s+sei\??\b",
    r"\bcome\s+ti\s+chiami\s+tu\??\b",
    r"\bchi\s+sei\s+tu\??\b",
    r"\bcome\s+ti\s+presenti\??\b",
    r"\bparlami\s+di\s+te\??\b",

    # Varianti "presentati"
    r"\bpresentati\b",
    r"\bpresenta\b",
    r"\bpresentadi\b",
    r"\bpresentati\s+pure\b",
    r"\bpresentati\s+per\s+favore\b",

    # Varianti "presentare/presentarti"
    r"\bti\s+vuoi\s+presentare\??\b",
    r"\bpuoi\s+presentarti\??\b",
    r"\bpotresti\s+presentarti\??\b",

    # Altre forme comuni
    r"\bfai\s+una\s+presentazione\??\b",
    r"\bpresentazione\??\b",
]

_COMPILED = [re.compile(p, flags=re.IGNORECASE) for p in _PATTERNS]

def can_handle(text: str, context: dict) -> bool:
    if not text:
        return False
    t = text.strip()
    for rx in _COMPILED:
        if rx.search(t):
            return True
    return False

def handle(text: str, context: dict) -> str:
    # Risposta di presentazione (personalizzabile)
    return (
        "Ciao! Sono Martino, un robot pensato per aiutarti con domande e compiti del quotidiano. "
        "Sono un po' birichino ma molto volenteroso! Come posso esserti utile adesso? "
        "Se vuoi, mi puoi anche adottare. Ciao per farmi le domande devi dire la parola marrtino e poi farmi la domanda"
    )
