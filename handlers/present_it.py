# -*- coding: utf-8 -*-
"""
Handler 'presentazione' per risposte di presentazione del robot.
Versione STRICT: matcha solo se la frase è ESATTAMENTE una richiesta di presentazione.
"""

import re

# Pattern "stretti" (evita parole troppo generiche tipo "presentazione" da sola)
_PATTERNS = [
    r"come\s+ti\s+chiami\??",
    r"chi\s+sei\??",
    r"come\s+ti\s+chiami\s+tu\??",
    r"chi\s+sei\s+tu\??",
    r"come\s+ti\s+presenti\??",
    r"parlami\s+di\s+te\??",
    r"presentati",
    r"presentati\s+pure",
    r"presentati\s+per\s+favore",
    r"ti\s+vuoi\s+presentare\??",
    r"puoi\s+presentarti\??",
    r"potresti\s+presentarti\??",
    r"fai\s+una\s+presentazione\??",
]

_COMPILED = [re.compile(rf"^{p}$", flags=re.IGNORECASE) for p in _PATTERNS]

def can_handle(text: str, context: dict) -> bool:
    if not text:
        return False
    t = text.strip()
    for rx in _COMPILED:
        if rx.fullmatch(t):
            return True
    return False

def handle(text: str, context: dict) -> str:
    return (
        "Ciao! Sono Martino, un robot pensato per aiutarti con domande e compiti del quotidiano. "
        "Sono un po' birichino ma molto volenteroso! Come posso esserti utile adesso? "
        "Per farmi le domande devi dire la parola marrtino e poi farmi la domanda."
    )
