import re

# Pattern italiani comuni per chiedere al robot di presentarsi
_PATTERNS = [
    r"\bcome\s+ti\s+chiami\??\b",
    r"\bchi\s+sei\??\b",
    r"\bcome\s+ti\s+chiami\s+tu\??\b",
    r"\bchi\s+sei\s+tu\??\b",
    r"\bcome\s+ti\s+presenti\??\b",
    r"\bparlami\s+di\s+te\??\b",
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
    # Risposta predefinita del robot
    # Puoi personalizzare questa parte in base a come vuoi che il robot si presenti
    return "Ciao, io sono un robot progettato per aiutarti con le tue domande e compiti, Mi chiamo martino e sono un pò biricchino , Come posso esserti utile oggi?, Se vuoi mi puoi adottare ."

