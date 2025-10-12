import re

# Pattern italiani comuni per chiedere informazioni su Robotics3D
_PATTERNS = [
    r"\bcos'\s+è\s+robotics3d\??\b",
    r"\bdi\s+cosa\s+si\s+occupa\s+robotics3d\??\b",
    r"\bparlami\s+di\s+robotics3d\??\b",
    r"\bchi\s+ha\s+creato\s+robotics3d\??\b",
    r"\bchi\s+sono\s+i\s+membri\s+di\s+robotics3d\??\b",
    r"\bchi\s+fa\s+parte\s+di\s+robotics3d\??\b",
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
    # Presentazione dell'azienda Robotics3D e del team
    return (
        "Robotics3D è un'azienda che si occupa di progettazione e sviluppo di robot educativi "
        "e soluzioni innovative nel campo della robotica, come il nostro robot MARRtino. "
        "Siamo un team appassionato e dedicato che lavora con entusiasmo per creare prodotti "
        "tecnologici all'avanguardia. "
        "Il nostro staff è composto da: Paolo, Fabio, Leo, Sara, Francesco ed Ennio. "
        "Ognuno di noi porta un contributo unico e fondamentale per la realizzazione dei nostri progetti!"
    )
