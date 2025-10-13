#!/usr/bin/env bash
set -Eeuo pipefail

# Vai nella cartella dove si trova lo script (attesa: .../eva)
cd "$(dirname "${BASH_SOURCE[0]}")"

# Controlli veloci
[[ -d "myenv" ]] || { echo "ERRORE: virtualenv 'myenv' non trovata."; exit 1; }
[[ -f "app-ollama.py" ]] || { echo "ERRORE: file 'app-ollama.py' non trovato."; exit 1; }

# Attiva l'ambiente
# shellcheck source=/dev/null
source myenv/bin/activate

# Avvia l’app (passa eventuali argomenti della CLI)
exec python3 app-ollama.py "$@"
