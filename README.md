# eva
EVA AI 


# prerequisite
sudo apt install python3.12-venv
sudo apt install python3-pip

# per creare enviroment
python3 -m venv myenv

# per attivarlo
source myenv/bin/activate
pip3 --version
@ installazione librerie
pip3 install -r requirements.txt

# Disattivazione 
deactivate

# scaricare le voci 

# (sei già in ~/eva/models/piper)

# crea cartella modelli (se non esiste)
mkdir -p ~/eva/models/piper && cd ~/eva/models/piper

# Paola (medium) – ONNX + JSON (Rhasspy)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium/it_IT-paola-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium/it_IT-paola-medium.onnx.json


# Aurora
wget https://huggingface.co/kirys79/piper_italiano/resolve/main/Aurora/it_IT-aurora-medium.onnx
wget https://huggingface.co/kirys79/piper_italiano/resolve/main/Aurora/it_IT-aurora-medium.onnx.json

# Leonardo
wget https://huggingface.co/kirys79/piper_italiano/resolve/main/Leonardo/leonardo-epoch=2024-step=996300.onnx
wget https://huggingface.co/kirys79/piper_italiano/resolve/main/Leonardo/leonardo-epoch=2024-step=996300.json

# Giorgio
wget https://huggingface.co/kirys79/piper_italiano/resolve/main/Giorgio/giorgio-epoch=5028-step=1098436.onnx
wget https://huggingface.co/kirys79/piper_italiano/resolve/main/Giorgio/giorgio-epoch=5028-step=1098436.json

# EVA – Chatbot con Ollama, Telegram e TTS Piper

Questo progetto integra:
- **app-ollama.py** → server Flask per interrogare Ollama via API REST.
- **tg_ollama_bridge.py** → bot Telegram che inoltra messaggi (testo e vocali) a Ollama e restituisce risposte in **testo** e **audio TTS** (con Piper).

---

## 🚀 Funzionalità
- Integrazione con **Ollama** (qualsiasi modello installato).
- Bot Telegram con comandi:
  - `/start` → info e stato
  - `/model <nome>` → cambia modello Ollama
  - `/health` → controlla connessione al server
- **ASR (riconoscimento vocale)**: trascrizione dei vocali con `faster-whisper`.
- **TTS (sintesi vocale)**: risposte convertite in audio con **Piper** e inviate come messaggio vocale Telegram.
- Supporto multi-voce (Paola, Aurora, Leonardo, Giorgio).

---

## 📋 Requisiti

- **Ubuntu 24.04 / WSL2**
- **Python 3.10+** (testato con 3.12)
- **ffmpeg**  
  ```bash
  sudo apt update && sudo apt install -y ffmpeg

# installazione di piper
sudo apt install -y pipx
pipx install piper-tts
# l'eseguibile di solito finisce in ~/.local/bin/piper

# test connessione con ollama
 curl -s http://192.168.1.13:11434/api/tags | jq

# esempio di .env
BOT_TOKEN=
APP_BASE_URL=http://127.0.0.1:5000/json
DEFAULT_MODEL=gemma2:2b
