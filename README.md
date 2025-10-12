# eva
EVA AI 
 Enhanced Virtual Assistant

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

# installare ollama
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.12.2 sh

curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.11.11 sh

Ollama - Installazione e Configurazione su Ubuntu 24.04
=========================================================

Questa guida ti mostra come installare **Ollama** su Ubuntu 24.04, configurarlo per l'avvio automatico tramite Snap e gestirlo con `systemd`.

Requisiti
----------

- Ubuntu 24.04 o superiore
- Una connessione internet attiva
- **Snap** già installato (preinstallato su Ubuntu 24.04)

Installazione tramite Snap
---------------------------

1. **Verifica se Snap è installato**:

   Apri il terminale e verifica che Snap sia installato con il seguente comando:

   .. code-block:: bash
      snap --version

   Se Snap non è installato, esegui il comando per installarlo:

   .. code-block:: bash
      sudo apt update
      sudo apt install snapd

2. **Installa Ollama tramite Snap**:

   Una volta che Snap è installato, esegui il comando seguente per installare **Ollama**:

   .. code-block:: bash
      sudo snap install ollama

3. **Verifica l'installazione**:

   Per verificare che **Ollama** sia stato correttamente installato, esegui:

   .. code-block:: bash
      ollama --version

Avvio Automatico di Ollama tramite systemd
------------------------------------------

Se desideri che **Ollama** parta automaticamente all'avvio del sistema, puoi configurarlo come servizio di sistema utilizzando `systemd`.

### 1. Creazione del file di servizio systemd

Esegui il comando per creare il file di servizio `systemd`:

.. code-block:: bash
   sudo nano /etc/systemd/system/ollama.service

Incolla il seguente contenuto nel file:

.. code-block::

   [Unit]
   Description=Ollama Service
   After=network.target

   [Service]
   ExecStart=/snap/bin/ollama serve
   User=snap
   Group=snap
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target

### 2. Ricarica systemd e abilita il servizio

Ricarica la configurazione di `systemd`:

.. code-block:: bash
   sudo systemctl daemon-reload

Abilita il servizio per far partire **Ollama** automaticamente all'avvio:

.. code-block:: bash
   sudo systemctl enable ollama

### 3. Avvia il servizio

Per avviare immediatamente il servizio **Ollama**, usa il seguente comando:

.. code-block:: bash
   sudo systemctl start ollama

### 4. Verifica lo stato del servizio

Per controllare se **Ollama** è attivo, esegui:

.. code-block:: bash
   sudo systemctl status ollama

Aggiornamenti
-------------

Per aggiornare **Ollama** a una versione successiva tramite Snap, esegui il comando:

.. code-block:: bash
   sudo snap refresh ollama

Disinstallazione
----------------

Se desideri rimuovere **Ollama**, puoi farlo con il comando:

.. code-block:: bash
   sudo snap remove ollama

Note
-----

- **Snap** si occupa di aggiornamenti automatici, quindi non è necessario gestirli manualmente.
- **systemd** offre un modo per monitorare e riavviare **Ollama** automaticamente in caso di problemi.

Link utili
----------

- [Snapcraft Ollama](https://snapcraft.io/ollama)
- [Documentazione Ollama](https://ollama.com/docs)

# Prerequisiti0





sudo apt-get update 
sudo apt-get install -y python3 python3-pip libportaudio2 alsa-utils sox
 
