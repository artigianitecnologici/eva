sudo apt update

# Python + strumenti base
sudo apt install -y \
  python3 python3-pip python3-venv python3-dev

# Audio (PortAudio + ALSA)
sudo apt install -y \
  libportaudio2 portaudio19-dev libportaudiocpp0 \
  libasound2 libasound2-dev

# SoX (per beep e per riprodurre l'audio generato da Piper)
sudo apt install -y sox

# Piper (sintesi vocale)
sudo apt install -y piper
