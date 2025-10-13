# 1) Utente/gruppo di servizio e cartelle modelli
sudo useradd --system --home /var/lib/ollama --shell /usr/sbin/nologin ollama 2>/dev/null || true
sudo mkdir -p /var/lib/ollama/models
sudo chown -R ollama:ollama /var/lib/ollama

# 2) Crea la unit file puntando al binario trovato
sudo tee /etc/systemd/system/ollama.service >/dev/null << 'EOF'
[Unit]
Description=Ollama server
After=network.target

[Service]
User=ollama
Group=ollama
WorkingDirectory=/var/lib/ollama
Environment=OLLAMA_MODELS=/var/lib/ollama/models
# Se vuoi esporlo in LAN, decommenta la riga sotto:
# Environment=OLLAMA_HOST=0.0.0.0:11434
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=2s
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

# 3) Ricarica systemd, abilita e avvia
sudo systemctl daemon-reload
sudo systemctl enable --now ollama

# 4) Controlli rapidi
systemctl status ollama --no-pager
journalctl -u ollama -n 50 --no-pager

# 5) Prova veloce dall’host
curl -s http://127.0.0.1:11434/api/tags | jq . 2>/dev/null || curl -s http://127.0.0.1:11434/api/tags
