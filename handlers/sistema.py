# -*- coding: utf-8 -*-
"""
Handler 'sistema' per comandi di sistema:
 - spegni / shutdown / poweroff / arresta
 - riavvia / reboot / restart
 - stato sistema / uptime / stato rete
Richiede sudo NOPASSWD per /sbin/shutdown.
"""

import os
import re
import socket
import subprocess
import threading
from time import time

# Regex comandi (case-insensitive). Supporta prefisso opzionale #@#
RX_PREFIX = r"^\s*(?:#@#\s*)?"
RX_SPEGNI = re.compile(RX_PREFIX + r"(?:eva\s+)?(?:(?:spegni|shutdown|power[\s-]*off|arresta)(?:\s+(?:pc|sistema|computer|robot))?)\s*$", re.I)
RX_RIAVVIA = re.compile(RX_PREFIX + r"(?:eva\s+)?(?:(?:riavvia|reboot|restart)(?:\s+(?:pc|sistema|computer|robot))?)\s*$", re.I)
RX_STATO  = re.compile(RX_PREFIX + r"(?:(?:stato|status)\s+(?:sistema|rete)|uptime)\b", re.I)

# Percorso shutdown (adatta se diverso)
SHUTDOWN_BIN = "/sbin/shutdown"

def _uptime_human():
    try:
        with open("/proc/uptime", "r") as f:
            secs = float(f.read().split()[0])
    except Exception:
        return "sconosciuto"
    mins, sec = divmod(int(secs), 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    parts = []
    if days: parts.append(f"{days}g")
    if hrs: parts.append(f"{hrs}h")
    if mins: parts.append(f"{mins}m")
    parts.append(f"{sec}s")
    return " ".join(parts)

def _ip_info():
    try:
        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        return f"{host} @ {ip}"
    except Exception:
        return "non disponibile"

def _shutdown_async(args):
    def run():
        try:
            # richiede: user ALL=(root) NOPASSWD: /sbin/shutdown
            subprocess.Popen(["sudo", "-n", SHUTDOWN_BIN] + args)
        except Exception as e:
            # non alziamo eccezioni: l'handler deve solo rispondere testo
            pass
    threading.Thread(target=run, daemon=True).start()

def can_handle(text: str, context: dict) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(RX_SPEGNI.search(t) or RX_RIAVVIA.search(t) or RX_STATO.search(t))

def handle(text: str, context: dict) -> str:
    t = (text or "").strip()

    if RX_SPEGNI.search(t):
        _shutdown_async(["-h", "now"])
        return "Ok, avvio lo spegnimento del sistema."

    if RX_RIAVVIA.search(t):
        _shutdown_async(["-r", "now"])
        return "Ok, avvio il riavvio del sistema."

    if RX_STATO.search(t):
        up = _uptime_human()
        ip = _ip_info()
        return f"Stato sistema: uptime {up}. Rete: {ip}."

    # di fallback non dovrebbe mai arrivarci, per sicurezza:
    return ""
