#!/bin/bash

# Definizione della sessione tmux
SESSION=init

# Funzione per killare la sessione
kill_session() {
  tmux has-session -t $SESSION 2>/dev/null
  if [ $? == 0 ]; then
    tmux kill-session -t $SESSION
    echo "Sessione '$SESSION' killata con successo."
  else
    echo "Nessuna sessione '$SESSION' trovata."
  fi
  exit 0
}

# Gestione del parametro --kill
if [ "$1" == "--kill" ]; then
  kill_session
fi

# Controllo se la sessione esiste giÃ 
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
  # Creazione della sessione tmux
   
  tmux -2 new-session -d -s $SESSION
  tmux rename-window -t $SESSION:0 'server-ai'  # Window 0 is renamed to 'config'
  tmux new-window -t $SESSION:1 -n 'tg-bridge'  # Window 1 named 'docker'
  tmux new-window -t $SESSION:2 -n 'free'  # Window 2 named 'cmdexe'
  tmux new-window -t $SESSION:3 -n 'free01'  # Window 3 named 'robot_bringup'
  tmux new-window -t $SESSION:4 -n 'free02'  # Window 3 named 'robot_bringup'



  # Log files for command output
  CMD_EXE_LOG="/tmp/cmdexe.log"
  ROBOT_BRINGUP_LOG="/tmp/robot_bringup.log"
  AUTOSTART_LOG="/tmp/autostart.log"


  # Commands to be executed in window 0
  tmux send-keys -t $SESSION:0 "source myenv/bin/activate" C-m
  tmux send-keys -t $SESSION:0 "python3 app-ollama.py" C-m

  # Commands to be executed in window 1
  tmux send-keys -t $SESSION:1 "source myenv/bin/activate" C-m
  tmux send-keys -t $SESSION:1 "python3 tg_ollama_bridge.py" C-m


fi

# Apertura della sessione tmux finale
tmux attach -t $SESSION
