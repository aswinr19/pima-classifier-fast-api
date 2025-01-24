#!/bin/bash

SESH="pima"

tmux has-session -t $SESH 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s $SESH -n "neovim"
  tmux send-keys -t $SESH:neovim "source bin/activate" C-m
  tmux send-keys -t $SESH:neovim "nvim src/main.py" C-m

  tmux new-window -t $SESH -n "run"
  tmux send-keys -t $SESH:run "source bin/activate" C-m
  tmux send-keys -t $SESH:run "clear" C-m

  tmux select-window -t $SESH:neovim
fi

tmux attach-session -t $SESH
