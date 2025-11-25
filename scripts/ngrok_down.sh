#!/usr/bin/env bash

echo "--- Stopping ngrok ---"

# Find ngrok PIDs owned by the current user
PIDS=$(pgrep -u "$(whoami)" ngrok)

if [ -z "$PIDS" ]; then
  echo "No ngrok processes found for user $(whoami)"
  exit 0
fi

echo "Killing ngrok PIDs: $PIDS"

# Kill only your own processes
kill $PIDS 2>/dev/null || true

# Wait 0.5s to ensure kill processed
sleep 0.5

echo "ngrok stopped."
