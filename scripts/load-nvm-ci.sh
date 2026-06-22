#!/usr/bin/env bash
# scripts/load-nvm-ci.sh
# Resolves NVM path, loads it, and runs npm installation.

export NVM_DIR="$HOME/.nvm"
if [ ! -s "$NVM_DIR/nvm.sh" ]; then
  export NVM_DIR="/home/ksr/.nvm"
fi

if [ -s "$NVM_DIR/nvm.sh" ]; then
  echo "Loading NVM from $NVM_DIR..."
  . "$NVM_DIR/nvm.sh"
else
  echo "NVM not found. Relying on host environment PATH."
fi

node -v || echo "node command not found"
npm -v || echo "npm command not found"

# Run dependency installation only if node_modules is missing
if [ ! -d "node_modules" ]; then
  npm ci --legacy-peer-deps || npm install --legacy-peer-deps
fi
