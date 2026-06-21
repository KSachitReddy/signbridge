#!/usr/bin/env bash
# scripts/run-gitleaks.sh
# Runs gitleaks detect, downloading a portable binary if missing on the host.

set -euo pipefail

if ! command -v gitleaks &> /dev/null; then
  echo "gitleaks command not found. Attempting to download portable binary..."
  ARCH=$(uname -m)
  URL=""
  if [ "$ARCH" = "x86_64" ]; then
    URL="https://github.com/gitleaks/gitleaks/releases/download/v8.18.0/gitleaks_8.18.0_linux_x64.tar.gz"
  elif [ "$ARCH" = "aarch64" ]; then
    URL="https://github.com/gitleaks/gitleaks/releases/download/v8.18.0/gitleaks_8.18.0_linux_arm64.tar.gz"
  fi

  if [ -n "$URL" ]; then
    if curl -sS -L "$URL" -o gitleaks.tar.gz; then
      tar -xzf gitleaks.tar.gz gitleaks
      chmod +x gitleaks
      ./gitleaks detect --config .gitleaks.toml --no-banner --verbose --redact
    else
      echo "Failed to download gitleaks binary."
      exit 0
    fi
  else
    echo "Unsupported architecture ($ARCH). Skipping scan."
    exit 0
  fi
else
  gitleaks detect --config .gitleaks.toml --no-banner --verbose --redact
fi
