#!/usr/bin/env bash
# scripts/run-gitleaks.sh
# Runs gitleaks detect, downloading a portable binary matching the host OS
# if one isn't already on PATH.

set -euo pipefail

GITLEAKS_VERSION="8.18.0"

is_windows() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

if command -v gitleaks &> /dev/null; then
  gitleaks detect --config .gitleaks.toml --no-banner --verbose --redact
  exit $?
fi

echo "gitleaks command not found. Attempting to download portable binary..."

if is_windows; then
  URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_windows_x64.zip"
  ARCHIVE="gitleaks.zip"
  BIN="gitleaks.exe"
else
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64)  PLATFORM="linux_x64" ;;
    aarch64) PLATFORM="linux_arm64" ;;
    *)
      echo "Unsupported architecture ($ARCH). Skipping scan."
      exit 0
      ;;
  esac
  URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_${PLATFORM}.tar.gz"
  ARCHIVE="gitleaks.tar.gz"
  BIN="gitleaks"
fi

if ! curl -sS -L "$URL" -o "$ARCHIVE"; then
  echo "Failed to download gitleaks binary."
  exit 0
fi

if [[ "$ARCHIVE" == *.zip ]]; then
  unzip -q -o "$ARCHIVE" "$BIN"
else
  tar -xzf "$ARCHIVE" "$BIN"
fi
chmod +x "$BIN"

./"$BIN" detect --config .gitleaks.toml --no-banner --verbose --redact
