#!/usr/bin/env bash
# scripts/load-nvm-ci.sh
#
# Ensures a working 64-bit Node.js toolchain is on PATH before any job script
# runs, then installs dependencies if they're missing or were installed under
# a different architecture.
#
# Root cause this works around: this runner's desktop only has a 32-bit
# (ia32) Node.js install. Several native-binding dependencies in this
# project's toolchain (rollup, lightningcss, oxc-resolver/knip) never
# published ia32 Windows builds, so anything touching Vite, Tailwind, or
# Knip fails with "Cannot find native binding" under that Node. We fetch a
# portable 64-bit Node distribution once, cache it outside the repo, and
# prepend it to PATH - then reinstall node_modules under it so npm resolves
# the matching x64 native packages instead of the broken ia32 ones.

set -euo pipefail

NODE_VERSION="22.20.0"
CI_NODE_ROOT="${CI_NODE_ROOT:-$HOME/.ci-node64}"
CI_NODE_HOME="$CI_NODE_ROOT/node-v${NODE_VERSION}-win-x64"

current_arch() {
  command -v node >/dev/null 2>&1 && node -p "process.arch" 2>/dev/null || echo "none"
}

is_windows() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

if is_windows && [ "$(current_arch)" != "x64" ]; then
  if [ ! -x "$CI_NODE_HOME/node.exe" ]; then
    echo "No working 64-bit Node found on PATH (arch: $(current_arch)). Fetching portable Node v${NODE_VERSION} (win-x64)..."
    mkdir -p "$CI_NODE_ROOT"
    TMP_ZIP="$(mktemp).zip"
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-win-x64.zip" -o "$TMP_ZIP"
    unzip -q -o "$TMP_ZIP" -d "$CI_NODE_ROOT"
    rm -f "$TMP_ZIP"
  fi
  export PATH="$CI_NODE_HOME:$PATH"
  hash -r
  echo "Using portable 64-bit Node: $CI_NODE_HOME"
else
  # Non-Windows runner (or already on x64): fall back to nvm if present.
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [ -s "$NVM_DIR/nvm.sh" ]; then
    echo "Loading NVM from $NVM_DIR..."
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"
  else
    echo "NVM not found. Relying on host environment PATH."
  fi
fi

node -v || echo "node command not found"
npm -v || echo "npm command not found"
echo "node arch: $(current_arch)"

# Reinstall dependencies if missing, or if the existing node_modules was
# installed under a different Node architecture than the one we're about to
# use - exactly the scenario that caused "Cannot find native binding" errors.
ARCH_MARKER="node_modules/.ci-arch-marker"
if [ ! -d node_modules ] || [ ! -f "$ARCH_MARKER" ] || [ "$(cat "$ARCH_MARKER" 2>/dev/null)" != "$(current_arch)" ]; then
  echo "Installing dependencies for arch $(current_arch)..."
  rm -rf node_modules
  npm ci --legacy-peer-deps || npm install --legacy-peer-deps
  echo "$(current_arch)" > "$ARCH_MARKER"
fi
