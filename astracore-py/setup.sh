#!/usr/bin/env bash
# setup.sh — Install astracore Python bindings
# Usage: bash setup.sh [--release]
set -e

RELEASE_FLAG=""
if [[ "$1" == "--release" ]]; then
  RELEASE_FLAG="--release"
fi

echo "=== AstraCore Python Setup ==="
echo ""

# 1. Check Rust / cargo
if ! command -v cargo &>/dev/null; then
  echo "[1/4] Installing Rust via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
else
  echo "[1/4] Rust/cargo found: $(cargo --version)"
fi

# 2. Check / install maturin
if ! command -v maturin &>/dev/null; then
  echo "[2/4] Installing maturin..."
  pip install --quiet "maturin>=1,<2"
else
  echo "[2/4] maturin found: $(maturin --version)"
fi

# 3. Ensure a virtualenv is active (maturin develop requires one)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
  echo "[3/4] No virtualenv active — creating .venv..."
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "      Activated .venv"
else
  echo "[3/4] Virtualenv active: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
fi

# 4. Build and install astracore-py
echo "[4/4] Building astracore Python bindings${RELEASE_FLAG:+ (release mode)}..."
maturin develop $RELEASE_FLAG

echo ""
echo "Done! Test with:"
echo "  python -c \"import astracore as ac; c = ac.Circuit(2); c.h(0); c.cnot(0,1); c.measure_all(); print(c.run().bitstring())\""
