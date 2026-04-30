#!/bin/bash
# install.sh — Installation script for the BirdNET VAMP plugin.
#
# This script performs the following steps:
#   1. Installs system build dependencies (cmake, g++, vamp-plugin-sdk).
#   2. Installs uv (if not already installed) — a fast Python package manager.
#   3. Pre-installs birdnet_run.py dependencies via uv sync --script, so the
#      plugin runs instantly inside Audacity without any lazy resolution.
#   4. Compiles the VAMP plugin into the build/ directory.
#   5. Copies birdnet_run.py into build/ alongside the plugin.
#   6. Copies compiled plugin and .py files to ~/vamp.
#
# Requirements:
#   - Ubuntu 22.04 or compatible Debian-based system
#   - Internet connection for downloading packages
#
# Usage:
#   bash install.sh or ./install.sh

set -e

REPO_DIR="$(realpath "$(dirname "$0")")"
VAMP_DIR="$REPO_DIR/build"

# ── Install system build dependencies ────────────────────────────────────────
echo ""
echo "==> Installing system dependencies..."
sudo apt update -qq
sudo apt install -y cmake g++ vamp-plugin-sdk curl libfuse2t64 libopengl0 libjack-jackd2-0 libsndfile1 ffmpeg

# ── Install uv ────────────────────────────────────────────────────────────────
echo ""
echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv installation failed or not found in PATH."
    echo "  Try reopening your terminal and running ./install.sh again."
    exit 1
fi
echo "    uv $(uv --version) ready."

# ── Pre-install script dependencies ──────────────────────────────────────────
echo ""
echo "==> Pre-installing birdnet_run.py dependencies (this may take a few minutes)..."
uv sync --script "$REPO_DIR/birdnet_run.py"
echo "    Dependencies installed and cached successfully."

# ── Compile the VAMP plugin ───────────────────────────────────────────────────
echo ""
echo "==> Compiling VAMP plugin..."
rm -rf "$VAMP_DIR"
mkdir -p "$VAMP_DIR" && cd "$VAMP_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd "$REPO_DIR"

# ── Copy inference script to build/ ──────────────────────────────────────────
cp "$REPO_DIR/birdnet_run.py"           "$VAMP_DIR/"
cp "$REPO_DIR/birdnet-vamp.cat"         "$VAMP_DIR/"
cp "$REPO_DIR/birdnet-vamp.n3"          "$VAMP_DIR/"
cp "$REPO_DIR/birdnet-vamp_COPYING.txt" "$VAMP_DIR/"

# ── Copy plugin + Python files to ~/vamp ─────────────────────────────────────
echo ""
echo "==> Copying plugin files to $HOME/vamp..."
mkdir -p "$HOME/vamp"
cp "$VAMP_DIR/birdnet-vamp.so"          "$HOME/vamp/"
cp "$VAMP_DIR/birdnet_run.py"           "$HOME/vamp/"
cp "$REPO_DIR/birdnet-vamp.cat"         "$HOME/vamp/"
cp "$REPO_DIR/birdnet-vamp.n3"          "$HOME/vamp/"
cp "$REPO_DIR/birdnet-vamp_COPYING.txt" "$HOME/vamp/"

echo ""
echo "Installation complete!"
echo ""
echo "To test the inference script directly:"
echo "  uv run birdnet_run.py audio.wav"
echo ""
echo "To launch Audacity-BirdNet execute:"
echo "  ./audacity.sh"
echo ""
echo "To launch Sonic-Visualiser-BirdNet execute:"
echo "  ./sonic-visualiser.sh"
