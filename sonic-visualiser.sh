#!/bin/bash
# sonic-visualiser.sh — Configuration and execution script for the BirdNET VAMP plugin on Sonic Visualiser.
#
# This script performs the following steps:
#   1. Verifies the Sonic-Visualiser AppImage integrity via SHA256 checksum.
#   2. Creates a desktop shortcut named Sonic-Visualiser-BirdNet that launches the
#      bundled AppImage with VAMP_PATH pointing to $HOME/vamp/.
#
# The Sonic-Visualiser AppImage is self-contained and does not interfere with 
# any existing Sonic-Visualiser installation on the system.
#
# Requirements:
#   - Run from the repository root directory (where this script is located) after executing ./install.sh.
#   - Ubuntu 22.04 or compatible Debian-based system
#
# Usage:
#   bash sonic-visualiser.sh or ./sonic-visualiser.sh

set -e

REPO_DIR="$(realpath "$(dirname "$0")")"
APPIMAGE_NAME="SonicVisualiser-5.2.1-x86_64.AppImage"
APPIMAGE_PATH="$REPO_DIR/$APPIMAGE_NAME"
APPIMAGE_SHA256="e31e7b970db7a9d6e5943c050c0a06c193d2c3fd4a82b6283aa08a9dc798599c"
ICON_PATH="$REPO_DIR/assets/sonic-icon-256.png"

# ── Create required directories ───────────────────────────────────────────────
mkdir -p "$HOME/.local/share/applications"

# ── Verify AppImage integrity (SHA256) ────────────────────────────────────────
echo "==> Verifying AppImage integrity..."
ACTUAL_SHA256=$(sha256sum "$APPIMAGE_PATH" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$APPIMAGE_SHA256" ]; then
    echo "ERROR: AppImage integrity check failed!"
    echo "  Expected: $APPIMAGE_SHA256"
    echo "  Got:      $ACTUAL_SHA256"
    echo "  The file may be corrupted or tampered with. Deleting it."
    rm -f "$APPIMAGE_PATH"
    exit 1
fi
chmod +x "$APPIMAGE_PATH"
echo "    Integrity check passed."

# ── Create desktop shortcut ───────────────────────────────────────────────────
echo ""
echo "==> Creating Sonic-Visualiser-BirdNet desktop shortcut..."
cat > "$HOME/.local/share/applications/sonic-visualiser-birdnet.desktop" << DESKTOP
[Desktop Entry]
Name=Sonic-BirdNET
Comment=Sonic Visualiser with BirdNET Plugin
Exec=bash -c 'VAMP_PATH=$HOME/vamp PATH=$HOME/.local/bin:$PATH "$APPIMAGE_PATH" %f'
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=Audio;AudioVideo;
DESKTOP
update-desktop-database "$HOME/.local/share/applications/" 2>/dev/null || true

# ── Execution ─────────────────────────────────────────────────────────────────
echo ""
echo "Configuration complete!"
echo ""
echo "Launch Sonic-Visualiser-BirdNet from the application menu or run:"
echo "  VAMP_PATH=$HOME/vamp PATH=$HOME/.local/bin:\$PATH ./$APPIMAGE_NAME"
echo ""
echo "Inside Sonic Visualiser:"
echo "  1. Open an audio file and select the track"
echo "  2. Go to Transform -> Analysis by Plugin Name -> BirdNET"
echo "  3. Detections will appear as labeled regions on the track"
echo ""
echo "Would you like to launch Sonic-Visualiser-BirdNet now? (y/n): "
read -r launch
if [[ $launch == "y" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    VAMP_PATH=$HOME/vamp $APPIMAGE_PATH
fi
