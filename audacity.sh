#!/bin/bash
# audacity.sh — Configuration and execution script for the BirdNET VAMP plugin on Audacity.
#
# This script performs the following steps:
#   1. Verifies the Audacity-VampFix-3.7.7-x86_64.AppImage integrity via SHA256 checksum.
#   2. Creates a desktop shortcut named Audacity-BirdNet that launches the
#      bundled AppImage with VAMP_PATH pointing to $HOME/vamp/.
#
# The Audacity AppImage is self-contained and does not interfere with 
# any existing Audacity installation on the system.
#
# Requirements:
#   - Run from the repository root directory (where this script is located) after executing ./install.sh.
#   - Ubuntu 22.04 or compatible Debian-based system
#
# Usage:
#   bash audacity.sh or ./audacity.sh

set -e

REPO_DIR="$(realpath "$(dirname "$0")")"
APPIMAGE_NAME="Audacity-VampFix-3.7.7-x86_64.AppImage"
APPIMAGE_PATH="$REPO_DIR/$APPIMAGE_NAME"
APPIMAGE_SHA256="b9dfee578ac4bbb6333ef9564c46a8f1ff348b1760abf4be0c0672d67c6eafd3"
ICON_PATH="$REPO_DIR/assets/audacity-icon-256.png"

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
echo "==> Creating Audacity-BirdNet desktop shortcut..."
cat > "$HOME/.local/share/applications/audacity-birdnet.desktop" << DESKTOP
[Desktop Entry]
Name=Audacity-BirdNET
Comment=Audacity Audio Editor with BirdNET Plugin
Exec=bash -c 'VAMP_PATH=$HOME/vamp PATH=$HOME/.local/bin:$PATH "$APPIMAGE_PATH" %f'
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=Audio;AudioVideo;
DESKTOP
update-desktop-database "$HOME/.local/share/applications/" 2>/dev/null || true

# ── Clear Audacity plugin cache ───────────────────────────────────────────────
echo ""
echo "==> Clearing Audacity plugin cache..."
rm -f ~/.config/audacity/pluginregistry.cfg
echo "    Cache cleared. Audacity will rescan plugins on next launch."

# ── Execution ─────────────────────────────────────────────────────────────────
echo ""
echo "Configuration complete!"
echo ""
echo "Launch Audacity-BirdNet from the application menu or run:"
echo "  VAMP_PATH=$HOME/vamp PATH=$HOME/.local/bin:\$PATH ./$APPIMAGE_NAME"
echo ""
echo "Inside Audacity:"
echo "  1. Open an audio file and select the track"
echo "  2. Go to Analyze -> BirdNET"
echo "  3. Detections will appear as labeled regions on the track"
echo ""
echo "Would you like to launch Audacity-BirdNet now? (y/n): "
read -r launch
if [[ $launch == "y" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    VAMP_PATH=$HOME/vamp $APPIMAGE_PATH
fi
