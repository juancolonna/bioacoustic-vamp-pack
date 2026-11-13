
#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Bioacoustic VAMP Pack - Automatic Installer
#
# Supported platforms:
#   Linux x86_64 (glibc >= 2.39)
#   macOS Apple Silicon (arm64)
#
# Installation directory:
#   ~/vamp
# ============================================================

REPO="juancolonna/bioacoustic-vamp-pack"
VAMP_DIR="$HOME/vamp"

echo ""
echo "================================================"
echo "       Bioacoustic VAMP Pack Installer"
echo "================================================"
echo ""

# ------------------------------------------------------------
# 1. Detect operating system and architecture
# ------------------------------------------------------------

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS-$ARCH" in

    Linux-x86_64)
        PLATFORM="linux_x86_64"
        ;;

    Darwin-arm64)
        PLATFORM="macos_arm64"
        ;;

    *)
        echo "ERROR: Unsupported platform: $OS ($ARCH)"
        echo ""
        echo "Supported platforms:"
        echo "  Linux x86_64"
        echo "  macOS Apple Silicon (arm64)"
        exit 1
        ;;

esac

echo "Platform detected: $PLATFORM"

# ------------------------------------------------------------
# 2. Check Linux glibc version
# ------------------------------------------------------------

if [[ "$OS" == "Linux" ]]; then

    if ! command -v getconf >/dev/null 2>&1; then
        echo "ERROR: getconf is required to check glibc."
        exit 1
    fi

    GLIBC_VERSION="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"

    if [[ -z "$GLIBC_VERSION" ]]; then
        echo "ERROR: Unable to detect glibc."
        exit 1
    fi

    if ! awk -v v="$GLIBC_VERSION" 'BEGIN {
        split(v, a, ".")
        exit !((a[1] > 2) || (a[1] == 2 && a[2] >= 39))
    }'; then
        echo "ERROR: glibc $GLIBC_VERSION detected."
        echo "This release requires glibc 2.39 or newer."
        echo "Please use Ubuntu 24.04 or newer."
        exit 1
    fi

    echo "glibc version: $GLIBC_VERSION (OK)"

fi

# ------------------------------------------------------------
# 3. Check required system utilities
# ------------------------------------------------------------

for CMD in curl unzip awk; do
    if ! command -v "$CMD" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $CMD"
        exit 1
    fi
done

if [[ "$OS" == "Darwin" ]]; then
    if ! command -v xattr >/dev/null 2>&1; then
        echo "ERROR: xattr is required on macOS."
        exit 1
    fi
fi

# ------------------------------------------------------------
# 4. Install uv if necessary
# ------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then

    echo ""
    echo "Installing uv..."

    curl -LsSf https://astral.sh/uv/install.sh | sh

fi

# Make uv available in the current shell
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installation failed."
    exit 1
fi

echo "uv version: $(uv --version)"

# ------------------------------------------------------------
# 5. Create installation directory
# ------------------------------------------------------------

echo ""
echo "Creating installation directory..."

mkdir -p "$VAMP_DIR"

# ------------------------------------------------------------
# 6. Download latest GitHub release
# ------------------------------------------------------------

ARCHIVE="bioacoustic-vamp-pack-${PLATFORM}.zip"

URL="https://github.com/${REPO}/releases/latest/download/${ARCHIVE}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo ""
echo "Downloading latest release..."
echo "File: $ARCHIVE"

curl -fL --progress-bar "$URL" -o "$TMP_DIR/$ARCHIVE"

# ------------------------------------------------------------
# 7. Extract plugin pack
# ------------------------------------------------------------

echo ""
echo "Extracting plugins into $VAMP_DIR..."

unzip -oq "$TMP_DIR/$ARCHIVE" -d "$VAMP_DIR"

# ------------------------------------------------------------
# 8. Remove macOS quarantine attributes
# ------------------------------------------------------------

if [[ "$OS" == "Darwin" ]]; then

    echo ""
    echo "Removing macOS quarantine attributes..."

    xattr -dr com.apple.quarantine "$VAMP_DIR" 2>/dev/null || true

fi

# ------------------------------------------------------------
# 9. Pre-install Python dependencies
# ------------------------------------------------------------

SCRIPTS=(
    birdnet_run.py
    perch_run.py
    piedtamarin_run.py
    surfperch_run.py
    yamnet_run.py
)

echo ""
echo "================================================"
echo "       Installing Python dependencies"
echo "================================================"

for SCRIPT in "${SCRIPTS[@]}"; do

    SCRIPT_PATH="$VAMP_DIR/$SCRIPT"

    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "ERROR: Script not found: $SCRIPT_PATH"
        exit 1
    fi

    echo ""
    echo "Preparing: $SCRIPT"

    uv sync --script "$SCRIPT_PATH"

done

# ------------------------------------------------------------
# 10. Installation completed
# ------------------------------------------------------------

echo ""
echo "================================================"
echo "       Installation completed successfully!"
echo "================================================"
echo ""
echo "Plugin directory: $VAMP_DIR"
echo "Platform: $PLATFORM"
echo ""
echo "All Python dependencies have been prepared."
echo ""
echo "You can now launch Audacity or Sonic Visualiser."
echo ""

