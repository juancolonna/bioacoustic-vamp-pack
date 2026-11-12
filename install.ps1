
# ============================================================
# Bioacoustic VAMP Pack - Automatic Installer
#
# Supported platform:
#   Windows 10+ x86_64
#
# Installation directory:
#   $HOME\vamp
# ============================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$REPO = "juancolonna/bioacoustic-vamp-pack"
$VAMP_DIR = Join-Path $HOME "vamp"
$PLATFORM = "windows_x86_64"

Write-Host ""
Write-Host "================================================"
Write-Host "       Bioacoustic VAMP Pack Installer"
Write-Host "================================================"
Write-Host ""

# ------------------------------------------------------------
# 1. Detect architecture
# ------------------------------------------------------------

$ARCH = $env:PROCESSOR_ARCHITECTURE

if ($env:PROCESSOR_ARCHITEW6432) {
    $ARCH = $env:PROCESSOR_ARCHITEW6432
}

if ($ARCH -ne "AMD64") {
    Write-Host "ERROR: Unsupported architecture: $ARCH" -ForegroundColor Red
    Write-Host "This installer requires Windows x86_64."
    exit 1
}

Write-Host "Platform detected: $PLATFORM"

# ------------------------------------------------------------
# 2. Check required PowerShell commands
# ------------------------------------------------------------

foreach ($CMD in @("Invoke-WebRequest", "Expand-Archive")) {
    if (-not (Get-Command $CMD -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: Required command not found: $CMD" -ForegroundColor Red
        exit 1
    }
}

# ------------------------------------------------------------
# 3. Install uv if necessary
# ------------------------------------------------------------

$UV = Get-Command uv.exe -ErrorAction SilentlyContinue

if (-not $UV) {

    Write-Host ""
    Write-Host "Installing uv..."

    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

    # Refresh PATH for the current PowerShell session
    $UV_BIN = Join-Path $HOME ".local\bin"

    if (Test-Path $UV_BIN) {
        $env:PATH = "$UV_BIN;$env:PATH"
    }

    $UV = Get-Command uv.exe -ErrorAction SilentlyContinue
}

if (-not $UV) {
    Write-Host "ERROR: uv installation failed." -ForegroundColor Red
    exit 1
}

$UV_PATH = $UV.Source

Write-Host "uv version:"
& $UV_PATH --version

# ------------------------------------------------------------
# 4. Create installation directory
# ------------------------------------------------------------

Write-Host ""
Write-Host "Creating installation directory..."

New-Item -ItemType Directory -Force -Path $VAMP_DIR | Out-Null

# ------------------------------------------------------------
# 5. Download latest GitHub release
# ------------------------------------------------------------

$ARCHIVE = "bioacoustic-vamp-pack-$PLATFORM.zip"

$URL = "https://github.com/$REPO/releases/latest/download/$ARCHIVE"

$TMP_DIR = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("bioacoustic-vamp-" + [guid]::NewGuid().ToString())

New-Item -ItemType Directory -Path $TMP_DIR | Out-Null

try {

    Write-Host ""
    Write-Host "Downloading latest release..."
    Write-Host "File: $ARCHIVE"

    $ARCHIVE_PATH = Join-Path $TMP_DIR $ARCHIVE

    Invoke-WebRequest -Uri $URL -OutFile $ARCHIVE_PATH

    # --------------------------------------------------------
    # 6. Extract plugin pack
    # --------------------------------------------------------

    Write-Host ""
    Write-Host "Extracting plugins into $VAMP_DIR..."

    Expand-Archive `
        -Path $ARCHIVE_PATH `
        -DestinationPath $VAMP_DIR `
        -Force

    # --------------------------------------------------------
    # 7. Pre-install Python dependencies
    # --------------------------------------------------------

    $SCRIPTS = @(
        "birdnet_run.py",
        "perch_run.py",
        "piedtamarin_run.py",
        "surfperch_run.py",
        "yamnet_run.py"
    )

    Write-Host ""
    Write-Host "================================================"
    Write-Host "       Installing Python dependencies"
    Write-Host "================================================"

    foreach ($SCRIPT in $SCRIPTS) {

        $SCRIPT_PATH = Join-Path $VAMP_DIR $SCRIPT

        if (-not (Test-Path $SCRIPT_PATH)) {
            throw "Script not found: $SCRIPT_PATH"
        }

        Write-Host ""
        Write-Host "Preparing: $SCRIPT"

        & $UV_PATH sync --script $SCRIPT_PATH

        if ($LASTEXITCODE -ne 0) {
            throw "Dependency installation failed for $SCRIPT"
        }
    }

    # --------------------------------------------------------
    # 8. Installation completed
    # --------------------------------------------------------

    Write-Host ""
    Write-Host "================================================"
    Write-Host "       Installation completed successfully!"
    Write-Host "================================================"
    Write-Host ""
    Write-Host "Plugin directory: $VAMP_DIR"
    Write-Host "Platform: $PLATFORM"
    Write-Host ""
    Write-Host "All Python dependencies have been prepared."
    Write-Host ""
    Write-Host "You can now launch Audacity or Sonic Visualiser."
    Write-Host ""

}
finally {

    # Remove temporary download directory
    if (Test-Path $TMP_DIR) {
        Remove-Item -Path $TMP_DIR -Recurse -Force
    }

}

