# 🎶 🐦‍⬛ Bioacoustic VAMP Plugins for Audacity and Sonic-Visualiser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-supported-00599C.svg)](https://isocpp.org/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

[![Audacity 3.7.7](https://img.shields.io/badge/Audacity-3.7.7-2C7ED6.svg)](https://www.audacityteam.org/)
[![Sonic-Visualiser](https://img.shields.io/badge/SonicVisualiser-5.2.1-red.svg)](https://www.sonicvisualiser.org/)

[![BirdNET 2.4](https://img.shields.io/badge/BirdNET-2.4-blue.svg)](https://github.com/birdnet-team/birdnet)
[![Perch 2](https://img.shields.io/badge/Perch-2.0-green.svg)](https://www.kaggle.com/models/google/perch)
[![SurfPerch 1](https://img.shields.io/badge/SurfPerch-1.0-orange.svg)](https://www.kaggle.com/models/google/surfperch)
[![YAMNet 1](https://img.shields.io/badge/YAMNet-1.0-lightgreen.svg)](https://www.tensorflow.org/hub/tutorials/yamnet)
[![PiedTamarin](https://img.shields.io/badge/PiedTamarin-sauim--detector-9C27B0.svg)](https://github.com/juancolonna/Sauim)

[![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624.svg)](#installation)
[![Windows](https://img.shields.io/badge/Windows-x86__64-0078D6.svg)](#installation)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-000000.svg)](#installation)

A collection of Bioacoustic VAMP plugins for [Audacity](https://www.audacityteam.org/) and/or [Sonic-Visualiser](https://sonicvisualiser.org/) that run various bioacoustic models to automatically detect and label sounds in audio recordings. Prebuilt binaries are available for **Linux, Windows, and macOS (Apple Silicon)**.

This repository includes plugins for:
- **BirdNET v2.4**: Automatic bird species detection
- **Perch v2**: Bird species detection with improved accuracy
- **SurfPerch v1**: Reef soundscape classification (anthropophony, biophony, geophony)
- **YAMNet v1**: General audio event classes from the AudioSet-YouTube corpus including biophony
- **PiedTamarin**: One-class detector for the critically endangered pied tamarin (*Saguinus bicolor*, "sauim-de-coleira"), endemic to the Manaus region of the Brazilian Amazon — built on Perch embeddings, a band-pass filter, and a One-Class SVM ([Colonna et al., 2025](https://www.biorxiv.org/content/10.1101/2025.10.11.681843))

Detections appear as labeled regions directly on the label track (Audacity) or as an annotation layer (Sonic-Visualiser), with the species/sound name and confidence score. Consecutive or overlapping detections of the same type are automatically merged into a single label.

### How it looks in Audacity
![BirdNET VAMP Plugin in Audacity](assets/screenshot_audacity.png)

### How it looks in Sonic-Visualiser
![BirdNET VAMP Plugin in Sonic-Visualiser](assets/screenshot_sonic.png)

## Features

- **BirdNET v2.4 Plugin**: Automatic bird species detection using BirdNET v2.4 (TensorFlow backend)
  - Nine configurable parameters:
    - **Confidence Threshold** — minimum confidence score to report a detection (default: 25%, interval [1:99])
    - **Top K Species** — maximum number of species candidates per segment (default: 10)
    - **Stride (s)** — sliding window step size in seconds (default: 3.0, interval [1.0,3.0])
    - **High-pass cutoff frequency** — minimum frequency for the bandpass filter in Hz (default: 0)
    - **Low-pass cutoff frequency** — maximum frequency for the bandpass filter in Hz (default: 15000)
    - **Latitude** — latitude for geographic species filtering; 90.0 or -90.0 = disabled (default: 90.0)
    - **Longitude** — longitude for geographic species filtering, used only when Latitude enables the filter (default: 0.0)
    - **Week of the Year** — week number (1–52) for seasonal filtering; 0 = disabled (default: 0)
    - **Geographic Model Confidence** — minimum confidence for the geographic model filter (default: 3.0%, interval [1:99])
- **Perch v2 and SurfPerch v1 Plugins**: Bird species detection with improved accuracy
  - Three configurable parameters:
    - **Confidence Threshold** — minimum confidence score to report a detection (default: 25%, interval [1,99])
    - **Top K Species** — maximum number of species candidates per segment (default: 10)
    - **Stride (s)** — sliding window step size in seconds (default: 3.0, interval [1.0,3.0])
- **YAMNet v1 Plugin**: General audio event detection using YAMNet.
  - Only two configurable parameters:
    - **Confidence Threshold** — minimum confidence score to report a detection (default: 25%, interval [1,99])
    - **Top K Events** — maximum number of acoustic events per segment (default: 10)
- **PiedTamarin Plugin**: One-class detection of pied tamarin (*Saguinus bicolor*) calls, using Perch embeddings, a band-pass filter, and a One-Class SVM (see [Colonna et al., 2025](https://www.biorxiv.org/content/10.1101/2025.10.11.681843) for methodology). Unlike the other plugins, this one outputs a single detected/not-detected label rather than a species list, which is why it ships without a `_labels.csv` file.

- Works on full recordings or selected segments
- Consecutive and overlapping detections of the same type are merged automatically
- Optional geographic and seasonal filtering using BirdNET's built-in geo model (BirdNET plugin only)

## Requirements

All platforms need [uv](https://github.com/astral-sh/uv) (an extremely fast Python package and project manager, written in Rust) to run the Python inference scripts. An internet connection is required the first time a plugin runs, so `uv` can resolve the pinned model dependencies.

| Platform | Additional requirement |
|---|---|
| 🐧 Linux | glibc as new as Ubuntu's current GitHub Actions runner image (Ubuntu 24.04 at the time of writing) or newer. Older distros such as Ubuntu 22.04 may not work — the binaries link dynamically against glibc, which is forward- but not backward-compatible. |
| 🪟 Windows | Windows 10 or later, x86_64 |
| 🍏 macOS | Apple Silicon (arm64). Intel Macs are not currently built. |

## Installation

Download the release archive for your platform from the [latest release](https://github.com/juancolonna/bioacoustic-vamp-pack/releases/latest):

| Platform | File |
|---|---|
| Linux | `bioacoustic-vamp-pack-linux_x86_64.zip` |
| Windows | `bioacoustic-vamp-pack-windows_x86_64.zip` |
| macOS | `bioacoustic-vamp-pack-macos_arm64.zip` |

### 🐧 Linux

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Extract the plugin pack straight into ~/vamp
mkdir -p ~/vamp
unzip bioacoustic-vamp-pack-linux_x86_64.zip -d ~/vamp
```

### 🪟 Windows

```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Extract the plugin pack straight into %USERPROFILE%\vamp
mkdir $HOME\vamp
Expand-Archive bioacoustic-vamp-pack-windows_x86_64.zip -DestinationPath $HOME\vamp
```

### 🍏 macOS

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Extract the plugin pack straight into ~/vamp
mkdir -p ~/vamp
unzip bioacoustic-vamp-pack-macos_arm64.zip -d ~/vamp

# 3. Remove the quarantine flag macOS attaches to downloaded files —
#    required, since the plugin libraries aren't notarized by Apple.
xattr -dr com.apple.quarantine ~/vamp
```

## Running

Set the `VAMP_PATH` environment variable to the folder from the Installation step, then launch Audacity or Sonic-Visualiser.

### 🐧 Linux

```bash
rm -f ~/.config/audacity/pluginregistry.cfg
export VAMP_PATH=$HOME/vamp
audacity
```
or
```bash
export VAMP_PATH=$HOME/vamp
sonic-visualiser
```

If you prefer to use an AppImage, download the official Audacity or Sonic-Visualiser AppImage from the project website and run it with `VAMP_PATH=$HOME/vamp`, for example:

```bash
sudo chmod +x ~/Downloads/Audacity-3.7.7-x86_64.AppImage
VAMP_PATH="$HOME/vamp" "$HOME/Downloads/Audacity-3.7.7-x86_64.AppImage"
```

or:

```bash
sudo chmod +x ~/Downloads/SonicVisualiser-5.2.1-x86_64.AppImage
VAMP_PATH="$HOME/vamp" "$HOME/Downloads/SonicVisualiser-5.2.1-x86_64.AppImage"
```

### 🪟 Windows

```powershell
Remove-Item "$env:APPDATA\audacity\pluginregistry.cfg" -ErrorAction SilentlyContinue
setx VAMP_PATH "$HOME\vamp"
```

> **Note:** `setx` sets the variable permanently, but it only takes effect in *new* processes. Close and reopen PowerShell (or restart your PC) before launching Audacity or Sonic Visualiser normally from the Start Menu.

### 🍏 macOS

macOS apps launched from Finder or the Dock don't inherit Terminal environment variables, so launch the app binary directly from Terminal instead:

```bash
export VAMP_PATH=$HOME/vamp
/Applications/Audacity.app/Contents/MacOS/Audacity
```
or
```bash
export VAMP_PATH=$HOME/vamp
"/Applications/Sonic Visualiser.app/Contents/MacOS/Sonic Visualiser"
```

The plugins will appear in the Analyze menu (Audacity) or Transform menu (Sonic-Visualiser).

## Usage on Audacity

1. Open an audio file in Audacity (**File → Open**)
2. Optionally select a specific region of the track to analyze
3. Go to **Analyze → Bioacoustics** and choose the desired plugin
4. Adjust parameters if desired
5. Click **OK** and wait for the analysis to complete
6. Detections appear as labeled regions on a new label track

## Usage on Sonic-Visualiser

1. Open an audio file in Sonic-Visualiser (**File → Open**)
2. Optionally select a specific region of the track to analyze
3. Go to **Transform → Analysis by Maker → Bioacoustics** and choose the desired plugin
4. Adjust parameters if desired
5. Click **OK** and wait for the analysis to complete
6. Detections appear as labeled regions on a new label layer

> **Note:** Stereo audio files are automatically mixed down to mono by averaging both channels when you execute any of the plugins, which may produce slightly different results compared to a native mono recording. If you are unsure, convert your audio to mono before running the analysis.

## Annotation format

Each label on the track follows the format:

```
Scientific Name (XX%)
```

For example:
```
Poecile atricapillus (56%)
Haemorhous mexicanus (65%)
...
```

Where `XX%` is the average confidence score across all merged segments.

> **Tip:** The output labels can be exported in CSV format via **File → Export Other → Export Labels** in Audacity, or via **File → Export Annotation Layer** in Sonic Visualiser, for further analysis.

## How it works

1. When a plugin (BirdNET, Perch, SurfPerch, YAMNet, or PiedTamarin) is triggered, the VAMP plugin accumulates all audio samples into a buffer
2. At the end of the stream, it writes the buffer to a temporary WAV file
3. Audio is mixed to mono and resampled by each Python script to the sample rate required by its model.
4. It invokes the corresponding Python script (`birdnet_run.py`, `perch_run.py`, `surfperch_run.py`, `yamnet_run.py`, or `piedtamarin_run.py`) as a subprocess using `uv run`, which resolves and reuses the pinned model dependencies
5. The Python script runs the respective model inference and returns detections as a JSON array via stdout
6. Consecutive or overlapping detections of the same type are merged into single labels
7. The plugin reads the JSON, creates VAMP features, and displays them as labeled regions in Audacity or Sonic-Visualiser
8. The temporary WAV file is deleted after processing

## Geographic and Seasonal Filtering on BirdNET only

When Latitude and Longitude are set to non-zero values, the plugin activates BirdNET's geographic model to filter the species list before running acoustic inference. This restricts detections to species that are realistically expected at the given location, significantly reducing false positives. Optionally, setting Week of the Year (1–52) further narrows the filter to species expected at that location during that season. For example, a migratory species present only in summer will be excluded outside its expected seasonal window.

The Geographic Model Confidence parameter controls how broadly the geo model selects candidate species. Lower values (e.g., 1%) include more species in the filter; higher values (e.g., 3%) apply a stricter regional filter.

> **Note:** Geographic filtering has no effect if latitude is set to 90.0 or -90.0 (the default).

## Troubleshooting

**Plugin does not appear in Analyze / Transform menu**
- Make sure `VAMP_PATH` is set correctly, in the same terminal session you launch Audacity or Sonic-Visualiser from (see [Running](#running) above)
- Ensure the plugin files (library, `.py`, and `.csv` files) are directly inside the `vamp` folder, not in a subfolder
- On macOS, make sure the quarantine flag was removed (see [Installation](#installation)) — otherwise Gatekeeper silently blocks the plugin from loading
- Delete the plugin registry cache and restart: `~/.config/audacity/pluginregistry.cfg` (Linux), `%APPDATA%\audacity\pluginregistry.cfg` (Windows), `~/Library/Application Support/audacity/pluginregistry.cfg` (macOS). Audacity caches which plugins it already found and won't necessarily rescan `VAMP_PATH` on its own — deleting only this file forces a fresh scan on next launch. Don't delete `audacity.cfg` (general preferences) or `pluginsettings.cfg` (saved effect parameters) — neither affects plugin discovery.
- If a plugin still doesn't show up after that, check **Tools → Add/Remove Plug-ins** (Audacity) — newly discovered plugins sometimes land there disabled and need to be enabled manually.

**Plugin fails to initialize**
- These plugins require the VAMP host to call them with equal `blockSize` and `stepSize`. Run Audacity or Sonic Visualiser from a terminal to see the diagnostic message.

**No detections produced**
- Try lowering the **Confidence Threshold** (e.g., 10%)
- Make sure the audio contains the expected sounds (bird vocalizations for BirdNET/Perch, reef sounds for SurfPerch, pied tamarin calls for PiedTamarin)
- Check that `uv` is correctly installed and on your `PATH`: run `uv --version` in a new terminal

**Audacity shows "not responding" during analysis**
- This is expected — model inference with TensorFlow can take 10–30 seconds depending on audio length
- Click **Wait** and the analysis will complete normally

## Citation

If you use this plugin in your research, please cite:

```bibtex
@software{colonna2026bioacoustic_vamp,
  author  = {Colonna, Juan G.},
  title   = {Bioacoustic VAMP Plugins for Audacity and Sonic-Visualiser},
  year    = {2026},
  url     = {https://github.com/juancolonna/birdnet-vamp-plugin}
}
```

## License and Author

MIT License — see [LICENSE](LICENSE) for details.

**Prof. Dr. Juan G. Colonna, IComp,UFAM** — [github.com/juancolonna](https://github.com/juancolonna)
