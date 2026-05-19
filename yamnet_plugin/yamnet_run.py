#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tensorflow", 
#   "tensorflow_hub",
#   "setuptools<82",
#   "librosa",
# ]
# ///
"""
yamnet_run.py — YamNet v1.0 inference script for the Audacity or Sonic-Visualiser VAMP plugin.
 
This script is called by the VAMP plugin (YamNetPlugin.cpp) as a subprocess.
It loads a YamNet v1.0 acoustic model, runs event detection on a WAV file,
and prints the predictions as a JSON array to stdout.
 
Consecutive or overlapping detections of the same acoustic event are merged into a
single detection spanning from the first to the last segment, with confidence
computed as the average across all merged segments.
 
Usage:
    uv run yamnet_run.py <wav_path> [threshold] [top_k] [stride]
 
Arguments:
    wav_path   : Path to the input WAV file.
    threshold  : Minimum confidence score to report a detection (default: 25.0%, interval: 1-99).
    top_k      : Maximum number of acoustic events to consider per segment (default: 10).
    stride     : Sliding window step in seconds, in range [1.0, 5.0] (default: 5.0).
 
Output:
    JSON array of detections, each containing:
        - species    : Common name of the detected acoustic events.
        - scientific : Scientific name of the detected acoustic events.
        - confidence : Average confidence score across merged segments (4 decimal places).
        - start_time     : Start time of the merged detection in seconds.
        - end_time      : End time of the merged detection in seconds.

 Author: Prof. Dr. Juan G. Colonna <github.com/juancolonna>
 License: MIT
"""
 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs

import sys
import csv
import json
import librosa
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import tensorflow_hub as hub


def load_labels(labels_file):
    """
    Load labels from a CSV file.

    The file must be in the same directory as this script and contain the
    column "inat2024_fsd50k". The order is preserved because model scores
    are matched to labels by index.

    Returns:
        list[str]: Labels read from the CSV file.
    """
    labels_path = os.path.join(os.path.dirname(__file__), labels_file)
    labels = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["display_name"])
    return labels


def merge_detections(detections):
    """
    Merge consecutive or overlapping detections of the same acoustic event.

    Two or more detections of the same acoustic event are merged if their start times
    are <= the end time of the current accumulated segment (i.e., they overlap
    or are exactly consecutive). The merged detection spans from the first
    start to the last end, and its confidence is the average of all merged
    segments.

    Args:
        detections : List of detection dicts sorted by start_time.

    Returns:
        List of merged detection dicts.
    """
    if not detections:
        return []

    # Sort by acoustic event then by start time for consistent merging
    detections.sort(key=lambda d: (d["species"], d["start_time"]))

    merged = []
    current = dict(detections[0])
    current["_conf_sum"]     = current["confidence"]
    current["_conf_count"]   = 1

    for det in detections[1:]:
        det_end = det["end_time"]
        same_species = det["species"] == current["species"]
        overlapping  = det["start_time"] <= current["end_time"]

        if same_species and overlapping:
            # Extend current segment and accumulate confidence
            current["end_time"]       = max(current["end_time"], det_end)
            current["_conf_sum"]  += det["confidence"]
            current["_conf_count"] += 1
        else:
            # Finalise current segment and start a new one
            current["confidence"] = round(current["_conf_sum"] / current["_conf_count"], 4)
            del current["_conf_sum"], current["_conf_count"]
            merged.append(current)
            current = dict(det)
            current["end_time"]       = det_end
            current["_conf_sum"]   = det["confidence"]
            current["_conf_count"] = 1

    # Finalise last segment
    current["confidence"] = round(current["_conf_sum"] / current["_conf_count"], 4)
    del current["_conf_sum"], current["_conf_count"]
    merged.append(current)

    # Re-sort by start time for output
    merged.sort(key=lambda d: d["start_time"])
    return merged


def main():
    # Parse command-line arguments
    wav_path  = sys.argv[1]
    # Convert threshold from percentage to 0..0.99
    threshold = (float(sys.argv[2]) if len(sys.argv) > 2 else 25.0) / 100.0 
    top_k     = int(sys.argv[3])    if len(sys.argv) > 3 else 5

    labels = load_labels('yamnet_labels.csv')

    # Load YamNet v1.0 acoustic model with TensorFlow backend
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    # Read audio file as mono 16 kHz waveform.
    sample_rate = 16000
    waveform, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    waveform = waveform.astype(np.float32, copy=False)
    waveform = waveform - np.mean(waveform)  # zero-mean normalization
    if np.max(np.abs(waveform)) > 1:
        waveform = waveform / np.max(np.abs(waveform))  # peak normalization

    scores, _, _ = model(waveform)
    scores = scores.numpy()

    patch_hop = 0.48
    patch_window = 0.96
    audio_duration = len(waveform) / sample_rate

    detections = []
    for patch_idx, row in enumerate(scores):
        start_time = patch_idx * patch_hop
        end_time = min(start_time + patch_window, audio_duration)
        candidate_indices = np.where(row >= threshold)[0]
        top_indices = candidate_indices[np.argsort(row[candidate_indices])[::-1][:top_k]]

        for class_idx in top_indices:
            detections.append({
                "species": labels[class_idx],
                "scientific": labels[class_idx],
                "confidence": round(float(row[class_idx]) * 100.0, 4),
                "start_time": round(start_time, 4),
                "end_time": round(end_time, 4),
        })
    
    # Merge consecutive/overlapping detections of the same acoustic event
    detections = merge_detections(detections)

    # Output predictions as JSON to stdout
    print(json.dumps(detections), flush=True)

if __name__ == "__main__":
    main()
