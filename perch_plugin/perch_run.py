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
perch_run.py — Perch inference script for the Audacity or Sonic-Visualiser VAMP plugin.
 
This script is called by the VAMP plugin (PerchPlugin.cpp) as a subprocess.
It loads a Perch acoustic model, runs species prediction on a WAV file,
and prints the predictions as a JSON array to stdout.
 
Consecutive or overlapping detections of the same species are merged into a
single detection spanning from the first to the last segment, with confidence
computed as the average across all merged segments.
 
Usage:
    uv run perch_run.py <wav_path> [threshold] [top_k] [stride] [freq_min] [freq_max] [geo_model_confidence] [lat] [lon] [week]
 
Arguments:
    wav_path   : Path to the input WAV file.
    threshold  : Minimum confidence score to report a detection (default: 25.0%, interval: 0-99).
    top_k      : Maximum number of species to consider per segment (default: 10).
    stride     : Sliding window step in seconds, in range [0.1, 3.0] (default: 3.0).
 
Output:
    JSON array of detections, each containing:
        - species    : Common name of the detected species.
        - scientific : Scientific name of the detected species.
        - confidence : Average confidence score across merged segments (4 decimal places).
        - start_time     : Start time of the merged detection in seconds.
        - end_time      : End time of the merged detection in seconds.

 Author: Prof. Dr. Juan G. Colonna <github.com/juancolonna>
 License: MIT
"""
 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs

import sys
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
# tf.experimental.numpy.experimental_enable_numpy_behavior()

import csv
labels_path = os.path.join(os.path.dirname(__file__), "labels.csv")
labels = []
with open(labels_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row["inat2024_fsd50k"])

def merge_detections(detections):
    """
    Merge consecutive or overlapping detections of the same species.

    Two or more detections of the same species are merged if their start times
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

    # Sort by species then by start time for consistent merging
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
    top_k     = int(sys.argv[3])    if len(sys.argv) > 3 else 10
    stride    = float(sys.argv[4])  if len(sys.argv) > 4 else 5.0

    # Clamp stride to valid range and compute overlap
    stride  = max(0.1, min(5.0, stride))  # ensure stride is in [0.1, 5.0]

    # Load Perch v2 acoustic model v2.4 with TensorFlow backend
    model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')

    # Read audio file as mono 32 kHz waveform.
    waveform, _ = librosa.load(wav_path, sr=32000, mono=True)
    waveform = waveform.astype(np.float32, copy=False)

    # Perch expects 5-second windows: 5 * 32000 samples.
    sample_rate = 32000
    window_len = int(5 * sample_rate)
    stride_len = int(stride * sample_rate)

    detections = []

    # Slide a 5-second window across the full waveform.
    for start_sample in range(0, len(waveform), stride_len):
        end_sample = start_sample + window_len

        window = waveform[start_sample:end_sample]

        if len(window) < window_len:
            window = np.pad(window, (0, window_len - len(window)))

        start_time = start_sample / sample_rate
        end_time = min(end_sample, len(waveform)) / sample_rate

        predictions = model.signatures["serving_default"](inputs=window[np.newaxis, :])

        scores = tf.sigmoid(predictions["label"]).numpy()[0]
        candidate_indices = np.where(scores >= threshold)[0]
        top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1][:top_k]]

        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            conf = round(100.0 * score, 4)
            label = labels[idx]

            detections.append({
                "species": label,
                "scientific": label,
                "confidence": conf,
                "start_time": round(start_time, 4),
                "end_time": round(end_time, 4),
            })

    # Merge consecutive/overlapping detections of the same species
    detections = merge_detections(detections)

    # Output predictions as JSON to stdout
    print(json.dumps(detections), flush=True)

if __name__ == "__main__":
    main()
