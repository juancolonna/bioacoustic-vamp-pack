#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "birdnet",
# ]
# ///
"""
birdnet_run.py — BirdNET inference script for the Audacity or Sonic-Visualiser VAMP plugin.
 
This script is called by the VAMP plugin (BirdNetPlugin.cpp) as a subprocess.
It loads a BirdNET acoustic model, runs species prediction on a WAV file,
and prints the predictions as a JSON array to stdout.
 
Consecutive or overlapping detections of the same species are merged into a
single detection spanning from the first to the last segment, with confidence
computed as the average across all merged segments.
 
Usage:
    uv run birdnet_run.py <wav_path> [threshold] [top_k] [stride] [freq_min] [freq_max] [geo_model_confidence] [lat] [lon] [week]
 
Arguments:
    wav_path   : Path to the input WAV file.
    threshold  : Minimum confidence score to report a detection (default: 25.0%, interval: 0-99).
    top_k      : Maximum number of species to consider per segment (default: 10).
    stride     : Sliding window step in seconds, in range [0.1, 3.0] (default: 3.0).
    freq_min   : Lower bound for the bandpass filter in Hz (default: 0).
    freq_max   : Upper bound for the bandpass filter in Hz (default: 15000).
    geo_model_confidence : Minimum confidence for geographic model filtering (default: 3.0%, interval: 0-99). It only has effect if lat and lon parameters are set.
    lat        : Latitude for geographic filtering, 0.0 = disabled (default: 0.0).
    lon        : Longitude for geographic filtering, 0.0 = disabled (default: 0.0).
    week       : Week of the year for seasonal filtering, 0 = disabled (default: 0). It only has effect if lat and lon parameters are set.
 
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
 
from calendar import week
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs
 
import sys
import json
import birdnet

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
    top_k     = int(sys.argv[3])   if len(sys.argv) > 3 else 10
    stride    = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    freq_min  = int(sys.argv[5])   if len(sys.argv) > 5 else 0
    freq_max  = int(sys.argv[6])   if len(sys.argv) > 6 else 15000
    # Convert geo_model_confidence from percentage to 0..0.99
    geo_model_confidence = (float(sys.argv[7]) if len(sys.argv) > 7 else 3.0) / 100.0
    lat       = float(sys.argv[8]) if len(sys.argv) > 8 else 90.0
    lon       = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    week      = int(sys.argv[10])   if len(sys.argv) > 10 else 0

    # Clamp stride to valid range and compute overlap
    stride  = max(0.1, min(3.0, stride))  # ensure stride is in [0.1, 3.0]
    overlap = max(0.0, 3.0 - stride)      # overlap = window_duration - stride

    # Apply geographic/seasonal species filter if coordinates are provided
    species_filter = None
    if (lat != 90.0 and lat != -90.0):
        geo_model      = birdnet.load("geo", "2.4", "tf")
        species_filter = geo_model.predict(lat, lon,
                                           week=week if week > 0 else None,
                                           # geo_model_confidence have to be a float in 0..0.99
                                           min_confidence=geo_model_confidence).to_set()

    # Load BirdNET acoustic model v2.4 with TensorFlow backend
    model = birdnet.load("acoustic", "2.4", "tf")

    # Run prediction with sliding window
    predictions = model.predict(wav_path,
                            # threshold have to be a float in 0..0.99
                            default_confidence_threshold=threshold, 
                            top_k=top_k,
                            overlap_duration_s=overlap,
                            custom_species_list=species_filter,
                            bandpass_fmin=freq_min,
                            bandpass_fmax=freq_max).to_structured_array()
    detections = []
    
    for row in predictions:
        species    = row['species_name']
        scientific = species.split("_")[0]
        common     = species.split("_")[1]
        conf       = int(100.0 * round(row['confidence'], 4)) # Converts confidence from 0..1 to an integer percentage
        
        detections.append({
            "species":    common,
            "scientific": scientific,
            "confidence": conf,
            "start_time":     float(row['start_time']),
            "end_time":      float(row['end_time']),
        })

    # Merge consecutive/overlapping detections of the same species
    detections = merge_detections(detections)

    # Output predictions as JSON to stdout (read by the VAMP plugin via popen)
    print(json.dumps(detections), flush=True)


if __name__ == "__main__":
    main()
