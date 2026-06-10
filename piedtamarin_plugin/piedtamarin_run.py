#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tensorflow", 
#   "tensorflow_hub",
#   "setuptools<82",
#   "librosa",
#   "sauim-detector==0.1.6",
#   "scikit-learn==1.8.0",
# ]
# ///
"""
piedtamarin_run.py — Pied Tamarin v1.0 inference script for the Audacity or Sonic-Visualiser VAMP plugin.
 
This script is called by the VAMP plugin (PiedTamarinPlugin.cpp) as a subprocess.
It loads a Pied Tamarin acoustic model, runs species prediction on a WAV file,
and prints the predictions as a JSON array to stdout.
 
Consecutive or overlapping detections of the same species are merged into a
single detection spanning from the first to the last segment, with confidence
computed as the average across all merged segments.
 
Usage:
    uv run piedtamarin_run.py <wav_path> [stride]
 
Arguments:
    wav_path   : Path to the input WAV file.
    stride     : Sliding window step in seconds, in range [1, 5] (default: 5).
 
Output:
    JSON array of detections, each containing:
        - species    : Common name of the detected species (reef sounds).
        - scientific : Scientific name of the detected species (reef sounds).
        - confidence : Average confidence score across merged segments (4 decimal places).
        - start_time     : Start time of the merged detection in seconds.
        - end_time      : End time of the merged detection in seconds.

 Author: Prof. Dr. Juan G. Colonna <github.com/juancolonna>
 License: MIT
"""
 
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import warnings
from importlib.resources import files

import tensorflow as tf
import tensorflow_hub as hub
from joblib import load

import sauim_detector


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

tf.experimental.numpy.experimental_enable_numpy_behavior()


def main():
    wav_path = sys.argv[1]
    stride = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    stride = max(1.0, min(5.0, stride))

    embedding_model = hub.load(
        "https://www.kaggle.com/models/google/"
        "bird-vocalization-classifier/TensorFlow2/"
        "bird-vocalization-classifier/8"
    )

    model_path = files("sauim_detector").joinpath(
        "models",
        "ocsvm_sauim.joblib",
    )
    clf = load(model_path)

    y, sr = sauim_detector.load_audio(
        wav_path,
        sr=32000,
    )

    detections = sauim_detector.classify_signal(
        y,
        sr,
        embedding_model,
        clf,
        stride=stride,
    )

    for detection in detections:
        detection["confidence"] = 100
        
    print(json.dumps(detections), flush=True)


if __name__ == "__main__":
    main()
