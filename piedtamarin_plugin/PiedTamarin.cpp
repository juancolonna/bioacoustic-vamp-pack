/**
 * PiedTamarinPlugin.cpp — VAMP plugin implementation for reef coral sounds detection with Pied Tamarin.
 *
 * Processing strategy:
 *   1. process()              — accumulates all input samples into m_audioBuffer.
 *   2. getRemainingFeatures() — writes a temporary WAV file, invokes piedtamarin_run.py
 *                               via popen(), parses the JSON output, and returns
 *                               labeled VAMP features with timestamps.
 *
 * The Python subprocess (piedtamarin_run.py) is executed via `uv run`, which
 * automatically resolves and reuses the pre-installed environment declared
 * as inline metadata (PEP 723) at the top of the script. Dependencies are
 * pre-installed during setup (install.sh), so no lazy resolution occurs
 * at analysis time.
 *
 * Paths are resolved from the VAMP_PATH environment variable, which points to
 * the directory containing both the plugin (.so) and the inference script (.py).
 *
 * Author: Prof. Dr. Juan G. Colonna <github.com/juancolonna>
 * License: MIT
 */

#include "PiedTamarinPlugin.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vamp/vamp.h>
#include <vamp-sdk/PluginAdapter.h>

using namespace Vamp;

// ── Constructor / Destructor ─────────────────────────────────────────────────

PiedTamarinPlugin::PiedTamarinPlugin(float inputSampleRate)
    : Plugin(inputSampleRate)
    , m_blockSize(0)
    // , m_threshold(25.0f)
    // , m_topK(10)
    , m_stride(5.0f)
{
    const char* vampPath = getenv("VAMP_PATH");

    std::string pluginDir = std::string(vampPath ? vampPath : "");

    m_scriptPath = pluginDir + "/piedtamarin_run.py";
    m_wavPath    = pluginDir + "/piedtamarin_analysis.wav";
}

PiedTamarinPlugin::~PiedTamarinPlugin() {}

// ── Initialisation ───────────────────────────────────────────────────────────

bool PiedTamarinPlugin::initialise(size_t channels, size_t stepSize, size_t blockSize) {
    // Verify block and step size are equal (no overlap allowed) 
    if (stepSize != blockSize) {
        std::cerr << "Unsupported VAMP block configuration. "
            << "stepSize and blockSize must be equal, but got "
            << "stepSize=" << stepSize
            << ", blockSize=" << blockSize
            << "." << std::endl;
        return false;
    }
    m_blockSize = (int)blockSize;
    m_channels  = (int)channels;
    m_audioBuffer.clear();
    return true;
}

void PiedTamarinPlugin::reset() {
    m_audioBuffer.clear();
}

// ── Audio accumulation ───────────────────────────────────────────────────────

Plugin::FeatureSet
PiedTamarinPlugin::process(const float* const* inputBuffers,
                          Vamp::RealTime timestamp)
{
    // Capture the start time from the first processed block
    if (m_audioBuffer.empty())
        m_startTime = timestamp;

    // Accumulate samples — mix to mono by averaging all channels
    for (int i = 0; i < m_blockSize; i++) {
        float sample = inputBuffers[0][i];
        if (m_channels > 1)
            sample = (sample + inputBuffers[1][i]) * 0.5f;
        m_audioBuffer.push_back(sample);
    }

    return FeatureSet();
}

// ── Full analysis at end of stream ───────────────────────────────────────────

Plugin::FeatureSet PiedTamarinPlugin::getRemainingFeatures() {
    FeatureSet output;

    if (m_audioBuffer.empty())
        return output;

    // Write accumulated samples to a temporary WAV file
    writeWAV(m_wavPath,
             m_audioBuffer.data(),
             (int)m_audioBuffer.size(),
             (int)m_inputSampleRate);
    m_audioBuffer.clear();

    // Build and run the Python subprocess via uv run
    std::ostringstream cmd;
    cmd << "uv run " << m_scriptPath
        << " " << m_wavPath
        // << " " << m_threshold
        // << " " << m_topK
        << " " << m_stride;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) return output;

    // Read JSON output from stdout
    std::string json;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe))
        json += buf;
    pclose(pipe);

    // Parse detections and build VAMP features
    for (auto& d : parseJSON(json)) {
        Feature f;
        f.hasTimestamp = true;
        f.timestamp    = RealTime::fromSeconds(d.start_time) + m_startTime;
        f.hasDuration  = true;
        f.duration     = RealTime::fromSeconds(d.end_time - d.start_time);
        f.label = d.species + " (" + std::to_string((int)d.confidence) + "%)";
        f.values.push_back(d.confidence);
        output[0].push_back(f);
    }

    // Remove temporary WAV file
    std::remove(m_wavPath.c_str());

    return output;
}

// ── WAV writer (32 bit mono) ─────────────────────────────────────────────

void PiedTamarinPlugin::writeWAV(const std::string& path,
                                  const float* samples,
                                  int n,
                                  int sr) const
{
    std::ofstream f(path, std::ios::binary);

    auto w16 = [&](uint16_t v){ f.write(reinterpret_cast<const char*>(&v), 2); };
    auto w32 = [&](uint32_t v){ f.write(reinterpret_cast<const char*>(&v), 4); };

    const uint16_t audioFormat    = 3;   // IEEE float
    const uint16_t channels       = 1;   // mono
    const uint16_t bitsPerSample  = 32;
    const uint16_t bytesPerSample = bitsPerSample / 8;
    const uint16_t blockAlign     = channels * bytesPerSample;
    const uint32_t byteRate       = sr * blockAlign;
    const uint32_t dataBytes      = n * blockAlign;

    f.write("RIFF", 4);
    w32(36 + dataBytes);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    w32(16);
    w16(audioFormat);             // 3 = IEEE float
    w16(channels);                // mono
    w32((uint32_t)sr);            // host sample rate
    w32(byteRate);
    w16(blockAlign);
    w16(bitsPerSample);

    f.write("data", 4);
    w32(dataBytes);

    for (int i = 0; i < n; i++) {
        float v = std::max(-1.0f, std::min(1.0f, samples[i]));
        f.write(reinterpret_cast<const char*>(&v), sizeof(float));
    }
}

// ── Minimal JSON parser ──────────────────────────────────────────────────────

std::vector<PiedTamarinPlugin::Detection>
PiedTamarinPlugin::parseJSON(const std::string& json) const
{
    std::vector<Detection> detections;
    size_t pos = 0;

    while ((pos = json.find('{', pos)) != std::string::npos) {
        size_t end = json.find('}', pos);
        if (end == std::string::npos) break;

        std::string obj = json.substr(pos, end - pos + 1);

        // Extract a string value by key
        auto str = [&](const std::string& key) -> std::string {
            auto k = obj.find("\"" + key + "\"");
            if (k == std::string::npos) return "";
            auto c  = obj.find(':', k);
            auto q1 = obj.find('"', c + 1);
            auto q2 = obj.find('"', q1 + 1);
            return obj.substr(q1 + 1, q2 - q1 - 1);
        };

        // Extract a numeric value by key
        auto num = [&](const std::string& key) -> float {
            auto k = obj.find("\"" + key + "\"");
            if (k == std::string::npos) return 0.f;
            auto c = obj.find(':', k);
            return std::strtof(obj.c_str() + c + 1, nullptr);
        };

        Detection d;
        d.species    = str("scientific");
        d.confidence = num("confidence");
        d.start_time     = num("start_time");
        d.end_time      = num("end_time");

        if (!d.species.empty())
            detections.push_back(d);

        pos = end + 1;
    }
    return detections;
}

// ── Preferred block and step size ────────────────────────────────────────────

size_t PiedTamarinPlugin::getPreferredBlockSize() const { return 256; }
size_t PiedTamarinPlugin::getPreferredStepSize()  const { return 256; }

// ── Configurable parameters ──────────────────────────────────────────────────

Plugin::ParameterList PiedTamarinPlugin::getParameterDescriptors() const {
    // ParameterDescriptor p{};
    // p.identifier   = "threshold";
    // p.name         = "Confidence Threshold";
    // p.description  = "Minimum confidence score (%) to report a detection";
    // p.unit         = "%";
    // p.minValue     = 0.0f;
    // p.maxValue     = 99.0f;
    // p.defaultValue = 25.0f;
    // p.isQuantized  = false;

    // ParameterDescriptor p2{};
    // p2.identifier   = "top_k";
    // p2.name         = "Top K Species";
    // p2.description  = "Maximum number of species candidates per segment";
    // p2.unit         = "";
    // p2.minValue     = 1.0f;
    // p2.maxValue     = 38.0f;
    // p2.defaultValue = 10.0f;
    // p2.isQuantized  = true;
    // p2.quantizeStep = 1.0f;

    ParameterDescriptor p{};
    p.identifier   = "stride";
    p.name         = "Stride";
    p.description  = "Sliding window step size in seconds";
    p.unit         = "s";
    p.minValue     = 1.0f;
    p.maxValue     = 5.0f;
    p.defaultValue = 5.0f;
    p.isQuantized  = false;

    // return { p, p2, p3 };
    return { p };
}

float PiedTamarinPlugin::getParameter(std::string id) const {
    // if (id == "threshold") return m_threshold;
    // if (id == "top_k")     return (float)m_topK;
    if (id == "stride")    return m_stride;
    return 0.0f;
}

void PiedTamarinPlugin::setParameter(std::string id, float value) {
    // if (id == "threshold") m_threshold = value;
    // if (id == "top_k")     m_topK = (int)value;
    if (id == "stride")    m_stride = value;
}

// ── VAMP metadata ────────────────────────────────────────────────────────────

std::string PiedTamarinPlugin::getIdentifier()    const { return "piedtamarin-vamp"; }
std::string PiedTamarinPlugin::getName()          const { return "Pied Tamarin v1.0"; }
std::string PiedTamarinPlugin::getDescription()   const { return "Reef coral sounds detection using Pied Tamarin v1.0"; }
std::string PiedTamarinPlugin::getMaker()         const { return "Bioacoustics"; }
std::string PiedTamarinPlugin::getCopyright()     const { return "MIT License — Prof. Dr. Juan G. Colonna <github.com/juancolonna>"; }
int         PiedTamarinPlugin::getPluginVersion() const { return 1; }

Plugin::InputDomain PiedTamarinPlugin::getInputDomain() const {
    return TimeDomain;
}

Plugin::OutputList PiedTamarinPlugin::getOutputDescriptors() const {
    OutputDescriptor d;
    d.identifier       = "detections";
    d.name             = "Pied Tamarin Detector v1.0";
    d.description      = "Detected pied tamarin calls with Perch embeddings and One-Class SVM.";
    d.unit             = "Sounds (confidence %)";
    d.hasFixedBinCount = true;
    d.binCount         = 1;
    d.sampleType       = OutputDescriptor::VariableSampleRate;
    d.hasDuration      = true;
    return { d };
}

// ── VAMP entry point ─────────────────────────────────────────────────────────

const VampPluginDescriptor*
vampGetPluginDescriptor(unsigned int version, unsigned int index) {
    if (version < 1 || index > 0) return nullptr;
    static Vamp::PluginAdapter<PiedTamarinPlugin> adapter;
    return adapter.getDescriptor();
}
