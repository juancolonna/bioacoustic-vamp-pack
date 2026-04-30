/**
 * BirdNetPlugin.cpp — VAMP plugin implementation for bird species detection.
 *
 * Processing strategy:
 *   1. process()              — accumulates all input samples into m_audioBuffer.
 *   2. getRemainingFeatures() — writes a temporary WAV file, invokes birdnet_run.py
 *                               via popen(), parses the JSON output, and returns
 *                               labeled VAMP features with timestamps.
 *
 * The Python subprocess (birdnet_run.py) is executed via `uv run`, which
 * automatically resolves and reuses the pre-installed environment declared
 * as inline metadata (PEP 723) at the top of the script. Dependencies are
 * pre-installed during setup (install.sh), so no lazy resolution occurs
 * at analysis time.
 *
 * Paths are resolved from the VAMP_PATH environment variable, which points to
 * the directory containing both the plugin (.so) and the inference script (.py).
 * The ~/.local/bin path (where uv lives) is included in PATH by the .desktop
 * shortcut created by audacity.sh / sonic-visualiser.sh, so `uv run` is always
 * found regardless of how the application was launched.
 *
 * Author: Prof. Dr. Juan G. Colonna <github.com/juancolonna>
 * License: MIT
 */

#include "BirdNetPlugin.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vamp/vamp.h>
#include <vamp-sdk/PluginAdapter.h>

using namespace Vamp;

// ── Constructor / Destructor ─────────────────────────────────────────────────

BirdNetPlugin::BirdNetPlugin(float inputSampleRate)
    : Plugin(inputSampleRate)
    , m_blockSize(0)
    , m_threshold(25.0f)
    , m_topK(10)
    , m_stride(3.0f)
    , m_fmin(0)
    , m_fmax(15000)
    , m_geo_model_confidence(3.0f)
    , m_lat(90.0f)
    , m_lon(0.0f)
    , m_week(0)
{
    const char* vampPath = getenv("VAMP_PATH");

    std::string pluginDir = std::string(vampPath ? vampPath : "");

    m_scriptPath = pluginDir + "/birdnet_run.py";
    m_wavPath    = pluginDir + "/birdnet_analysis.wav";
}

BirdNetPlugin::~BirdNetPlugin() {}

// ── Initialisation ───────────────────────────────────────────────────────────

bool BirdNetPlugin::initialise(size_t channels, size_t, size_t blockSize) {
    m_blockSize = (int)blockSize;
    m_channels  = (int)channels;
    m_audioBuffer.clear();
    return true;
}

void BirdNetPlugin::reset() {
    m_audioBuffer.clear();
}

// ── Audio accumulation ───────────────────────────────────────────────────────

Plugin::FeatureSet
BirdNetPlugin::process(const float* const* inputBuffers,
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

Plugin::FeatureSet BirdNetPlugin::getRemainingFeatures() {
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
        << " " << m_threshold
        << " " << m_topK
        << " " << m_stride
        << " " << m_fmin
        << " " << m_fmax
        << " " << m_geo_model_confidence
        << " " << m_lat
        << " " << m_lon
        << " " << m_week;

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

// ── WAV writer (16-bit PCM mono) ─────────────────────────────────────────────

void BirdNetPlugin::writeWAV(const std::string& path,
                              const float* samples,
                              int n, int sr) const
{
    std::ofstream f(path, std::ios::binary);

    auto w16 = [&](uint16_t v){ f.write((char*)&v, 2); };
    auto w32 = [&](uint32_t v){ f.write((char*)&v, 4); };

    uint32_t dataBytes = (uint32_t)(n * 2);
    f.write("RIFF", 4); w32(36 + dataBytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16);
    w16(1);                       // PCM format
    w16(1);                       // mono
    w32((uint32_t)sr);            // sample rate
    w32((uint32_t)(sr * 2));      // byte rate
    w16(2);                       // block align
    w16(16);                      // bits per sample
    f.write("data", 4); w32(dataBytes);

    for (int i = 0; i < n; i++) {
        float   v = std::max(-1.0f, std::min(1.0f, samples[i]));
        int16_t s = (int16_t)(v * 32767.0f);
        f.write((char*)&s, 2);
    }
}

// ── Minimal JSON parser ──────────────────────────────────────────────────────

std::vector<BirdNetPlugin::Detection>
BirdNetPlugin::parseJSON(const std::string& json) const
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

size_t BirdNetPlugin::getPreferredBlockSize() const { return 1024; }
size_t BirdNetPlugin::getPreferredStepSize()  const { return 1024; }

// ── Configurable parameters ──────────────────────────────────────────────────

Plugin::ParameterList BirdNetPlugin::getParameterDescriptors() const {
    ParameterDescriptor p{};
    p.identifier   = "threshold";
    p.name         = "Confidence Threshold";
    p.description  = "Minimum confidence score (%) to report a detection";
    p.unit         = "%";
    p.minValue     = 0.0f;
    p.maxValue     = 99.0f;
    p.defaultValue = 25.0f;
    p.isQuantized  = false;

    ParameterDescriptor p2{};
    p2.identifier   = "top_k";
    p2.name         = "Top K Species";
    p2.description  = "Maximum number of species candidates per segment";
    p2.unit         = "";
    p2.minValue     = 1.0f;
    p2.maxValue     = 20.0f;
    p2.defaultValue = 10.0f;
    p2.isQuantized  = true;
    p2.quantizeStep = 1.0f;

    ParameterDescriptor p3{};
    p3.identifier   = "stride";
    p3.name         = "Stride";
    p3.description  = "Sliding window step size in seconds";
    p3.unit         = "s";
    p3.minValue     = 1.0f;
    p3.maxValue     = 3.0f;
    p3.defaultValue = 3.0f;
    p3.isQuantized  = false;

    ParameterDescriptor p4{};
    p4.identifier   = "bandpass_fmin";
    p4.name         = "High-pass cutoff frequency";
    p4.description  = "Minimum frequency for bandpass filter";
    p4.unit         = "Hz";
    p4.minValue     = 0.0f;
    p4.maxValue     = 15000.0f;
    p4.defaultValue = 0.0f;
    p4.isQuantized  = true;
    p4.quantizeStep = 1.0f;

    ParameterDescriptor p5{};
    p5.identifier   = "bandpass_fmax";
    p5.name         = "Low-pass cutoff frequency";
    p5.description  = "Maximum frequency for bandpass filter";
    p5.unit         = "Hz";
    p5.minValue     = 0.0f;
    p5.maxValue     = 15000.0f;
    p5.defaultValue = 15000.0f;
    p5.isQuantized  = true;
    p5.quantizeStep = 1.0f;

    ParameterDescriptor p6{};
    p6.identifier   = "geo_model_confidence";
    p6.name         = "Geographic Model Confidence";
    p6.description  = "Minimum confidence (%) for geographic model filtering. It only has effect if lat and lon parameters are set.";
    p6.unit         = "%";
    p6.minValue     = 0.0f;
    p6.maxValue     = 99.0f;
    p6.defaultValue = 3.0f;
    p6.isQuantized  = false;

    ParameterDescriptor p7{};
    p7.identifier   = "lat";
    p7.name         = "Latitude";
    p7.description  = "Latitude for geographic filtering, 0.0 = disabled";
    p7.unit         = "°";
    p7.minValue     = -90.0f;
    p7.maxValue     = 90.0f;
    p7.defaultValue = 90.0f;
    p7.isQuantized  = false;

    ParameterDescriptor p8{};
    p8.identifier   = "lon";
    p8.name         = "Longitude";
    p8.description  = "Longitude for geographic filtering, 0.0 = disabled";
    p8.unit         = "°";
    p8.minValue     = -180.0f;
    p8.maxValue     = 180.0f;
    p8.defaultValue = 0.0f;
    p8.isQuantized  = false;

    ParameterDescriptor p9{};
    p9.identifier   = "week";
    p9.name         = "Week of the Year";
    p9.description  = "Week of the year for seasonal filtering, 0 = disabled. It only has effect if lat and lon parameters are set.";
    p9.unit         = "";
    p9.minValue     = 0.0f;
    p9.maxValue     = 52.0f;
    p9.defaultValue = 0.0f;
    p9.isQuantized  = true;
    p9.quantizeStep = 1.0f;

    return { p, p2, p3, p4, p5, p6, p7, p8, p9 };
}

float BirdNetPlugin::getParameter(std::string id) const {
    if (id == "threshold") return m_threshold;
    if (id == "top_k")     return (float)m_topK;
    if (id == "stride")    return m_stride;
    if (id == "bandpass_fmin") return (float)m_fmin;
    if (id == "bandpass_fmax") return (float)m_fmax;
    if (id == "geo_model_confidence") return m_geo_model_confidence;
    if (id == "lat") return m_lat;
    if (id == "lon") return m_lon;
    if (id == "week") return (float)m_week;
    return 0.0f;
}

void BirdNetPlugin::setParameter(std::string id, float value) {
    if (id == "threshold") m_threshold = value;
    if (id == "top_k")     m_topK = (int)value;
    if (id == "stride")    m_stride = value;
    if (id == "bandpass_fmin") m_fmin = (int)value;
    if (id == "bandpass_fmax") m_fmax = (int)value;
    if (id == "geo_model_confidence") m_geo_model_confidence = value;
    if (id == "lat") m_lat = value;
    if (id == "lon") m_lon = value;
    if (id == "week") m_week = (int)value;
}

// ── VAMP metadata ────────────────────────────────────────────────────────────

std::string BirdNetPlugin::getIdentifier()    const { return "birdnet-vamp"; }
std::string BirdNetPlugin::getName()          const { return "BirdNET"; }
std::string BirdNetPlugin::getDescription()   const { return "Bird species detection using BirdNET v2.4"; }
std::string BirdNetPlugin::getMaker()         const { return "Bioacoustics"; }
std::string BirdNetPlugin::getCopyright()     const { return "MIT License — Prof. Dr. Juan G. Colonna <github.com/juancolonna>"; }
int         BirdNetPlugin::getPluginVersion() const { return 1; }

Plugin::InputDomain BirdNetPlugin::getInputDomain() const {
    return TimeDomain;
}

Plugin::OutputList BirdNetPlugin::getOutputDescriptors() const {
    OutputDescriptor d;
    d.identifier       = "detections";
    d.name             = "BirdNET Detections";
    d.description      = "Detected species with confidence score and timestamp";
    d.unit             = "Species (confidence %)";
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
    static Vamp::PluginAdapter<BirdNetPlugin> adapter;
    return adapter.getDescriptor();
}
