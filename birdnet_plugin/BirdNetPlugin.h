/**
 * BirdNetPlugin.h — VAMP plugin header for bird species detection using BirdNET.
 *
 * This plugin accumulates audio samples during processing, writes them to a
 * temporary WAV file, and invokes the BirdNET inference script (birdnet_run.py)
 * via a Python subprocess. Detections are returned as labeled features with
 * timestamps, visible as label tracks in Audacity or Sonic-Visualiser.
 *
 * Author: Prof. Dr. Juan G. Colonna <github.com/juancolonna>
 * License: MIT
 */

#pragma once
#include <vamp-sdk/Plugin.h>
#include <vector>
#include <string>

class BirdNetPlugin : public Vamp::Plugin {
public:
    BirdNetPlugin(float inputSampleRate);
    virtual ~BirdNetPlugin();

    // ── VAMP metadata ────────────────────────────────────────────────────────
    std::string getIdentifier()    const override;
    std::string getName()          const override;
    std::string getDescription()   const override;
    std::string getMaker()         const override;
    std::string getCopyright()     const override;
    int         getPluginVersion() const override;

    InputDomain getInputDomain() const override;

    // ── Initialisation and reset ─────────────────────────────────────────────
    bool initialise(size_t channels, size_t stepSize,
                    size_t blockSize) override;
    void reset() override;

    // ── Output and parameter descriptors ─────────────────────────────────────
    OutputList     getOutputDescriptors()    const override;
    ParameterList  getParameterDescriptors() const override;
    float          getParameter(std::string id) const override;
    void           setParameter(std::string id, float value) override;

    // ── Audio processing ─────────────────────────────────────────────────────
    FeatureSet process(const float* const* inputBuffers,
                       Vamp::RealTime timestamp) override;
    FeatureSet getRemainingFeatures() override;

    // ── Preferred block and step size ────────────────────────────────────────
    size_t getPreferredBlockSize() const override;
    size_t getPreferredStepSize()  const override;

private:
    // Writes accumulated audio samples to a 16-bit PCM mono WAV file
    void writeWAV(const std::string& path,
                  const float* samples,
                  int n, int sr) const;

    // Holds a single BirdNET detection parsed from JSON output
    struct Detection {
        std::string species;     // scientific name
        float       confidence;  // average confidence score in %
        float       start_time;      // merged segment start time in seconds
        float       end_time;       // merged segment end time in seconds
    };

    // Parses the JSON array returned by birdnet_run.py
    std::vector<Detection> parseJSON(const std::string& json) const;

    // ── Internal state ───────────────────────────────────────────────────────
    std::vector<float> m_audioBuffer;   // accumulates all input samples
    std::string        m_pythonPath;    // path to conda env Python binary
    std::string        m_scriptPath;    // path to birdnet_run.py
    std::string        m_wavPath;       // path to temporary WAV file
    int                m_blockSize;     // VAMP block size (samples per call)
    int                m_channels;      // number of input audio channels
    int                m_topK;          // max species per segment
    float              m_stride;        // sliding window step in seconds
    float              m_threshold;     // minimum confidence threshold (default: 25.0%)
    int                m_fmin; // minimum frequency for bandpass filter
    int                m_fmax; // maximum frequency for bandpass filter
    float              m_geo_model_confidence; // Minimum confidence for geographic model filtering (default: 3.0%). It olny has effect if lat parameters is set different from 90.0 or -90.0.
    float              m_lat;           // Latitude for geographic filtering, 0.0 = disabled (default: 90.0)
    float              m_lon;           // Longitude for geographic filtering, 0.0 = disabled (default: 0.0)
    int                m_week;          // Week of the year for seasonal filtering, 0 = disabled (default: 0). It olny has effect if lat and lon parameters are set.
    Vamp::RealTime     m_startTime;     // timestamp of the first processed block
};
