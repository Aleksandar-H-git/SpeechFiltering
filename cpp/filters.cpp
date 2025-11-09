#include "filters.h"
#include <cmath>

// Include the AudioData structure definition
#include "audio_data.h"

namespace filters {

const double PI = 3.14159265358979323846;

BiquadFilter::BiquadFilter(Type type, float frequency, float Q, float sampleRate)
    : type(type), frequency(frequency), Q(Q), sampleRate(sampleRate) {
    calculateCoefficients();
}

void BiquadFilter::calculateCoefficients() {
    float omega = 2.0f * PI * frequency / sampleRate;
    float alpha = std::sin(omega) / (2.0f * Q);
    float cosw = std::cos(omega);
    
    float a0;
    
    switch(type) {
        case Type::LowPass:
            a0 = 1.0f + alpha;
            coeffs[0] = ((1.0f - cosw) / 2.0f) / a0;           // b0
            coeffs[1] = (1.0f - cosw) / a0;                    // b1
            coeffs[2] = ((1.0f - cosw) / 2.0f) / a0;          // b2
            coeffs[3] = (-2.0f * cosw) / a0;                   // a1
            coeffs[4] = (1.0f - alpha) / a0;                   // a2
            break;
            
        case Type::HighPass:
            a0 = 1.0f + alpha;
            coeffs[0] = ((1.0f + cosw) / 2.0f) / a0;          // b0
            coeffs[1] = -(1.0f + cosw) / a0;                  // b1
            coeffs[2] = ((1.0f + cosw) / 2.0f) / a0;         // b2
            coeffs[3] = (-2.0f * cosw) / a0;                  // a1
            coeffs[4] = (1.0f - alpha) / a0;                  // a2
            break;
            
        case Type::BandPass:
            a0 = 1.0f + alpha;
            coeffs[0] = alpha / a0;                           // b0
            coeffs[1] = 0.0f;                                // b1
            coeffs[2] = -alpha / a0;                         // b2
            coeffs[3] = (-2.0f * cosw) / a0;                 // a1
            coeffs[4] = (1.0f - alpha) / a0;                 // a2
            break;
    }
}

float BiquadFilter::process(float input) {
    float output = coeffs[0] * input + coeffs[1] * x1 + coeffs[2] * x2 -
                  coeffs[3] * y1 - coeffs[4] * y2;
    
    x2 = x1;
    x1 = input;
    y2 = y1;
    y1 = output;
    
    return output;
}

void BiquadFilter::reset() {
    x1 = x2 = y1 = y2 = 0.0f;
}

void applySpeechBandpass(AudioData& audio) {
    // Extreme band-pass: strongly attenuate outside 500-1000 Hz
    const float lowCutHz = 80.0f;
    const float highCutHz = 3000.0f;
    const float Q = 0.8f; // moderate Q to reduce ringing
    const int stages = 10; // 10 cascaded stages

    // Process per-channel to keep filter states separate per channel
    const uint16_t channels = audio.channels ? audio.channels : 1;
    for (uint16_t ch = 0; ch < channels; ++ch) {
        // build cascades for this channel
        std::vector<BiquadFilter> hpStages;
        std::vector<BiquadFilter> lpStages;
        hpStages.reserve(stages);
        lpStages.reserve(stages);
        for (int s = 0; s < stages; ++s) {
            hpStages.emplace_back(BiquadFilter::Type::HighPass, lowCutHz, Q, float(audio.sampleRate));
            lpStages.emplace_back(BiquadFilter::Type::LowPass, highCutHz, Q, float(audio.sampleRate));
        }

        // Process samples for this channel
        for (size_t i = ch; i < audio.samples.size(); i += channels) {
            float s = audio.samples[i] / 32768.0f;
            // pass through all HP stages
            for (auto &f : hpStages) s = f.process(s);
            // then through all LP stages
            for (auto &f : lpStages) s = f.process(s);

            // clamp to avoid overflow
            if (s > 0.9999f) s = 0.9999f;
            if (s < -0.9999f) s = -0.9999f;

            audio.samples[i] = static_cast<int16_t>(s * 32767.0f);
        }
    }
}

void applyFilter(AudioData& audio, BiquadFilter& filter) {
    for (size_t i = 0; i < audio.samples.size(); i++) {
        float sample = audio.samples[i] / 32768.0f; // Convert to float (-1 to 1)
        sample = filter.process(sample);
        audio.samples[i] = static_cast<int16_t>(sample * 32768.0f);
    }
}

} 