#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <cmath>
#include <array>

// Forward declaration of AudioData
struct AudioData;

namespace filters {

class BiquadFilter {
public:
    // Filter types
    enum class Type {
        LowPass,
        HighPass,
        BandPass
    };

    // Initialize filter with type, frequency, Q factor, and sample rate
    BiquadFilter(Type type, float frequency, float Q, float sampleRate);
    
    // Process a single sample
    float process(float input);
    
    // Reset filter state
    void reset();

private:
    // Calculate coefficients based on parameters
    void calculateCoefficients();
    
    Type type;
    float frequency;
    float Q;
    float sampleRate;
    
    // Filter coefficients (b0, b1, b2, a1, a2)
    std::array<float, 5> coeffs;
    
    // Filter state
    float x1 = 0.0f;
    float x2 = 0.0f;
    float y1 = 0.0f;
    float y2 = 0.0f;
};

// Speech band-pass filter (500-2500 Hz)
void applySpeechBandpass(AudioData& audio);

// Generic filter application function
void applyFilter(AudioData& audio, BiquadFilter& filter);

} // namespace filters

#endif // FILTERS_H