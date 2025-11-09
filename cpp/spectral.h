#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <vector>
#include <functional>
#include "audio_data.h"

namespace spectral {

// STFT parameters
struct STFTParams {
    int frameSize;      // FFT size (power of 2, e.g., 1024, 2048)
    int hopSize;        // Hop between frames (e.g., frameSize/4 for 75% overlap)
    int sampleRate;     // Sample rate of audio
    
    STFTParams(int fs = 2048, int hs = 512, int sr = 44100)
        : frameSize(fs), hopSize(hs), sampleRate(sr) {}
};

// Represents one frame in frequency domain
struct SpectralFrame {
    std::vector<float> magnitude;  // Magnitude spectrum (linear scale)
    std::vector<float> phase;      // Phase spectrum (radians)
};

// Callback function type for processing spectral frames
// Parameters: frame data, frame index, channel index
// Modify frame.magnitude to apply spectral processing
using SpectralProcessor = std::function<void(SpectralFrame&, int, int)>;

// Main STFT processing function
// Applies spectral processing to each frame via callback
// Uses overlap-add reconstruction with Hann windowing
void processSTFT(AudioData& audio, const STFTParams& params, 
                 SpectralProcessor processor);

// Utility: Apply Hann window to a time-domain frame
void applyHannWindow(std::vector<float>& frame);

} // namespace spectral

#endif // SPECTRAL_H