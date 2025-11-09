#ifndef SPEECH_ENHANCE_H
#define SPEECH_ENHANCE_H

#include "spectral.h"
#include <vector>

namespace speech_enhance {

// Spectral subtraction parameters
struct SpectralSubtractionParams {
    float noiseEstimateSeconds;  // Time to estimate noise (e.g., 0.5s)
    float overSubtractionFactor; // Alpha (1.0 = basic, >1 = aggressive)
    float spectralFloor;         // Minimum gain to prevent total silence
    
    SpectralSubtractionParams()
        : noiseEstimateSeconds(0.5f), 
          overSubtractionFactor(1.5f),
          spectralFloor(0.01f) {}
};

// Apply spectral subtraction
void applySpectralSubtraction(AudioData& audio, 
                               const spectral::STFTParams& stftParams,
                               const SpectralSubtractionParams& params,
                               const std::vector<bool>& voiceActivity = {});

// Wiener filter parameters
struct WienerFilterParams {
    float noiseEstimateSeconds;
    float priorSNR;  // A priori SNR estimation factor
    
    WienerFilterParams()
        : noiseEstimateSeconds(0.5f), priorSNR(0.98f) {}
};

// Apply Wiener filtering
void applyWienerFilter(AudioData& audio,
                       const spectral::STFTParams& stftParams,
                       const WienerFilterParams& params);

// Simple voice activity detection
// Returns vector of bools (true = speech present in frame)
std::vector<bool> detectVoiceActivity(const AudioData& audio,
                                       const spectral::STFTParams& stftParams,
                                       float energyThreshold = 0.02f);

} // namespace speech_enhance

#endif // SPEECH_ENHANCE_H