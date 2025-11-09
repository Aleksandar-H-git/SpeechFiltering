#include "speech_enhance.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace speech_enhance {

void applySpectralSubtraction(AudioData& audio, 
                             const spectral::STFTParams& stftParams,
                             const SpectralSubtractionParams& params) {
    // Calculate number of frames for noise estimation
    int framesForNoise = static_cast<int>(params.noiseEstimateSeconds * 
                                         stftParams.sampleRate / stftParams.hopSize);
    
    // Vector to store noise profile (average magnitude spectrum)
    std::vector<float> noiseProfile(stftParams.frameSize / 2 + 1, 0.0f);
    int noiseFrameCount = 0;

    // First pass: estimate noise from initial frames
    spectral::processSTFT(audio, stftParams,
        [&](spectral::SpectralFrame& frame, int frameIndex, int channel) {
            if (frameIndex < framesForNoise) {
                // Accumulate magnitude spectra
                for (size_t i = 0; i < frame.magnitude.size(); ++i) {
                    noiseProfile[i] += frame.magnitude[i];
                }
                noiseFrameCount++;
            }
        });

    // Average the noise profile
    if (noiseFrameCount > 0) {
        for (auto& val : noiseProfile) {
            val /= noiseFrameCount;
        }
    }

    // Second pass: apply spectral subtraction
    spectral::processSTFT(audio, stftParams,
        [&](spectral::SpectralFrame& frame, int frameIndex, int channel) {
            for (size_t i = 0; i < frame.magnitude.size(); ++i) {
                // Apply over-subtraction and spectral floor
                float subtracted = frame.magnitude[i] - 
                                 params.overSubtractionFactor * noiseProfile[i];
                frame.magnitude[i] = std::max(
                    subtracted,
                    params.spectralFloor * frame.magnitude[i]
                );
            }
        });
}

void applyWienerFilter(AudioData& audio,
                       const spectral::STFTParams& stftParams,
                       const WienerFilterParams& params) {
    // Calculate number of frames for noise estimation
    int framesForNoise = static_cast<int>(params.noiseEstimateSeconds * 
                                         stftParams.sampleRate / stftParams.hopSize);
    
    // Vector to store noise power spectrum
    std::vector<float> noisePower(stftParams.frameSize / 2 + 1, 0.0f);
    int noiseFrameCount = 0;

    // First pass: estimate noise power spectrum
    spectral::processSTFT(audio, stftParams,
        [&](spectral::SpectralFrame& frame, int frameIndex, int channel) {
            if (frameIndex < framesForNoise) {
                // Accumulate power spectra
                for (size_t i = 0; i < frame.magnitude.size(); ++i) {
                    float power = frame.magnitude[i] * frame.magnitude[i];
                    noisePower[i] += power;
                }
                noiseFrameCount++;
            }
        });

    // Average the noise power spectrum
    if (noiseFrameCount > 0) {
        for (auto& val : noisePower) {
            val /= noiseFrameCount;
        }
    }

    // Variables for a priori SNR estimation
    std::vector<float> prevPowerSpectrum(stftParams.frameSize / 2 + 1, 0.0f);
    std::vector<float> prevGain(stftParams.frameSize / 2 + 1, 1.0f);

    // Second pass: apply Wiener filter
    spectral::processSTFT(audio, stftParams,
        [&](spectral::SpectralFrame& frame, int frameIndex, int channel) {
            std::vector<float> currentPowerSpectrum(frame.magnitude.size());
            
            // Calculate current power spectrum
            for (size_t i = 0; i < frame.magnitude.size(); ++i) {
                currentPowerSpectrum[i] = frame.magnitude[i] * frame.magnitude[i];
            }

            // Apply Wiener filter with a priori SNR estimation
            for (size_t i = 0; i < frame.magnitude.size(); ++i) {
                // Estimate a priori SNR using decision-directed approach
                float posteriorSNR = std::max(currentPowerSpectrum[i] / noisePower[i] - 1.0f, 0.0f);
                float priorSNR = params.priorSNR * (prevGain[i] * prevGain[i] * prevPowerSpectrum[i] / noisePower[i]) +
                                (1.0f - params.priorSNR) * posteriorSNR;
                
                // Calculate Wiener filter gain
                float gain = priorSNR / (1.0f + priorSNR);
                
                // Store values for next frame
                prevPowerSpectrum[i] = currentPowerSpectrum[i];
                prevGain[i] = gain;
                
                // Apply gain to magnitude
                frame.magnitude[i] *= gain;
            }
        });
}

std::vector<bool> detectVoiceActivity(const AudioData& audio,
                                    const spectral::STFTParams& stftParams,
                                    float energyThreshold) {
    std::vector<bool> voiceActivity;
    bool isFirstFrame = true;
    float maxEnergy = 0.0f;

    spectral::processSTFT(const_cast<AudioData&>(audio), stftParams,
        [&](spectral::SpectralFrame& frame, int frameIndex, int channel) {
            if (channel > 0) return; // Process only first channel
            
            // Calculate frame energy
            float energy = std::accumulate(frame.magnitude.begin(), 
                                         frame.magnitude.end(), 
                                         0.0f,
                                         [](float sum, float mag) {
                                             return sum + mag * mag;
                                         });
            
            // Normalize by frame size
            energy /= frame.magnitude.size();
            
            // Update max energy (for first pass)
            if (isFirstFrame) {
                maxEnergy = energy;
                isFirstFrame = false;
            } else {
                maxEnergy = std::max(maxEnergy, energy);
            }
            
            // Classify frame as speech/non-speech
            bool isSpeech = energy > (energyThreshold * maxEnergy);
            
            // Only store result once per frame (not per channel)
            if (channel == 0) {
                voiceActivity.push_back(isSpeech);
            }
        });

    return voiceActivity;
}

} // namespace speech_enhance
