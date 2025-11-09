#include "spectral.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// KissFFT includes
extern "C" {
    #include "kiss_fft.h"
}

namespace spectral {

const float PI = 3.14159265358979323846f;

// Apply Hann window to a frame
void applyHannWindow(std::vector<float>& frame) {
    int N = frame.size();
    for (int i = 0; i < N; ++i) {
        float window = 0.5f * (1.0f - std::cos(2.0f * PI * i / (N - 1)));
        frame[i] *= window;
    }
}

// Convert complex FFT output to magnitude and phase
static void complexToMagnitudePhase(const kiss_fft_cpx* fft_out, int fftSize,
                                    std::vector<float>& magnitude,
                                    std::vector<float>& phase) {
    int numBins = fftSize / 2 + 1;
    magnitude.resize(numBins);
    phase.resize(numBins);
    
    for (int i = 0; i < numBins; ++i) {
        float real = fft_out[i].r;
        float imag = fft_out[i].i;
        magnitude[i] = std::sqrt(real * real + imag * imag);
        phase[i] = std::atan2(imag, real);
    }
}

// Convert magnitude and phase back to complex form
static void magnitudePhaseToComplex(const std::vector<float>& magnitude,
                                    const std::vector<float>& phase,
                                    kiss_fft_cpx* fft_out, int fftSize) {
    int numBins = fftSize / 2 + 1;
    
    // Positive frequencies
    for (int i = 0; i < numBins; ++i) {
        fft_out[i].r = magnitude[i] * std::cos(phase[i]);
        fft_out[i].i = magnitude[i] * std::sin(phase[i]);
    }
    
    // Negative frequencies (complex conjugate symmetry for real signals)
    for (int i = numBins; i < fftSize; ++i) {
        int mirror = fftSize - i;
        fft_out[i].r = fft_out[mirror].r;
        fft_out[i].i = -fft_out[mirror].i;
    }
}

// Main STFT processing function
void processSTFT(AudioData& audio, const STFTParams& params, 
                 SpectralProcessor processor) {
    
    if (audio.samples.empty()) return;
    
    int frameSize = params.frameSize;
    int hopSize = params.hopSize;
    int channels = audio.channels;
    int totalSamples = audio.samples.size() / channels;
    
    // Calculate number of frames
    int numFrames = (totalSamples - frameSize) / hopSize + 1;
    if (numFrames <= 0) return;
    
    // Setup FFT
    kiss_fft_cfg fft_cfg = kiss_fft_alloc(frameSize, 0, nullptr, nullptr);
    kiss_fft_cfg ifft_cfg = kiss_fft_alloc(frameSize, 1, nullptr, nullptr);
    
    if (!fft_cfg || !ifft_cfg) {
        if (fft_cfg) kiss_fft_free(fft_cfg);
        if (ifft_cfg) kiss_fft_free(ifft_cfg);
        return;
    }
    
    // Process each channel separately
    for (int ch = 0; ch < channels; ++ch) {
        // Create output buffer (initialize with zeros)
        std::vector<float> output(totalSamples, 0.0f);
        std::vector<float> windowSum(totalSamples, 0.0f);
        
        // Create Hann window
        std::vector<float> window(frameSize);
        for (int i = 0; i < frameSize; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / (frameSize - 1)));
        }
        
        // Buffers for FFT
        std::vector<kiss_fft_cpx> fft_in(frameSize);
        std::vector<kiss_fft_cpx> fft_out(frameSize);
        
        // Process each frame
        for (int frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
            int frameStart = frameIdx * hopSize;
            
            // Extract frame from interleaved audio
            std::vector<float> frame(frameSize, 0.0f);
            for (int i = 0; i < frameSize && (frameStart + i) < totalSamples; ++i) {
                int sampleIdx = (frameStart + i) * channels + ch;
                frame[i] = audio.samples[sampleIdx] / 32768.0f;
            }
            
            // Apply window
            for (int i = 0; i < frameSize; ++i) {
                frame[i] *= window[i];
            }
            
            // Prepare FFT input
            for (int i = 0; i < frameSize; ++i) {
                fft_in[i].r = frame[i];
                fft_in[i].i = 0.0f;
            }
            
            // Perform FFT
            kiss_fft(fft_cfg, fft_in.data(), fft_out.data());
            
            // Convert to magnitude and phase
            SpectralFrame spectralFrame;
            complexToMagnitudePhase(fft_out.data(), frameSize, 
                                   spectralFrame.magnitude, spectralFrame.phase);
            
            // Apply user's spectral processing
            processor(spectralFrame, frameIdx, ch);
            
            // Convert back to complex
            magnitudePhaseToComplex(spectralFrame.magnitude, spectralFrame.phase,
                                   fft_out.data(), frameSize);
            
            // Perform IFFT
            kiss_fft(ifft_cfg, fft_out.data(), fft_in.data());
            
            // Overlap-add with windowing
            for (int i = 0; i < frameSize && (frameStart + i) < totalSamples; ++i) {
                float sample = fft_in[i].r / frameSize; // Normalize
                output[frameStart + i] += sample * window[i];
                windowSum[frameStart + i] += window[i] * window[i];
            }
        }
        
        // Normalize by window sum and write back to audio
        for (int i = 0; i < totalSamples; ++i) {
            if (windowSum[i] > 1e-6f) {
                output[i] /= windowSum[i];
            }
            
            // Clamp and convert back to int16_t
            float sample = output[i];
            if (sample > 0.9999f) sample = 0.9999f;
            if (sample < -0.9999f) sample = -0.9999f;
            
            audio.samples[i * channels + ch] = static_cast<int16_t>(sample * 32767.0f);
        }
    }
    
    // Cleanup
    kiss_fft_free(fft_cfg);
    kiss_fft_free(ifft_cfg);
}

} // namespace spectral