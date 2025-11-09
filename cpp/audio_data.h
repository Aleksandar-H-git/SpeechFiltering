#ifndef AUDIO_DATA_H
#define AUDIO_DATA_H

#include <vector>
#include <cstdint>

struct AudioData {
    std::vector<int16_t> samples;  // Interleaved audio samples
    uint32_t sampleRate;
    uint16_t channels;
    uint16_t bitsPerSample;

    AudioData() : sampleRate(44100), channels(2), bitsPerSample(16) {}
};

#endif // AUDIO_DATA_H