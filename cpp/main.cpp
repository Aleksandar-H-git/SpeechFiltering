#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <cstring>

#define MINIMP3_IMPLEMENTATION
#include "../external/minimp3/minimp3.h"

#include "audio_data.h"
#include "filters.h"
#include "spectral.h"
#include "speech_enhance.h"

// WAV file structures
#pragma pack(push, 1)
struct WAVHeader {
    // RIFF Header
    char riffHeader[4];     // Contains "RIFF"
    uint32_t wavSize;       // Size of the wav portion of the file
    char waveHeader[4];     // Contains "WAVE"
    
    // Format Header
    char formatHeader[4];   // Contains "fmt "
    uint32_t formatLength;
    uint16_t formatType;    // 1 for PCM
    uint16_t channels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    
    // Data Header
    char dataHeader[4];     // Contains "data"
    uint32_t dataBytes;
};
#pragma pack(pop)


static std::string toLower(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
    return out;
}

static std::string getExtension(const std::string &path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return std::string();
    return toLower(path.substr(pos + 1));
}


enum class ProcessingMode {
    TimeDomain,           // Original bandpass filtering
    SpectralSubtraction,  // FFT-based spectral subtraction
    WienerFilter,         // FFT-based Wiener filtering
    Hybrid                // Time-domain pre-filter + spectral processing
};

// Replace the existing processAudioData function with this:
static void processAudioData(AudioData& audio, ProcessingMode mode = ProcessingMode::Hybrid) {
    std::cout << "Processing with mode: ";
    
    switch(mode) {
        case ProcessingMode::TimeDomain:
            std::cout << "Time-Domain Bandpass\n";
            filters::applySpeechBandpass(audio);
            break;
            
        case ProcessingMode::SpectralSubtraction: {
            std::cout << "Spectral Subtraction\n";
            spectral::STFTParams stftParams(2048, 512, audio.sampleRate);
            speech_enhance::SpectralSubtractionParams ssParams;
            ssParams.overSubtractionFactor = 2.0f;  // More aggressive
            ssParams.spectralFloor = 0.02f;
            speech_enhance::applySpectralSubtraction(audio, stftParams, ssParams);
            break;
        }
        
        case ProcessingMode::WienerFilter: {
            std::cout << "Wiener Filter\n";
            spectral::STFTParams stftParams(2048, 512, audio.sampleRate);
            speech_enhance::WienerFilterParams wfParams;
            speech_enhance::applyWienerFilter(audio, stftParams, wfParams);
            break;
        }
        
        case ProcessingMode::Hybrid: {
            std::cout << "Hybrid (Spectral + Wiener + Time-Domain)\n";

            spectral::STFTParams stftParams(2048, 512, audio.sampleRate);
            speech_enhance::SpectralSubtractionParams ssParams;
            ssParams.overSubtractionFactor = 1.5f;
            ssParams.spectralFloor = 0.02f;
            speech_enhance::applySpectralSubtraction(audio, stftParams, ssParams);
            
            // Second: Wiener filter for additional noise reduction and speech enhancement
            speech_enhance::WienerFilterParams wfParams;
            wfParams.priorSNR = 0.98f;  // Strong temporal smoothing
            wfParams.noiseEstimateSeconds = 0.3f;
            speech_enhance::applyWienerFilter(audio, stftParams, wfParams);
            
            filters::applySpeechBandpass(audio);

            break;
        }
    }
}

// Write WAV file from AudioData
static bool writeWAVFile(const std::string& filename, const AudioData& audio) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to create output WAV file: " << filename << "\n";
        return false;
    }

    WAVHeader header = {};
    std::memcpy(header.riffHeader, "RIFF", 4);
    std::memcpy(header.waveHeader, "WAVE", 4);
    std::memcpy(header.formatHeader, "fmt ", 4);
    std::memcpy(header.dataHeader, "data", 4);
    
    header.formatLength = 16;
    header.formatType = 1; // PCM
    header.channels = audio.channels;
    header.sampleRate = audio.sampleRate;
    header.bitsPerSample = audio.bitsPerSample;
    header.blockAlign = header.channels * header.bitsPerSample / 8;
    header.byteRate = header.sampleRate * header.blockAlign;
    
    header.dataBytes = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    header.wavSize = header.dataBytes + 36; // Size of entire file - 8
    
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));
    ofs.write(reinterpret_cast<const char*>(audio.samples.data()), header.dataBytes);
    
    return true;
}

// Process WAV file and return audio data
static bool processWAV(const std::string &input, const std::string &output, ProcessingMode mode = ProcessingMode::Hybrid) {
    std::ifstream ifs(input, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open input WAV: " << input << "\n";
        return false;
    }

    // Read and verify RIFF header
    char header[4];
    if (!ifs.read(header, 4) || std::string(header, 4) != "RIFF") {
        std::cerr << "Not a valid WAV file (missing RIFF header)\n";
        return false;
    }

    // Skip file size
    uint32_t fileSize;
    if (!ifs.read(reinterpret_cast<char*>(&fileSize), 4)) {
        std::cerr << "Failed to read WAV file size\n";
        return false;
    }

    // Verify WAVE header
    if (!ifs.read(header, 4) || std::string(header, 4) != "WAVE") {
        std::cerr << "Not a valid WAV file (missing WAVE header)\n";
        return false;
    }

    // Find fmt chunk
    while (ifs.read(header, 4)) {
        uint32_t chunkSize;
        if (!ifs.read(reinterpret_cast<char*>(&chunkSize), 4)) {
            std::cerr << "Failed to read chunk size\n";
            return false;
        }

        if (std::string(header, 4) == "fmt ") {
            // Read format data
            if (chunkSize < 16) {
                std::cerr << "Invalid fmt chunk size\n";
                return false;
            }

            AudioData audio;
            uint16_t formatType;
            if (!ifs.read(reinterpret_cast<char*>(&formatType), 2) ||
                !ifs.read(reinterpret_cast<char*>(&audio.channels), 2) ||
                !ifs.read(reinterpret_cast<char*>(&audio.sampleRate), 4)) {
                std::cerr << "Failed to read WAV format\n";
                return false;
            }

            // Skip byte rate and block align
            uint32_t byteRate;
            uint16_t blockAlign;
            if (!ifs.read(reinterpret_cast<char*>(&byteRate), 4) ||
                !ifs.read(reinterpret_cast<char*>(&blockAlign), 2) ||
                !ifs.read(reinterpret_cast<char*>(&audio.bitsPerSample), 2)) {
                std::cerr << "Failed to read WAV format\n";
                return false;
            }

            // Skip any extra format bytes
            if (chunkSize > 16) {
                ifs.seekg(chunkSize - 16, std::ios::cur);
            }

            // Find data chunk
            while (ifs.read(header, 4)) {
                if (!ifs.read(reinterpret_cast<char*>(&chunkSize), 4)) {
                    std::cerr << "Failed to read chunk size\n";
                    return false;
                }

                if (std::string(header, 4) == "data") {
                    // Read the actual audio data
                    size_t numSamples = chunkSize / sizeof(int16_t);
                    audio.samples.resize(numSamples);

                    if (!ifs.read(reinterpret_cast<char*>(audio.samples.data()), chunkSize)) {
                        std::cerr << "Failed to read WAV data\n";
                        return false;
                    }

                    // Process the audio data
                    processAudioData(audio, mode);

                    // Write processed data to output WAV
                    if (!writeWAVFile(output, audio)) {
                        return false;
                    }

                    std::cout << "Processed WAV file saved to: " << output << "\n";
                    std::cout << "Details: " << audio.sampleRate << " Hz, " 
                             << audio.channels << " channels, "
                             << audio.samples.size() << " samples\n";
                    return true;
                }

                // Skip this chunk if it's not "data"
                ifs.seekg(chunkSize, std::ios::cur);
            }
            std::cerr << "No data chunk found in WAV file\n";
            return false;
        }
        // Skip this chunk if it's not "fmt "
        ifs.seekg(chunkSize, std::ios::cur);
    }

    std::cerr << "No fmt chunk found in WAV file\n";
    return false;
}

static bool processMP3(const std::string &input, const std::string &output, ProcessingMode mode = ProcessingMode::Hybrid) {
    // Read whole file
    std::ifstream ifs(input, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "Failed to open input MP3: " << input << "\n";
        return false;
    }
    auto file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<uint8_t> mp3buf(file_size);
    ifs.read(reinterpret_cast<char*>(mp3buf.data()), file_size);

    mp3dec_t dec;
    mp3dec_init(&dec);

    AudioData audio;
    audio.samples.reserve(4096);

    size_t offset = 0;
    mp3dec_frame_info_t info;
    const int MAX_SAMPLES = 2304; // safe upper bound for MP3 frame samples * channels
    std::vector<int16_t> outbuf(MAX_SAMPLES);

    while (offset < mp3buf.size()) {
        int samples = mp3dec_decode_frame(&dec, mp3buf.data() + offset, static_cast<int>(mp3buf.size() - offset), outbuf.data(), &info);
        if (info.frame_bytes <= 0) break;
        offset += info.frame_bytes;
        if (samples > 0) {
            // info.channels indicates channels, samples is number of samples per channel
            // minimp3 returns interleaved samples when using int16_t buffer
            audio.samples.insert(audio.samples.end(), outbuf.begin(), outbuf.begin() + samples * info.channels);
            audio.sampleRate = info.hz;
            audio.channels = info.channels;
        }
    }

    if (audio.samples.empty()) {
        std::cerr << "No PCM samples decoded from MP3\n";
        return false;
    }

    // Process the audio data through our pipeline
    processAudioData(audio, mode);

    // Write processed data to output WAV
    if (!writeWAVFile(output, audio)) {
        return false;
    }

    std::cout << "Processed MP3 saved to: " << output << " (" << audio.sampleRate << " Hz, " << audio.channels << " channels)\n";
    return true;
}


int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <input.(wav|mp3)> [mode]\n";
        std::cerr << "Output will be saved as output.wav\n";
        std::cerr << "Available modes:\n";
        std::cerr << "  time     - Time-domain bandpass filtering only\n";
        std::cerr << "  spectral - Spectral subtraction\n";
        std::cerr << "  wiener   - Wiener filtering\n";
        std::cerr << "  hybrid   - Time-domain + spectral processing (default)\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output = "output.wav";
    
    // Parse processing mode
    ProcessingMode mode = ProcessingMode::Hybrid; // Default mode
    if (argc == 3) {
        std::string modeStr = toLower(argv[2]);
        if (modeStr == "time") {
            mode = ProcessingMode::TimeDomain;
        } else if (modeStr == "spectral") {
            mode = ProcessingMode::SpectralSubtraction;
        } else if (modeStr == "wiener") {
            mode = ProcessingMode::WienerFilter;
        } else if (modeStr == "hybrid") {
            mode = ProcessingMode::Hybrid;
        } else {
            std::cerr << "Unknown processing mode: " << argv[2] << "\n";
            return 1;
        }
    }
    
    std::string ext = getExtension(input);
    if (ext == "wav") {
        if (!processWAV(input, output, mode)) return 2;
    } else if (ext == "mp3") {
        if (!processMP3(input, output, mode)) return 3;
    } else {
        std::cerr << "Unsupported input extension: " << ext << ". Only wav and mp3 are supported.\n";
        return 4;
    }
    return 0;
}
