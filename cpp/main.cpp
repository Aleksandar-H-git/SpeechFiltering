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

#include <torch/script.h>
#include <torch/torch.h>
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

static torch::jit::script::Module loadDenoiserModel(const std::string& modelPath) {
    try {
        torch::jit::script::Module module = torch::jit::load(modelPath);
        module.eval();
        std::cout << "✅ Loaded TorchScript model: " << modelPath << "\n";
        return module;
    } catch (const c10::Error& e) {
        std::cerr << "❌ Failed to load model: " << e.what() << "\n";
        throw;
    }
}

// Run model inference on separate channels and write result back into audio.samples
static void runDenoiser(torch::jit::script::Module& model, AudioData& audio) {
    // Handle mono audio
    if (audio.channels == 1) {
        // Convert int16_t samples to float32 normalized [-1,1]
        std::vector<float> floatSamples(audio.samples.size());
        for (size_t i = 0; i < audio.samples.size(); ++i)
            floatSamples[i] = static_cast<float>(audio.samples[i]) / 32768.0f;

        // Create tensor (1, 1, N)
        torch::Tensor input = torch::from_blob(floatSamples.data(), {(int64_t)1, 1, (int64_t)audio.samples.size()}, torch::kFloat32).clone();

        // Run the model
        torch::NoGradGuard noGrad;
        torch::Tensor output = model.forward({input}).toTensor();

        // Convert back to int16_t
        output = output.squeeze().clamp(-1.0, 1.0);
        output = output * 32768.0f;
        auto outVec = output.to(torch::kCPU).to(torch::kInt16).contiguous();

        audio.samples.resize(outVec.numel());
        std::memcpy(audio.samples.data(), outVec.data_ptr<int16_t>(), outVec.numel() * sizeof(int16_t));

        std::cout << "✅ Model inference done (mono, " << audio.samples.size() << " samples)\n";
    }
    else {
        // Handle multi-channel audio: process each channel separately
        size_t samplesPerChannel = audio.samples.size() / audio.channels;
        std::vector<int16_t> processedSamples(audio.samples.size());

        for (uint16_t ch = 0; ch < audio.channels; ++ch) {
            // Extract channel data
            std::vector<float> floatSamples(samplesPerChannel);
            for (size_t i = 0; i < samplesPerChannel; ++i) {
                floatSamples[i] = static_cast<float>(audio.samples[i * audio.channels + ch]) / 32768.0f;
            }

            // Create tensor (1, 1, N) for this channel
            torch::Tensor input = torch::from_blob(floatSamples.data(), {(int64_t)1, 1, (int64_t)samplesPerChannel}, torch::kFloat32).clone();

            // Run the model
            torch::NoGradGuard noGrad;
            torch::Tensor output = model.forward({input}).toTensor();

            // Convert back to int16_t
            output = output.squeeze().clamp(-1.0, 1.0);
            output = output * 32768.0f;
            auto outVec = output.to(torch::kCPU).to(torch::kInt16).contiguous();

            // Write processed channel back to interleaved buffer
            for (int64_t i = 0; i < outVec.numel(); ++i) {
                processedSamples[i * audio.channels + ch] = outVec[i].item<int16_t>();
            }

            std::cout << "✅ Channel " << (ch + 1) << "/" << audio.channels << " inference done\n";
        }

        audio.samples = std::move(processedSamples);
        std::cout << "✅ Model inference completed for all " << audio.channels << " channels\n";
    }
}

// Downsample audio to 16kHz using simple averaging
static void downsampleTo16kHz(AudioData& audio) {
    if (audio.sampleRate == 16000) {
        std::cout << "Audio already at 16kHz, skipping downsampling\n";
        return;
    }

    if (audio.sampleRate < 16000) {
        std::cerr << "Audio sample rate (" << audio.sampleRate << " Hz) is already below 16kHz\n";
        return;
    }

    // Calculate downsampling factor
    int downsamplingFactor = audio.sampleRate / 16000;
    std::vector<int16_t> downsampledSamples;
    downsampledSamples.reserve(audio.samples.size() / downsamplingFactor);

    // Simple averaging downsampling
    for (size_t i = 0; i < audio.samples.size(); i += downsamplingFactor) {
        int32_t sum = 0;
        int count = 0;
        
        for (int j = 0; j < downsamplingFactor && i + j < audio.samples.size(); ++j) {
            sum += audio.samples[i + j];
            count++;
        }
        
        int16_t averaged = static_cast<int16_t>(sum / count);
        downsampledSamples.push_back(averaged);
    }

    // Replace audio samples with downsampled version
    audio.samples = std::move(downsampledSamples);
    audio.sampleRate = 16000;

    std::cout << "Downsampled to 16kHz (" << audio.samples.size() << " samples)\n";
}

//for now, we always use Hybrid
static void processAudioData(AudioData& audio) {
    std::cout << "Processing with mode: ";
    
   
     std::cout << "Hybrid (Time-Domain + SpectralSubtraction + Wiener)\n";
            
            // First: time-domain bandpass filtering
            filters::applySpeechBandpass(audio);
            // Next: Spectral Subtraction
            std::vector<bool> emptyVAD;
            spectral::STFTParams stftParams(2048, 512, audio.sampleRate);
            speech_enhance::SpectralSubtractionParams ssParams;
            ssParams.overSubtractionFactor = 1.5f;
            ssParams.spectralFloor = 0.02f;
            speech_enhance::applySpectralSubtraction(audio, stftParams, ssParams, emptyVAD);
            
            // Finally: Wiener Filtering
            speech_enhance::WienerFilterParams wfParams;
            wfParams.priorSNR = 0.98f;  // Strong temporal smoothing
            wfParams.noiseEstimateSeconds = 0.3f;
            speech_enhance::applyWienerFilter(audio, stftParams, wfParams);

            std::cout << "Processing with TorchScript model (LibTorch)...\n";
            static torch::jit::script::Module denoiser = loadDenoiserModel("python\\speech_denoiser.pt");
            downsampleTo16kHz(audio);
            runDenoiser(denoiser, audio);
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
                    processAudioData(audio/*, mode*/);

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
    processAudioData(audio/*, mode*/);

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
