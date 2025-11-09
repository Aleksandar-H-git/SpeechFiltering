import os
import random
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset: mixes clean speech + noise
# -----------------------------
class SpeechDenoiseDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sample_rate=16000, segment_len=32000):
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".wav")]
        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
        self.sample_rate = sample_rate
        self.segment_len = segment_len

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        clean, sr = torchaudio.load(clean_path)
        clean = torchaudio.functional.resample(clean, sr, self.sample_rate)

        noise_path = random.choice(self.noise_files)
        noise, nsr = torchaudio.load(noise_path)
        noise = torchaudio.functional.resample(noise, nsr, self.sample_rate)

        # Trim or pad to segment_len
        clean = clean[:, :self.segment_len]
        noise = noise[:, :self.segment_len]
        if clean.size(1) < self.segment_len:
            clean = torch.nn.functional.pad(clean, (0, self.segment_len - clean.size(1)))
        if noise.size(1) < self.segment_len:
            noise = torch.nn.functional.pad(noise, (0, self.segment_len - noise.size(1)))

        # Random SNR between 0 and 15 dB
        snr_db = random.uniform(0, 15)
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()
        noise = noise * torch.sqrt(clean_power / (noise_power * 10**(snr_db / 10)))

        noisy = clean + noise
        return noisy, clean


# -----------------------------
# Model: 1D Conv Autoencoder
# -----------------------------
class ConvDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(32, 64, 15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(64, 128, 15, stride=2, padding=7), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 15, stride=2, padding=7, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 15, stride=2, padding=7, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 15, stride=2, padding=7, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# -----------------------------
# Training loop
# -----------------------------
def train_model(clean_dir="data/clean", noise_dir="data/noise", epochs=20, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeechDenoiseDataset(clean_dir, noise_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConvDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        total_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            noisy = noisy[:, 0:1, :]  # make sure it’s mono
            clean = clean[:, 0:1, :]

            output = model(noisy)
            loss = criterion(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), "speech_denoiser.pt")
    print("✅ Model saved to speech_denoiser.pt")

if __name__ == "__main__":
    train_model()
