import torch
import torchvision
import torchaudio

print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("🧩 PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())

    # Quick test computation on GPU
    x = torch.randn(10000, 10000, device="cuda")
    y = torch.randn(10000, 10000, device="cuda")
    z = torch.matmul(x, y)
    print("Computation test OK — GPU is working ✅")

else:
    print("⚠️ No GPU detected — training will run on CPU.")