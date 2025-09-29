import torch

print("=== PyTorch Device Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Determine device
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print("MPS (Metal Performance Shaders) is available")
else:
    device = "cpu"
    print("Using CPU")

print(f"\nSelected device: {device}")

# Test with a simple tensor
test_tensor = torch.randn(3, 3)
print(f"\nTest tensor device (default): {test_tensor.device}")

# Move to selected device
test_tensor = test_tensor.to(device)
print(f"Test tensor device (after .to({device})): {test_tensor.device}")

# Test a simple operation
result = test_tensor @ test_tensor.T
print(f"Operation result device: {result.device}")

print(f"\nYou are {'NOT ' if device == 'cpu' else ''}using MPS for training!") 