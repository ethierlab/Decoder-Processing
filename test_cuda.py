import torch
print(torch.__version__)  # Should print 2.6.0+cu126
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU name (e.g., NVIDIA RTX 4090)