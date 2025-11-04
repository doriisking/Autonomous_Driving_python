import torch

# CUDA 사용 가능 여부
print("CUDA available:", torch.cuda.is_available())

# GPU 장치 개수
print("GPU count:", torch.cuda.device_count())

# 각 GPU 이름 출력
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# 현재 기본 장치 확인
print("Current device index:", torch.cuda.current_device())
