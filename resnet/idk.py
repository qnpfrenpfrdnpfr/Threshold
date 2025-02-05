'''import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class MNIST_CNN(nn.Module):
    def __init__(self, threshold=0.1):
        super(MNIST_CNN, self).__init__()
        self.threshold = threshold
        
        # CNN layers
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1채널 입력, 32채널 출력
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32채널 입력, 64채널 출력
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64채널 입력, 128채널 출력
        
        # Adjust fc1 input size based on flattened tensor size
        self.fc1 = nn.Linear(128 * 7 * 7, 128)  # 필요한 크기로 수정합니다.
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1
        out = F.relu(self.layer1(x))
        
        # Apply thresholding only during testing
        if not self.training:
            channel_means = out.mean(dim=(2, 3))  # H, W에 대해 평균 계산
            for i in range(out.size(1)):  # out.size(1) = 채널 수
                mean_val = channel_means[:, i].mean().item()  # 각 채널의 평균값
                if mean_val < self.threshold:
                    out[:, i, :, :] = 0  # 채널을 0으로 설정하여 conv2로 넘기지 않음
        
        # Layer 2
        out = F.relu(self.layer2(out))
        
        # Layer 3
        out = F.relu(self.layer3(out))
        
        # Flatten and fully connected layers
        out = out.view(out.size(0), -1)  # Flatten
        print(out.shape)  # Flatten된 텐서의 크기를 확인
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out


# Hyperparameters and other configurations
batch_size = 64
learning_rate = 0.001
epochs = 10

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
model = MNIST_CNN(threshold=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] Loss: {loss.item():.6f}")

# Testing function with time measurement
def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()  # 시작 시간 측정
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    end_time = time.time()  # 종료 시간 측정
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)")
    print(f"Test time: {end_time - start_time:.4f} seconds\n")  # 테스트 시간 출력

# Training and testing the model
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, criterion, epoch)
    test(model, test_loader, criterion)'''

'''import torch
import torch.nn as nn

# Conv2d 레이어 생성
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

# Conv2d 레이어의 가중치가 위치한 디바이스 확인
print(conv_layer.weight.device)
'''

'''import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conv = nn.Conv2d(3, 16, 3).to(device)
x = torch.randn(1, 3, 224, 224).to(device)

# Profiler 사용
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    output = conv(x)

# 프로파일링 결과 출력
print(prof.key_averages().table(sort_by="self_cuda_time_total" if device.type == 'cuda' else "self_cpu_time_total"))'''


"""import torch.nn as nn
from torch.autograd import profiler
from model import resnet
import os

import torch
# GPU 활성화 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 모델과 데이터를 GPU로 이동
threshold = 0.0295
model = resnet(threshold=threshold).to(device)
x = torch.randn(1, 3, 32, 32).to(device)

# GPU 초기화 (Lazy Initialization 방지)
torch.cuda.synchronize()

# Profiler 사용
print("Profiling 시작...")
with profiler.profile(use_cuda=True) as prof:
    output = model(x)

# 결과 출력
print(prof.key_averages().table(
    sort_by="cuda_time_total",  # GPU 실행 시간 기준으로 정렬
    row_limit=20
))
"""
'''import torch
import torch.nn as nn
from torch.autograd import profiler
from model import resnet

# 모델 불러오기 (기본값은 CPU 사용)
threshold = 0.0295
model = resnet(threshold=threshold)  # device를 명시적으로 설정하지 않음

# 입력 데이터 생성 (기본적으로 CPU에 생성됨)
x = torch.randn(1, 3, 32, 32)

# Profiler 사용
print("Profiling 시작...")
with profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    output = model(x)

# Profiling 결과 출력
print(prof.key_averages().table(
    sort_by="self_cpu_time_total",  # CPU 실행 시간 기준으로 정렬
    row_limit=20  # 상위 20개 연산 출력
))'''

'''import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from model import resnet

# 모델 정의 및 데이터 준비
threshold = 0.0295
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet(threshold=threshold).to(device)
x = torch.randn(1, 3, 32, 32).to(device)

# Profiler 사용
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # CPU 및 GPU 활성화
    record_shapes=True,  # 입력 크기 기록
    with_stack=True      # 스택 추적 기록
) as prof:
    with torch.no_grad():
        output = model(x)

# 프로파일링 결과 출력
print(prof.key_averages().table(
    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
    row_limit=20  # 상위 20개 연산
))

# TensorBoard 추적 파일 저장 (선택 사항)
prof.export_chrome_trace("trace.json")
'''

import torch
from torch.profiler import profile, ProfilerActivity

# 예제 모델과 입력 데이터
x = torch.randn(1, 3, 32, 32).cuda()
conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()

# Profiler로 CPU와 GPU 활동 기록
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    conv(x)
65
# 프로파일링 결과 출력
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

