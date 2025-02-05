import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import time

from model import resnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터 전처리
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 테스트 데이터셋 로드
dataset_test = CIFAR10(root='../data', train=False, download=True, transform=transforms_test)

# CIFAR-10 클래스 이름
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 사전 학습된 모델 로드
print('==> Loading pretrained model..')
pretrained_path = './save_model/ckpt.pth'
checkpoint = torch.load(pretrained_path)
net = resnet()  # 모델 객체 생성
net.load_state_dict(checkpoint['net'])
net = net.to(device)

def test_single_cifar_image(index):
    """CIFAR-10 데이터셋에서 단일 이미지를 테스트하고 시간을 측정하는 함수"""
    net.eval()

    # 데이터셋에서 이미지를 가져오기
    img, label = dataset_test[index]
    img = img.unsqueeze(0).to(device)  # 배치 차원을 추가하고, GPU로 전송

    # 평가 시간 측정 시작
    start_time = time.time()

    # 예측 수행
    with torch.no_grad():
        output = net(img)
        _, predicted = output.max(1)

    # 평가 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 결과 출력
    predicted_class = classes[predicted.item()]
    true_class = classes[label]
    correct = predicted.item() == label

    print(f"True Class: {true_class}, Predicted Class: {predicted_class}")
    print(f"Correct Prediction: {correct}")
    print(f"Accuracy: {100.0 if correct else 0.0}%")
    print(f"Time taken for evaluation: {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    index = 0  # 테스트할 이미지의 인덱스 (0부터 9999까지 가능)
    print(f"Testing CIFAR-10 image at index {index}:")
    test_single_cifar_image(index)
