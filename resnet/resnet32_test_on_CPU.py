import os
import time  # CPU 시간 측정을 위해 time 모듈 사용
import torch
import torch.nn as nn
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from model import resnet

from thop import profile  # thop을 사용하여 FLOPs 계산

# 'cuda' 대신 'cpu'로 설정
device = 'cpu'  # 강제로 CPU에서 테스트

print('==> Preparing data..')
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_test = CIFAR10(root='../data', train=False, download=True, transform=transforms_test)
test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Loading pretrained model..')
pretrained_path = './save_model/ckpt.pth'
checkpoint = torch.load(pretrained_path, map_location=device)  # 모델을 CPU에 로드
net = resnet()  # 모델 함수를 호출하여 모델 객체 생성
net.load_state_dict(checkpoint['net'], strict=False)
net = net.to(device)  # CPU로 모델 이동

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('logs')

# 모델의 총 파라미터 수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# FLOPs 계산 (thop 사용)
def count_flops(model, input_size=(1, 3, 32, 32)):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)  # CPU로 더미 입력 생성
    flops, params = profile(model, inputs=(dummy_input,))
    return flops

def test(epoch, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    start_time = time.time()  # 테스트 시작 시간 기록

    with torch.no_grad():
        with tqdm(test_loader, desc=f"Test Epoch {epoch}") as pbar:  # 에포크당 하나의 tqdm 생성
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # 모델의 forward pass 호출
                outputs = net(inputs)  # [0] 제거
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # tqdm 업데이트
                pbar.set_postfix({
                    'loss': test_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

    end_time = time.time()  # 테스트 종료 시간 기록

    epoch_time = end_time - start_time  # 에폭당 실행 시간 계산
    acc = 100 * correct / total

    print('Test Epoch: {} | Loss: {:.3f} | Acc: {:.3f}% | Epoch Time: {:.2f} seconds'.format(
        epoch, test_loss / (batch_idx + 1), acc, epoch_time))

    writer.add_scalar('log/test error', 100 - acc, epoch)
    writer.add_scalar('log/test_time', epoch_time, epoch)

    if acc > best_acc:
        print('==> Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_path = './save_model'
        os.makedirs(save_path, exist_ok=True)  # 디렉토리 생성
        torch.save(state, os.path.join(save_path, 'ckpt.pth'))
        best_acc = acc

    return best_acc, epoch_time



if __name__ == '__main__':
    best_acc = 0
    num_epochs = 100  # 원하는 에폭 수로 설정
    num_iterations = 1  # 반복 테스트를 하지 않으려면 1로 유지
    test_times = []

    # 모델의 총 파라미터 수 출력
    total_params = count_parameters(net)
    print(f'Model Total Parameters: {total_params}')

    # 모델의 FLOPs 출력 (thop 사용)
    flops = count_flops(net, input_size=(1, 3, 32, 32))
    print(f'Model Total FLOPs: {flops}')

    for epoch in range(num_epochs):
        epoch_test_times = []
        for _ in range(num_iterations):
            best_acc, epoch_time = test(epoch, best_acc)
            epoch_test_times.append(epoch_time)

        average_test_time = sum(epoch_test_times) / len(epoch_test_times)
        test_times.append(average_test_time)

        print('Average test time for epoch {}: {:.2f} seconds'.format(epoch, average_test_time))

    overall_average_time = sum(test_times) / len(test_times)
    print('Overall average test time over all epochs: {:.2f} seconds'.format(overall_average_time))
    print('Best test accuracy is', best_acc)
