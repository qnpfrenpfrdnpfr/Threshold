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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
checkpoint = torch.load(pretrained_path)
net = resnet()  # 모델 함수를 호출하여 모델 객체 생성
net.load_state_dict(checkpoint['net'], strict=False)
net = net.to(device)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('logs')

# 모델의 총 파라미터 수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# FLOPs 계산 (thop 사용)
def count_flops(model, input_size=(1, 3, 32, 32)):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops

def test(epoch, best_acc):
    net.eval()
  
    test_loss = 0
    correct = 0
    total = 0

    batch_infer_times = []

    with tqdm(total=len(test_loader), desc=f'Test Epoch {epoch}') as pbar:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 추론 시간 측정 시작 (CPU 시간)
                start_time = time.time()

                outputs = net(inputs)

                # 추론 시간 측정 종료 (CPU 시간)
                end_time = time.time()

                # 추론 시간 계산
                batch_infer_time = end_time - start_time
                batch_infer_times.append(batch_infer_time)

                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
                pbar.update(1)
    
    epoch_infer_time = sum(batch_infer_times)
    
    acc = 100 * correct / total
    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f} | time: {:.2f} seconds'.format(
           epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc, epoch_infer_time))

    writer.add_scalar('log/test error', 100 - acc, epoch)
    writer.add_scalar('log/test_time', epoch_infer_time, epoch)
    
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

    return best_acc, epoch_infer_time


if __name__=='__main__':
    best_acc = 0
    num_epochs = 200  # 원하는 에폭 수로 설정
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
            best_acc, epoch_infer_time = test(epoch, best_acc)
            epoch_test_times.append(epoch_infer_time)
        
        average_test_time = sum(epoch_test_times) / len(epoch_test_times)
        test_times.append(average_test_time)

        print('Average test time for epoch {}: {:.2f} seconds'.format(epoch, average_test_time))
    
    overall_average_time = sum(test_times) / len(test_times)
    print('Overall average test time over all epochs: {:.2f} seconds'.format(overall_average_time))
    print('Best test accuracy is', best_acc)