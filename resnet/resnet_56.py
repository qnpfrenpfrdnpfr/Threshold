import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from model import resnet

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
pretrained_path = './save56_model/ckpt.pth'  # 프리트레인된 모델 경로 변경
checkpoint = torch.load(pretrained_path)
net = resnet()  # model 함수를 호출하여 모델 객체 생성
net.load_state_dict(checkpoint['net'])
net = net.to(device)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('logs')

def test(epoch, best_acc, total_test_time):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    batch_infer_times = []

    with tqdm(total=len(test_loader), desc=f'Test Epoch {epoch}') as pbar:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 추론 시간 측정 시작
                starter.record()

                outputs = net(inputs)

                # 추론 시간 측정 종료
                ender.record()
                
                if device == 'cuda':
                    torch.cuda.synchronize()  # 모든 GPU 연산이 완료될 때까지 대기

                batch_infer_time = starter.elapsed_time(ender)  # milliseconds
                batch_infer_times.append(batch_infer_time * 1e-3)  # seconds

                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
                pbar.update(1)
    
    epoch_test_time = sum(batch_infer_times)
    total_test_time += epoch_test_time
    
    acc = 100 * correct / total
    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f} | time: {:.2f} seconds'.format(
           epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc, epoch_test_time))

    writer.add_scalar('log/test error', 100 - acc, epoch)
    
    if acc > best_acc:
        print('==> Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_path = './save56_model'  # 모델을 저장할 경로를 동일하게 유지
        os.makedirs(save_path, exist_ok=True)  # 디렉토리 생성
        torch.save(state, os.path.join(save_path, 'ckpt.pth'))
        best_acc = acc

    return best_acc, total_test_time


if __name__=='__main__':
    best_acc = 0
    total_test_time = 0.0
    
    for epoch in range(1, 201):  # 1부터 200까지 200 에포크 실행
        best_acc, total_test_time = test(epoch, best_acc, total_test_time)
    
    print('Total test time: {:.2f} seconds'.format(total_test_time))
    print('Best test accuracy: {:.2f}%'.format(best_acc))