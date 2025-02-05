#Time_jeanungeo
import torch
import torch.nn as nn
import time

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()
        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = nn.functional.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out

class ResidualBlock(nn.Module):
    #total_conv1_time = 0.0  # conv1 총 시간 저장 변수
    #total_conv2_time = 0.0  # conv2 총 시간 저장 변수

    def __init__(self, in_channels, out_channels, stride=1, down_sample=False, device='cpu', threshold=0.2987):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.threshold = threshold

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        device = x.device
        shortcut = x.to(device)

        # conv1 연산 시간 측정
        #start_time = time.time()
        out = self.conv1(x)
        #conv1_time = time.time() - start_time
        #ResidualBlock.total_conv1_time += conv1_time  # 누적 conv1 시간 기록

        out = self.bn1(out)
        out = self.relu(out)

        # 채널별 평균 활성화 값 계산
        channel_means = out.mean(dim=(2, 3))  # H, W에 대해 평균 계산

        # threshold보다 작은 평균값을 가진 채널을 0으로 설정
        for i in range(out.size(1)):  # 채널 수만큼 반복 (8개 채널)
            condition = (channel_means[:, i] < self.threshold).unsqueeze(1).unsqueeze(2)
            out[:, i, :, :] = torch.where(condition, torch.zeros_like(out[:, i, :, :]), out[:, i, :, :])


        # conv2 연산 시간 측정
        #start_time = time.time()
        out = self.conv2(out)  # threshold보다 큰 활성화 값을 가진 채널만 conv2로 넘어감
        #conv2_time = time.time() - start_time
        #ResidualBlock.total_conv2_time += conv2_time  # 누적 conv2 시간 기록

        out = self.bn2(out)

        # If downsampling, modify shortcut
        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10, device='cpu', threshold=0.2987):
        super(ResNet, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layers_2n = self.get_layers(block, 16, 16, stride=1, threshold=threshold)
        self.layers_4n = self.get_layers(block, 16, 32, stride=2, threshold=threshold)
        self.layers_6n = self.get_layers(block, 32, 64, stride=2, threshold=threshold)

        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(64, num_classes)

    def get_layers(self, block, in_channels, out_channels, stride, threshold):
        down_sample = stride == 2
        layers_list = nn.ModuleList([block(in_channels, out_channels, stride, down_sample, device=self.device, threshold=threshold)])
        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels, device=self.device, threshold=threshold))
        return nn.Sequential(*layers_list)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x


def resnet(device='cpu', threshold=0.0295):
    block = ResidualBlock
    model = ResNet(5, block, device=device, threshold=threshold)
    return model