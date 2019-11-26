import torch
import torch.nn as nn


class Full(nn.Module):
    def __init__(self, input_dim, output_dim, do_activation=True):
        super(Full, self).__init__()
        if not do_activation:
            self.model = nn.Sequential(nn.Linear(in_features=input_dim, out_features=output_dim))
        else:
            self.model = nn.Sequential(nn.Linear(in_features=input_dim, out_features=output_dim),
                                       nn.BatchNorm1d(num_features=output_dim),  # 原实现中这里用的BatchNorm2d我觉得是有问题的，所以这里用1d
                                       nn.ReLU())

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return self.model(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, do_activation=True):
        super(Conv, self).__init__()
        if not do_activation:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding))
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU())

    def forward(self, x):
        return self.model(x)


# 随着沙漏层数越往里应当channels越多
# TODO 对比不改变channels和对比channels的结果
class Hourglass(nn.Module):
    def __init__(self, depth, channels, next_depth_add_channels):
        super(Hourglass, self).__init__()
        next_depth_channels = channels + next_depth_add_channels
        self.model = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv(channels, next_depth_channels),
                                   Hourglass(depth - 1, next_depth_channels,
                                             next_depth_add_channels) if depth else Conv(
                                       channels, next_depth_channels),
                                   Conv(next_depth_channels, channels),
                                   nn.UpsamplingNearest2d(scale_factor=2))  # TODO 修改此处上采样的方式
        self.skip = Conv(channels, channels)

    def forward(self, x):
        return self.skip(x) + self.model(x)


if __name__ == "__main__":
    full = Full(input_dim=10, output_dim=100)
    print(full)
    conv = Conv(in_channels=200, out_channels=358)
    print(conv)
    hourglass = Hourglass(depth=4, channels=64, next_depth_add_channels=64)
    print(hourglass)
