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


# 这里隐藏了一个条件就是卷积层过后每个feature map大小不变，所以要根据kernel_size和stride的值来自己设置padding
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, do_activation=True):
        super(Conv, self).__init__()
        if not do_activation:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2),
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
                                       next_depth_channels, next_depth_channels),
                                   Conv(next_depth_channels, channels),
                                   nn.UpsamplingNearest2d(scale_factor=2))  # TODO 修改此处上采样的方式
        self.skip = Conv(channels, channels)

    def forward(self, x):
        # print(self.skip(x).size())
        # print(self.model(x).size())
        return self.skip(x) + self.model(x)


class StackedHourglassNet(nn.Module):
    def __init__(self, num_stacks, input_channels, output_channels, next_depth_add_channels):
        super(StackedHourglassNet, self).__init__()
        self.num_stacks = num_stacks
        self.prepare = nn.Sequential(Conv(1, 64, kernel_size=7, stride=1),
                                     # TODO 这里PosNet是stride为3，padding为2，之后可以试试，先保持和SHNet一样
                                     Conv(64, 128),  # 这里对于原来PosNet来说少了个下采样MaxPooling，我觉得是合理的
                                     Conv(128, 128),
                                     Conv(128, input_channels))
        # 堆叠沙漏模块
        self.hourglass = nn.ModuleList(
            nn.Sequential(Hourglass(depth=4, channels=input_channels, next_depth_add_channels=next_depth_add_channels),
                          Conv(input_channels, input_channels),
                          Conv(input_channels, input_channels, 1)) for i in range(num_stacks))
        # 输出Mask的上面的path
        self.output = nn.ModuleList(Conv(input_channels, output_channels) for i in range(num_stacks))
        # 输向下一个沙漏的features的下面的path
        self.next = nn.ModuleList(Conv(input_channels, input_channels) for i in range(num_stacks - 1))
        # 合并上面和下面，所以这里相当于下一个网络在拟合残差
        self.merge = nn.ModuleList(Conv(output_channels, input_channels) for i in range(num_stacks - 1))

    def forward(self, x):
        x = self.prepare(x)
        # 要输出多个堆叠网络的结果
        predicts = []
        for i in range(self.num_stacks):
            x = self.hourglass[i](x)
            predicts.append(self.output[i](x))
            if i != self.num_stacks - 1:
                x = self.merge[i](predicts[-1]) + self.next[i](x)
        # 因为第一维是batch
        return torch.stack(predicts, 1)


if __name__ == "__main__":
    full = Full(input_dim=10, output_dim=100)
    print(full)
    conv = Conv(in_channels=200, out_channels=358)
    print(conv)
    hourglass = Hourglass(depth=4, channels=64, next_depth_add_channels=64)
    print(hourglass)
    input = torch.rand(3, 64, 128, 128)
    output = hourglass(input)
    print(input)
    print(output)
    stackedHourglassNet = StackedHourglassNet(num_stacks=4, input_channels=64, output_channels=2,
                                              next_depth_add_channels=64)
    print(stackedHourglassNet)
    input = torch.rand(3, 1, 128, 128)
    output = stackedHourglassNet(input)
    print(input)
    print(input.size())
    print(output.size())
