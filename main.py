import Utils
import Model
import torch
import numpy as np
import tqdm

MAX_ITERATIONS = 100  # 进行这么多batch的训练
BATCH_SIZE = 4  # 每个batch的大小
STACKED_LEVEL = 2  # 堆叠沙漏网络的层数
SAVE_POINT = 1  # 保存点


def judge(input, output, predict):
    los = torch.empty(1)
    for j in range(STACKED_LEVEL):
        los += torch.mean(torch.abs(predict[j].mul(input) - output))
    return los


def train():
    cnt = 0
    train_data = []
    for _, _, _, left_mag, right_mag, mixed_mag in Utils.mir_1k_data_generator(train=True):
        train_data.append((left_mag, right_mag, mixed_mag))
        cnt += 1
        if cnt == 10:
            break
    print("-------------------train data loaded----------------------")
    # TODO 这里设成输出为2通道，可以尝试输出一通道效果如何，同时也得更改loss计算方式
    net = Model.StackedHourglassNet(num_stacks=2, first_depth_channels=64, output_channels=2, next_depth_add_channels=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    loss_sum = torch.empty(1)
    print("-------------------begin training...----------------------")
    for i in tqdm.tqdm(range(MAX_ITERATIONS)):
        input = torch.randn(BATCH_SIZE, 1, 512, 64)
        output = torch.randn(BATCH_SIZE, 2, 512, 64)
        # 每个batch都是从所有训练数据中随机得到的
        for j in range(BATCH_SIZE):
            index = np.random.randint(len(train_data))  # np.random.randint()是左闭右开
            left_mag, right_mag, mixed_mag = train_data[index]
            start = np.random.randint(mixed_mag.shape[-1] - 64)
            input[j, 0, :, :] = torch.from_numpy(mixed_mag[:512, start:start + 64])
            output[j, 0, :, :] = torch.from_numpy(left_mag[:512, start:start + 64])
            output[j, 1, :, :] = torch.from_numpy(right_mag[:512, start:start + 64])
        optimizer.zero_grad()
        predict = net(input)
        loss = judge(input, output, predict)
        loss.backward()
        loss_sum += loss
        optimizer.step()
        if i % SAVE_POINT == SAVE_POINT - 1:
            print("loss of {} is {}".format(i, loss_sum / SAVE_POINT))
    print("-------------------end training.....----------------------")


if __name__ == '__main__':
    train()
