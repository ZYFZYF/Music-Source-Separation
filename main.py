import Utils
import Model
import torch
import numpy as np
import tqdm
import librosa
import time
import os

MAX_ITERATIONS = 1000  # 进行这么多batch的训练
BATCH_SIZE = 4  # 每个batch的大小
NUM_STACKS = 4  # 堆叠沙漏网络的层数
FIRST_DEPTH_CHANNELS = 64  # 沙漏网络第一层的通道数
OUTPUT_CHANNELS = 2  # 整个网络输出的通道数
NEXT_DEPTH_ADD_CHANNELS = 0  # 沙漏网络中每下一层增加的通道数
TRAIN_SAVE_POINT = 50  # 保存点

TEST_STEP = 20  # 测试时
TOTAL_TEST = 800

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('train device: {}'.format(device))


def judge(input, output, predict):
    # 每个堆叠沙漏网络的输出的loss和
    return sum(torch.mean(torch.abs(predict[x].mul(input) - output)) for x in range(NUM_STACKS))


def get_model():
    return Model.StackedHourglassNet(num_stacks=NUM_STACKS, first_depth_channels=FIRST_DEPTH_CHANNELS,
                                     output_channels=OUTPUT_CHANNELS,
                                     next_depth_add_channels=NEXT_DEPTH_ADD_CHANNELS)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


print('the number of parameters in model: {}'.format(get_parameter_number(get_model())))


def train():
    # TODO 这里设成输出为2通道，可以尝试输出一通道效果如何，同时也得更改loss计算方式
    net = get_model()
    net.to(device)
    cnt = 0
    train_data = []
    for _, _, _, left_mag, right_mag, mixed_mag, _, _ in Utils.mir_1k_data_generator(train=True):
        train_data.append((left_mag, right_mag, mixed_mag))
        cnt += 1

    print("-------------------train data loaded----------------------")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_sum = torch.empty(1).to(device)
    print("-------------------begin training...----------------------")
    for i in tqdm.tqdm(range(MAX_ITERATIONS)):
        input = torch.empty(BATCH_SIZE, 1, 512, 64)
        output = torch.empty(BATCH_SIZE, 2, 512, 64)
        # 每个batch都是从所有训练数据中随机得到的
        for j in range(BATCH_SIZE):
            index = np.random.randint(len(train_data))  # np.random.randint()是左闭右开
            left_mag, right_mag, mixed_mag = train_data[index]
            start = np.random.randint(mixed_mag.shape[-1] - 64)
            input[j, 0, :, :] = torch.from_numpy(mixed_mag[:512, start:start + 64])
            output[j, 0, :, :] = torch.from_numpy(left_mag[:512, start:start + 64])
            output[j, 1, :, :] = torch.from_numpy(right_mag[:512, start:start + 64])
            input = input.to(device)
            output = output.to(device)
        optimizer.zero_grad()
        predict = net(input).to(device)
        loss = judge(input, output, predict)
        loss.backward()
        loss_sum += loss
        optimizer.step()
        if (i + 1) % TRAIN_SAVE_POINT == 0:
            torch.save(net.state_dict(), 'Model/checkpoint_{}.pt'.format(i + 1))
            print("loss of {} is {}".format(i + 1, loss_sum / TRAIN_SAVE_POINT))
            loss_sum = 0
    torch.save(net.state_dict(), 'Model/checkpoint_final.pt')
    print("-------------------end training.....----------------------")


def test():
    print('-------------------begin testing.......-------------------')
    # TODO 这里设成输出为2通道，可以尝试输出一通道效果如何，同时也得更改loss计算方式
    net = get_model()
    net.load_state_dict(torch.load('Model/checkpoint_final.pt'))
    net.to(device)
    input = np.empty((BATCH_SIZE, 1, 512, 64), dtype=np.float32)
    gnsdr = 0.
    gsir = 0.
    gsar = 0.
    totalLen = 0.
    pbar = tqdm.tqdm(total=TOTAL_TEST)
    cnt = 0
    for left_origin, right_origin, mix_origin, left_mag, right_mag, mix_mag, max_value, mix_spec_phase in Utils.mir_1k_data_generator(
            train=False):
        srcLen = mix_mag.shape[-1]
        startIndex = 0
        predict_left = np.zeros((512, srcLen), dtype=np.float32)
        predict_right = np.zeros((512, srcLen), dtype=np.float32)
        # 除了第一次和最后一次，其余计算时都要用上上下文的信息，然后只提取中间长度为32的结果作为预测的结果
        start = time.time()
        while startIndex + 64 < srcLen:
            # 四个同时算，理论上能4倍加速
            if startIndex and startIndex + 96 + 64 < srcLen:
                for i in range(4):
                    input[i, 0, :, :] = mix_mag[0:512, startIndex + i * 32:startIndex + i * 32 + 64]
                output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
                for i in range(4):
                    predict_left[:, startIndex + i * 32 + 16: startIndex + i * 32 + 48] = output[i, 0, :, 16:48]
                    predict_right[:, startIndex + i * 32 + 16: startIndex + i * 32 + 48] = output[i, 1, :, 16:48]
                startIndex += 128
            else:
                input[0, 0, :, :] = mix_mag[0:512, startIndex:startIndex + 64]
                output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
                if startIndex == 0:
                    predict_left[:, 0:64] = output[0, 0, :, :]
                    predict_right[:, 0:64] = output[0, 1, :, :]
                else:
                    predict_left[:, startIndex + 16: startIndex + 48] = output[0, 0, :, 16:48]
                    predict_right[:, startIndex + 16:startIndex + 48] = output[0, 1, :, 16:48]
                startIndex += 32
        input[0, 0, :, :] = mix_mag[0:512, srcLen - 64:srcLen]
        output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
        length = srcLen - startIndex - 16
        predict_left[:, startIndex + 16:srcLen] = output[0, 0, :, 64 - length:64]
        predict_right[:, startIndex + 16:srcLen] = output[0, 1, :, 64 - length:64]
        # 有NaN的话<号以及librosa会出问题
        np.nan_to_num(predict_left)
        np.nan_to_num(predict_right)
        predict_left[np.where(predict_left < 0)] = 0
        predict_right[np.where(predict_right < 0)] = 0
        predict_left = predict_left * mix_mag[0:512, :] * max_value
        predict_right = predict_right * mix_mag[0:512, :] * max_value
        predict_left_wav = Utils.to_wav(predict_left, mix_spec_phase[0:512, :])
        predict_right_wav = Utils.to_wav(predict_right, mix_spec_phase[0:512, :])
        predict_left_wav = librosa.resample(predict_left_wav, 8000, 16000)
        predict_right_wav = librosa.resample(predict_right_wav, 8000, 16000)
        nsdr, sir, sar, lens = Utils.bss_eval(mix_origin, left_origin, right_origin, predict_left_wav,
                                              predict_right_wav)
        totalLen = totalLen + lens
        gnsdr = gnsdr + nsdr * lens
        gsir = gsir + sir * lens
        gsar = gsar + sar * lens
        # print(nsdr, sir, sar, lens)
        pbar.update(1)
        cnt += 1
        if cnt % TEST_STEP == 0:
            print('人声GNSDR={:.3f} 人声GSIR={:.3f} 人声GSAR={:.3f} 伴奏GNSDR={:.3f} 伴奏GSIR={:.3f} 伴奏GSAR={:.3f}'.format(
                (gnsdr / totalLen)[0][0],
                (gsir / totalLen)[0][0],
                (gsar / totalLen)[0][0],
                (gnsdr / totalLen)[1][0],
                (gsir / totalLen)[1][0],
                (gsar / totalLen)[1][0]))
            # 顺便把这个输出
            Utils.write_wav(predict_left_wav, 'Samples/{}_accompaniments_predict.wav'.format(cnt // TEST_STEP))
            Utils.write_wav(predict_right_wav, 'Samples/{}_voice_predict.wav'.format(cnt // TEST_STEP))
            Utils.write_wav(left_origin, 'Samples/{}_accompaniments_origin.wav'.format(cnt // TEST_STEP))
            Utils.write_wav(right_origin, 'Samples/{}_voice_origin.wav'.format(cnt // TEST_STEP))
            Utils.write_wav(mix_origin, 'Samples/{}_mixed.wav'.format(cnt // TEST_STEP))
        if cnt == TOTAL_TEST:
            break
    print('人声GNSDR={:.3f} 人声GSIR={:.3f} 人声GSAR={:.3f} 伴奏GNSDR={:.3f} 伴奏GSIR={:.3f} 伴奏GSAR={:.3f}'.format(
        (gnsdr / totalLen)[0][0],
        (gsir / totalLen)[0][0],
        (gsar / totalLen)[0][0],
        (gnsdr / totalLen)[1][0],
        (gsir / totalLen)[1][0],
        (gsar / totalLen)[1][0]))


if __name__ == '__main__':
    # 先把需要的文件夹建出来
    if not os.path.exists('Samples'):
        os.mkdir('Samples')
    if not os.path.exists('Model'):
        os.mkdir('Model')
    if not os.path.exists('Dataset/MIR-1K/Wavfile/'):
        print('Dataset is not prepared')
        exit(0)
    train()
    test()
