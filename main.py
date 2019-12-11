import Utils
import Model
import torch
import numpy as np
import tqdm
import librosa
import time
import os
import argparse

MAX_ITERATIONS = 150000  # 进行这么多batch的训练
BATCH_SIZE = 4  # 每个batch的大小

NUM_STACKS = 4  # 堆叠沙漏网络的层数
FIRST_DEPTH_CHANNELS = 64  # 沙漏网络第一层的通道数
OUTPUT_CHANNELS = 2  # 整个网络输出的通道数
NEXT_DEPTH_ADD_CHANNELS = 64  # 沙漏网络中每下一层增加的通道数

TRAIN_SAVE_POINT = 50  # 保存点

TEST_STEP = 20  # 测试时
TOTAL_TEST = 825
# 总共测多少条

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


def get_train_data():
    cnt = 0
    train_data = []
    total_train_num = 0
    train_num = []
    train_pos = []
    for _, _, _, left_mag, right_mag, mixed_mag, _, _ in Utils.mir_1k_data_generator(train=True):
        train_data.append((left_mag, right_mag, mixed_mag))
        total_train_num += mixed_mag.shape[-1] - 64
        train_num.append(mixed_mag.shape[-1] - 64)
        for i in range(mixed_mag.shape[-1] - 64):
            train_pos.append((cnt, i))
        cnt += 1
    # print(train_pos)
    # print(len(train_pos))
    # print(sorted(train_num))
    print('there are {} songs and {} train data'.format(cnt, total_train_num))
    return train_data, train_pos


def train():
    net = get_model()
    net.to(device)
    train_data, train_pos = get_train_data()

    print("-------------------train data loaded----------------------")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_sum = torch.empty(1).to(device)
    print("-------------------begin training...----------------------")
    min_loss = 100000000
    loss_not_update_round = 0
    best_iteration = 0
    loss_sum_sum = 0
    min_loss_average = 1000000000
    loss_average_not_update = 0
    for i in tqdm.tqdm(range(MAX_ITERATIONS)):
        input = torch.empty(BATCH_SIZE, 1, 512, 64)
        output = torch.empty(BATCH_SIZE, 2, 512, 64)
        # 每个batch都是从所有训练数据中随机得到的
        for j in range(BATCH_SIZE):
            index, start = train_pos[np.random.randint(len(train_pos))]  # np.random.randint()是左闭右开
            left_mag, right_mag, mixed_mag = train_data[index]
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
            loss_sum_sum += loss_sum.item()
            if loss_sum_sum / (i + 1) < min_loss_average:
                min_loss_average = loss_sum_sum / (i + 1)
                loss_average_not_update = 0
            else:
                loss_average_not_update += 1
            if loss_sum / TRAIN_SAVE_POINT < min_loss:
                min_loss = loss_sum / TRAIN_SAVE_POINT
                loss_not_update_round = 0
                best_iteration = i + 1
            else:
                loss_not_update_round += 1
            print(
                "loss of {} is {}, loss_not_update_round is {}, best_iteration is {}, loss_average is {}, loss_average_not_update_roung is {}".format(
                    i + 1,
                    loss_sum / TRAIN_SAVE_POINT,
                    loss_not_update_round,
                    best_iteration,
                    loss_sum_sum / (i + 1), loss_average_not_update))
            loss_sum = 0
    torch.save(net.state_dict(), 'Model/checkpoint_infinite.pt')
    print("-------------------end training.....----------------------")


def train_continue(model='Model/checkpoint_23600.pt', origin_iteration=23600):
    net = get_model()
    net.load_state_dict(torch.load(model))
    net.to(device)
    train_data, _ = get_train_data()

    print("-------------------train data loaded----------------------")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_sum = torch.empty(1).to(device)
    print("-------------------begin training...----------------------")
    min_loss = 100000000
    loss_not_update_round = 0
    best_iteration = origin_iteration
    loss_sum_sum = 0
    min_loss_average = 1000000000
    loss_average_not_update = 0
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
            torch.save(net.state_dict(), 'Model/checkpoint_{}.pt'.format(i + 1 + origin_iteration))
            loss_sum_sum += loss_sum.item()
            if loss_sum_sum / (i + 1) < min_loss_average:
                min_loss_average = loss_sum_sum / (i + 1)
                loss_average_not_update = 0
            else:
                loss_average_not_update += 1
            if loss_sum / TRAIN_SAVE_POINT < min_loss:
                min_loss = loss_sum / TRAIN_SAVE_POINT
                loss_not_update_round = 0
                best_iteration = i + 1 + origin_iteration
            else:
                loss_not_update_round += 1
            print(
                "loss of {} is {}, loss_not_update_round is {}, best_iteration is {}, loss_average is {}, loss_average_not_update_roung is {}".format(
                    i + 1 + origin_iteration,
                    loss_sum / TRAIN_SAVE_POINT,
                    loss_not_update_round,
                    best_iteration,
                    loss_sum_sum / (i + 1), loss_average_not_update))
            loss_sum = 0

    torch.save(net.state_dict(), 'Model/checkpoint_final_more.pt')
    print("-------------------end training.....----------------------")


def test(model='Model/checkpoint_final.pt'):
    print('model is {}'.format(model))
    print('-------------------begin testing.......-------------------')
    net = get_model()
    net.load_state_dict(torch.load(model))
    net.to(device)
    input = np.empty((BATCH_SIZE, 1, 512, 64), dtype=np.float32)
    global_normalized_sdr = 0.
    global_sir = 0.
    global_sar = 0.
    total_length = 0.
    progress_bar = tqdm.tqdm(total=TOTAL_TEST)
    cnt = 0
    for left_origin, right_origin, mix_origin, left_mag, right_mag, mix_mag, max_value, mix_spec_phase in Utils.mir_1k_data_generator(
            train=False):
        src_length = mix_mag.shape[-1]
        start_index = 0
        predict_left = np.zeros((512, src_length), dtype=np.float32)
        predict_right = np.zeros((512, src_length), dtype=np.float32)
        # 除了第一次和最后一次，其余计算时都要用上上下文的信息，然后只提取中间长度为32的结果作为预测的结果
        start = time.time()
        while start_index + 64 < src_length:
            # 四个同时算，理论上能4倍加速
            if start_index and start_index + (BATCH_SIZE - 1) * 32 + 64 < src_length:
                for i in range(BATCH_SIZE):
                    input[i, 0, :, :] = mix_mag[0:512, start_index + i * 32:start_index + i * 32 + 64]
                output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
                for i in range(BATCH_SIZE):
                    predict_left[:, start_index + i * 32 + 16: start_index + i * 32 + 48] = output[i, 0, :, 16:48]
                    predict_right[:, start_index + i * 32 + 16: start_index + i * 32 + 48] = output[i, 1, :, 16:48]
                start_index += BATCH_SIZE * 32
            else:
                input[0, 0, :, :] = mix_mag[0:512, start_index:start_index + 64]
                output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
                if start_index == 0:
                    predict_left[:, 0:64] = output[0, 0, :, :]
                    predict_right[:, 0:64] = output[0, 1, :, :]
                else:
                    predict_left[:, start_index + 16: start_index + 48] = output[0, 0, :, 16:48]
                    predict_right[:, start_index + 16:start_index + 48] = output[0, 1, :, 16:48]
                start_index += 32
        input[0, 0, :, :] = mix_mag[0:512, src_length - 64:src_length]
        output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
        length = src_length - start_index - 16
        predict_left[:, start_index + 16:src_length] = output[0, 0, :, 64 - length:64]
        predict_right[:, start_index + 16:src_length] = output[0, 1, :, 64 - length:64]
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
        total_length = total_length + lens
        global_normalized_sdr = global_normalized_sdr + nsdr * lens
        global_sir = global_sir + sir * lens
        global_sar = global_sar + sar * lens
        progress_bar.update(1)
        cnt += 1
        if cnt % TEST_STEP == 0:
            print('人声GNSDR={:.3f} 人声GSIR={:.3f} 人声GSAR={:.3f} 伴奏GNSDR={:.3f} 伴奏GSIR={:.3f} 伴奏GSAR={:.3f}'.format(
                (global_normalized_sdr / total_length)[1][0],
                (global_sir / total_length)[1][0],
                (global_sar / total_length)[1][0],
                (global_normalized_sdr / total_length)[0][0],
                (global_sir / total_length)[0][0],
                (global_sar / total_length)[0][0]))
            # 顺便把这个输出
            Utils.write_wav(predict_left_wav,
                            'Samples/{}_accompaniments_predict_lr0.001_20000.wav'.format(cnt // TEST_STEP))
            Utils.write_wav(predict_right_wav, 'Samples/{}_voice_lr0.001_20000.wav'.format(cnt // TEST_STEP))
            # Utils.write_wav(left_origin, 'Samples/{}_accompaniments_origin.wav'.format(cnt // TEST_STEP))
            # Utils.write_wav(right_origin, 'Samples/{}_voice_origin.wav'.format(cnt // TEST_STEP))
            # Utils.write_wav(mix_origin, 'Samples/{}_mixed.wav'.format(cnt // TEST_STEP))
        if cnt == TOTAL_TEST:
            break
    print('人声GNSDR={:.3f} 人声GSIR={:.3f} 人声GSAR={:.3f} 伴奏GNSDR={:.3f} 伴奏GSIR={:.3f} 伴奏GSAR={:.3f}'.format(
        (global_normalized_sdr / total_length)[1][0],
        (global_sir / total_length)[1][0],
        (global_sar / total_length)[1][0],
        (global_normalized_sdr / total_length)[0][0],
        (global_sir / total_length)[0][0],
        (global_sar / total_length)[0][0]))
    print('-------------------end testing.......---------------------')


def generate(model, wav):
    print('model = {} and wav = {}'.format(model, wav))
    print('-------------------begin generating.......-------------------')
    net = get_model()
    net.load_state_dict(torch.load(model))
    net.to(device)
    input = np.empty((BATCH_SIZE, 1, 512, 64), dtype=np.float32)
    mix_mag, max_value, mix_spec_phase = Utils.load_wav(wav)
    src_len = mix_mag.shape[-1]
    start_index = 0
    predict_left = np.zeros((512, src_len), dtype=np.float32)
    predict_right = np.zeros((512, src_len), dtype=np.float32)
    # 除了第一次和最后一次，其余计算时都要用上上下文的信息，然后只提取中间长度为32的结果作为预测的结果
    progress_bar = tqdm.tqdm(total=src_len)
    start = time.time()
    while start_index + 64 < src_len:
        # 四个同时算，理论上能4倍加速
        if start_index and start_index + (BATCH_SIZE - 1) * 32 + 64 < src_len:
            for i in range(BATCH_SIZE):
                input[i, 0, :, :] = mix_mag[0:512, start_index + i * 32:start_index + i * 32 + 64]
            output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
            for i in range(BATCH_SIZE):
                predict_left[:, start_index + i * 32 + 16: start_index + i * 32 + 48] = output[i, 0, :, 16:48]
                predict_right[:, start_index + i * 32 + 16: start_index + i * 32 + 48] = output[i, 1, :, 16:48]
            start_index += BATCH_SIZE * 32
            progress_bar.update(BATCH_SIZE * 32)
        else:
            input[0, 0, :, :] = mix_mag[0:512, start_index:start_index + 64]
            output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
            if start_index == 0:
                predict_left[:, 0:64] = output[0, 0, :, :]
                predict_right[:, 0:64] = output[0, 1, :, :]
                progress_bar.update(64)
            else:
                predict_left[:, start_index + 16: start_index + 48] = output[0, 0, :, 16:48]
                predict_right[:, start_index + 16:start_index + 48] = output[0, 1, :, 16:48]
                progress_bar.update(32)
            start_index += 32
    input[0, 0, :, :] = mix_mag[0:512, src_len - 64:src_len]
    output = net(torch.from_numpy(input).to(device))[-1].data.cpu().numpy()  # 取最后一个沙漏的输出作为输出，并且要拷贝到内存里
    length = src_len - start_index - 16
    predict_left[:, start_index + 16:src_len] = output[0, 0, :, 64 - length:64]
    predict_right[:, start_index + 16:src_len] = output[0, 1, :, 64 - length:64]
    progress_bar.update(src_len - start_index + 16)
    # 有NaN的话<号以及librosa会出问题
    np.nan_to_num(predict_left)
    np.nan_to_num(predict_right)
    predict_left[np.where(predict_left < 0)] = 0
    predict_right[np.where(predict_right < 0)] = 0
    predict_left = predict_left * mix_mag[0:512, :] * max_value
    predict_right = predict_right * mix_mag[0:512, :] * max_value
    predict_left_wav = Utils.to_wav(predict_left, mix_spec_phase[0:512, :])
    predict_right_wav = Utils.to_wav(predict_right, mix_spec_phase[0:512, :])
    # 只恢复到8000采样率，不考虑原采样率（因为差别过大可能会失真）
    predict_left_wav = librosa.resample(predict_left_wav, 8000, 16000)
    predict_right_wav = librosa.resample(predict_right_wav, 8000, 16000)
    # 输出
    # wav_name = Utils.get_filename_from_path(wav)
    wav_prefix = os.path.splitext(wav)[0]
    print(wav_prefix)
    Utils.write_wav(predict_left_wav, '{}_accompaniments_predict.wav'.format(wav_prefix))
    Utils.write_wav(predict_right_wav, '{}_voice_predict.wav'.format(wav_prefix))
    print('-------------------end generating.......-------------------')


if __name__ == '__main__':
    # 先把需要的文件夹建出来
    for dir in ['Samples', 'Model']:
        if not os.path.exists(dir):
            os.mkdir(dir)
    if not os.path.exists('Dataset/MIR-1K/Wavfile/'):
        print('Dataset is not prepared')
        exit(0)
    # 设置parser的信息
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, help='test pretrained model')
    parser.add_argument('--generate', '-g', '-G', type=str, help='execute a single music source separation task')
    parser.add_argument('--model', '-m', '-M', type=str, help='specify which model to use')
    args = parser.parse_args()
    if args.test:
        test(model=args.test)
    elif args.generate:
        if os.path.isdir(args.generate):
            for src in os.listdir(args.generate):
                if src.endswith('mp3') or src.endswith('ncm'):
                    generate(model=args.model, wav=args.generate + src)
        else:
            generate(model=args.model, wav=args.generate)
    else:
        train()
    # train_continue()
