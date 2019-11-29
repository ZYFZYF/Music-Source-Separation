import os
import glob
import librosa
import numpy as np


def mir_1k_data_generator(train):
    for wav in glob.glob('Dataset/MIR-1K/Wavfile/*.wav'):
        filename = os.path.split(wav)[1]
        if (filename.startswith('abjones') or filename.startswith('amy')) == train:
            origin_source, _ = librosa.load(wav, sr=8000, mono=False)  # TODO 这里的采样率，该文中用了8k，实际上原音频是16k
            mixed_source_origin = librosa.to_mono(origin_source)
            left_source_origin = origin_source[0]
            right_source_origin = origin_source[1]
            # print(np.min(left_source), np.min(right_source), np.min(mixed_source)) TODO 为什么声压有负的？
            mixed_source_magnitude_spectrum = np.abs(to_spectrum(mixed_source_origin))
            left_source_magnitude_spectrum = np.abs(
                to_spectrum(np.asfortranarray(left_source_origin)))  # 以前是不需要做这一步的来让flags['F_CONTIGUOUS']=True的
            right_source_magnitude_spectrum = np.abs(to_spectrum(np.asfortranarray(right_source_origin)))

            # 归一化 TODO 可以试试不做归一化会怎么样
            max_value = np.max(mixed_source_magnitude_spectrum)
            mixed_source_magnitude_spectrum = mixed_source_magnitude_spectrum / max_value
            left_source_magnitude_spectrum = left_source_magnitude_spectrum / max_value
            right_source_magnitude_spectrum = right_source_magnitude_spectrum / max_value
            # TODO 原文只用了幅度来做，能不能把相位也加进来

            yield left_source_origin, right_source_origin, mixed_source_origin, left_source_magnitude_spectrum, right_source_magnitude_spectrum, mixed_source_magnitude_spectrum


def to_spectrum(wav, window_size=1024, hop_size=256):
    return librosa.stft(wav, n_fft=window_size, hop_length=hop_size)


if __name__ == "__main__":
    for x, y, z, w, u, v in mir_1k_data_generator(train=True):
        print(len(x), len(y), len(z))
