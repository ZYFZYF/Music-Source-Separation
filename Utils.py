import os
import glob
import librosa
import numpy as np
import soundfile
import mir_eval


def mir_1k_data_generator(train):
    for wav in glob.glob('Dataset/MIR-1K/Wavfile/*.wav'):
        filename = os.path.split(wav)[1]
        if (filename.startswith('abjones') or filename.startswith('amy')) == train:
            origin_source, origin_sr = librosa.load(wav, sr=None, mono=False)  # TODO 这里的采样率，该文中用了8k，实际上原音频是16k
            resample_source = librosa.resample(origin_source, origin_sr, 8000)
            mixed_source_origin = librosa.to_mono(resample_source)
            left_resample_origin = resample_source[0]
            right_resample_origin = resample_source[1]
            # print(np.min(left_source), np.min(right_source), np.min(mixed_source)) TODO 为什么声压有负的？
            mixed_source_magnitude_spectrum = np.abs(to_spectrum(mixed_source_origin))
            left_source_magnitude_spectrum = np.abs(
                to_spectrum(np.asfortranarray(left_resample_origin)))  # 以前是不需要做这一步的来让flags['F_CONTIGUOUS']=True的
            right_source_magnitude_spectrum = np.abs(to_spectrum(np.asfortranarray(right_resample_origin)))

            # 归一化 TODO 可以试试不做归一化会怎么样
            max_value = np.max(mixed_source_magnitude_spectrum)
            mixed_source_magnitude_spectrum = mixed_source_magnitude_spectrum / max_value
            left_source_magnitude_spectrum = left_source_magnitude_spectrum / max_value
            right_source_magnitude_spectrum = right_source_magnitude_spectrum / max_value
            # TODO 原文只用了幅度来做，能不能把相位也加进来
            mixed_spec_phase = np.angle(to_spectrum(mixed_source_origin))

            yield origin_source[0, :], origin_source[1, :], librosa.to_mono(
                origin_source), left_source_magnitude_spectrum, right_source_magnitude_spectrum, mixed_source_magnitude_spectrum, max_value, mixed_spec_phase


def to_spectrum(wav, window_size=1024, hop_size=256):
    return librosa.stft(wav, n_fft=window_size, hop_length=hop_size)


def to_wav(mag, phase, len_hop=256):
    stft_matrix = mag * np.exp(1.j * phase)
    return np.array(librosa.istft(stft_matrix, hop_length=len_hop))


def write_wav(data, path):
    soundfile.write(path, data, 8000, format='wav', subtype='PCM_16')


def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len = pred_src1_wav.shape[0]
    src1_wav = src1_wav[:len]
    src2_wav = src2_wav[:len]
    mixed_wav = mixed_wav[:len]
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(np.array([src1_wav, src2_wav]),
                                                            np.array([pred_src1_wav, pred_src2_wav]))
    sdr_mixed, _, _, _ = mir_eval.separation.bss_eval_sources(np.array([src1_wav, src2_wav]),
                                                              np.array([mixed_wav, mixed_wav]))
    nsdr = sdr - sdr_mixed
    return nsdr, sir, sar, len


if __name__ == "__main__":
    for x, y, z, w, u, v in mir_1k_data_generator(train=True):
        print(len(x), len(y), len(z))
