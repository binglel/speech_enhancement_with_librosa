import librosa
import utils

def pitch(n, y, sr):
    return librosa.effects.pitch_shift(y, sr, n_steps=n)


def speed(n, y, sr):
    return librosa.effects.time_stretch(y, n)


def sample(n, y, sr):
    return librosa.resample(y, sr, int(sr // n))


def reback(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool(D, size=(1, n))
    D = utils.repeat(D, n)
    return librosa.istft(D)


def iron(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.drop(D, n)
    return librosa.istft(D)


def quality(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.roll(D, n)
    return librosa.istft(D)


def shrink(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool(D, n, True)
    return librosa.istft(D)


def shrinkstep(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool_step(D, n)
    return librosa.istft(D)


def spread(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.spread(D, n)
    return librosa.istft(D)


def vague(n, y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = utils.pool(D, (1, n))
    D = utils.spread(D, (1, n))
    return librosa.istft(D)


if __name__ == '__main__':
    outputpath = "./data/A2_1_se.wav"
    inputpath = "./data/A2_1.wav"
    y, sr = librosa.load(inputpath,sr=None)
    print (y,sr)

    # 通过 librosa.ifgram方法获取stft短时傅立叶变换的矩阵,D为stft变换的矩阵.
    frequencies, D = librosa.ifgram(y, sr=sr)
    print (frequencies.shape)
    print (D.shape)

    # 模糊
    # y = vague(6, y, sr)

    # 音调或童声
    # y = pitch(6, y, sr)

    # 语速
    # y = speed(6, y, sr)

    # 重采样
    # y = sample(6, y, sr)

    # 回音
    # y = reback(5, y, sr)

    # 失真
    # y = iron(6, y, sr)

    # 位移
    # y = quality(6, y, sr)
    # y = quality(80, y, sr)

    # 池化
    # y = shrink(3, y, sr)

    # 步长池化
    # y = shrinkstep(10, y, sr)

    # 传播或语速
    # y = spread(3, y, sr)

    librosa.output.write_wav(outputpath, y, sr)