import numpy as np

from scipy.io import wavfile


def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = normalize(wavfile.read(path)[1][:, 0])
    #  data_mu_law = np.sign(data_) * (np.log(1 + 255 * np.abs(data_)) / np.log(1 + 255))

    # bins 值在 -1 到 +1 之間 ( 取不到 +1 )， 切成長度為 256
    # Ex：(-1, -0.003......0.999)
    # 取值  在[-∞ ,-1) 區間, return 0
    #       在[ -1,1)區間, 在哪個區間的索引，最低為 1
    #       在[1,+∞ )區間，return 1
    bins = np.linspace(-1, 1, 256)
    # 把 bins 的值取出來
    inputs = bins[(np.digitize(data[0:-1], bins, right=False) - 1)].reshape(1, -1, 1)
    # 輸入往前移一個就是要預測的
    targets = (np.digitize(data[1::], bins, right=False) - 1).reshape(1, -1)
    return inputs, targets
