import numpy as np

class checkstep:
    def __init__(self, step):
        self.step = step
        self.index = 0

    def __call__(self, *args):
        self.index += 1
        return self.index % self.step != 0


'''信号失真'''


def drop(D, n):
    return D[n:]


'''帧重复'''


def repeat(D, n, axis=0):
    return np.repeat(D, n, axis=axis)


'''频谱位移'''


def roll(D, n):
    return np.roll(D, n, axis=0)


'''边长补偿'''


def rewardshape(D, shape):
    x = shape[0] - D.shape[0]
    y = shape[1] - D.shape[1]
    if x > 0:
        bottomlist = np.zeros([x, D.shape[1]])
        D = np.r_[D, bottomlist]
    if y > 0:
        rightlist = np.zeros([D.shape[0], y])
        D = np.c_[D, rightlist]
    return D


'''步长池化'''


def pool_step(D, step):
    _shape = D.shape
    if step < 2:
        return D
    cs = checkstep(step)
    return rewardshape(np.array(list(filter(cs, D))), _shape)


'''池化'''


def _pool(D, poolsize):
    x = D.shape[1] // poolsize
    restsize = D.shape[1] % poolsize
    if restsize > 0:
        x += 1
        rightlist = np.zeros([D.shape[0], poolsize - restsize])
        D = np.c_[D, rightlist]
    D = D.reshape((-1, poolsize))
    D = D.sum(axis=1).reshape(-1, x)
    return D


'''池化'''


def pool(D, size=(3, 3), shapeed=False):
    _shape = D.shape
    if isinstance(size, tuple):
        if size[1] > 1:
            D = _pool(D, size[1])
        if size[0] > 1:
            D = _pool(D.T, size[0]).T
    elif isinstance(size, int):
        D = _pool(D.T, size).T
    if shapeed:
        D = rewardshape(D, _shape)
    return D


'''传播'''


def spread(D, size=(3, 3)):
    if isinstance(size, tuple):
        if size[0] > 1:
            D = np.repeat(D, size[0], axis=0)
        if size[1] > 1:
            D = np.repeat(D, size[1], axis=1)
        if size[0] * size[1] > 1:
            D = D / (size[0] * size[1])
    elif isinstance(size, int):
        D = np.repeat(D, size, axis=1)
    return D
