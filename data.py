
import numpy as np
import tifffile
import matplotlib.pyplot as plt


inDir = './data'


def test_data(shape):
    # load data
    X = tifffile.imread(inDir + '/test-volume.tif')

    X = (X/255.0)*2.0 - 1.0
    X = X.astype('float32')

    Xt = np.zeros((X.shape[0], 1, shape[0], shape[1]))
    for i in range(X.shape[0]):
        rx = np.random.randint(0, X.shape[1]-shape[0])
        ry = np.random.randint(0, X.shape[2]-shape[1])
        Xt[i] = X[i, rx:rx+shape[0], ry:ry+shape[1]]
    return Xt


def gen(shape, batch_size):
    # load data
    X = tifffile.imread(inDir + '/train-volume.tif')
    Y = tifffile.imread(inDir + '/train-labels.tif')

    X = (X/255.0)*2.0 - 1.0
    X = X.astype('float32')
    Y = Y/255.0
    Y = Y.astype('float32')

    while True:
        # preall data
        Xb = np.zeros((batch_size, 1, shape[0], shape[1]))
        Yb = np.zeros((batch_size, 1, shape[0], shape[1]))

        count = 0
        while count < batch_size:
            # random sample from X and Y
            rind = np.random.randint(0, 30)
            rx = np.random.randint(0, X.shape[1]-shape[0])
            ry = np.random.randint(0, X.shape[2]-shape[1])
            Xr = X[rind, rx:rx+shape[0], ry:ry+shape[1]]
            Yr = Y[rind, rx:rx+shape[0], ry:ry+shape[1]]

            # augmentation
            gauss_noise = np.random.normal(size=shape)/100.0
            Xr = Xr + gauss_noise

            Xb[count] = Xr
            Yb[count] = Yr
            count += 1

        yield (Xb, Yb)


if __name__ == '__main__':
    shape = (256,256)
    X = tifffile.imread(inDir + '/train-volume.tif')
    Y = tifffile.imread(inDir + '/train-labels.tif')
    X = (X/255.0)*2.0 - 1.0
    X = X.astype('float32')
    Y = Y/255.0
    Y = Y.astype('float32')
    while True:
        rind = np.random.randint(0, 30)
        rx = np.random.randint(0, X.shape[1]-shape[0])
        ry = np.random.randint(0, X.shape[2]-shape[1])
        Xr = X[rind, rx:rx+shape[0], ry:ry+shape[1]]
        Yr = Y[rind, rx:rx+shape[0], ry:ry+shape[1]]

        # augmentation
        gauss_noise = np.random.normal(size=shape)/100.0
        Xr = Xr + gauss_noise

        plt.subplot(1,2,1)
        plt.imshow(Xr)
        plt.subplot(1,2,2)
        plt.imshow(Yr)
        plt.show()