# %%
from complexnn.bn import ComplexBatchNormalization
from keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, concatenate, LSTM, Add, Multiply, Layer
from keras.layers.merge import subtract
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
import numpy as np
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.utils import plot_model
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from utils import *
import warnings
from losses import categorical_focal_loss
warnings.filterwarnings("ignore", category=FutureWarning)
from complexnn import ComplexBN, ComplexDense


# %%
H = K.constant([[1, 1]])
SNR1 = 7
SNR2 = 7
SNRs = [SNR1, SNR2]
ebno = [calc_ebno(SNR) for SNR in SNRs]

k = 2
n_channel = 2
M = 2 ** k
k = int(k)
R = k / n_channel

#%%
def TransmissionLayer(x, H, R, ebno, t, k):
    if isinstance(x, list):
        ebno[k] = calc_ebno(x[1])
        x = x[0]

    signal = H[t, k] * x

    for i in range(t):
        if i == k:
            continue
        interference = H[i, k] * x
        signal = signal + interference

    
    noise = K.random_normal(K.shape(signal),
                        mean=0,
                        stddev=K.sqrt(1 / (2 * K.variable(R) * ebno[k])))
    return signal + noise
#%%

input_signal1 = Input(shape=(M,), name="input1")

encoder1 = Sequential([
    ComplexDense(M, "tanh"),
    ComplexDense(n_channel, "tanh"),
    ComplexDense(n_channel, "linear"),
    ComplexBatchNormalization(center=False, scale=False),], name="encoder1"
)

signal_input1 = Lambda(TransmissionLayer, arguments={"H": H,"R": R, "ebno": ebno, "t":0, "k":0}, name="transmit1")

