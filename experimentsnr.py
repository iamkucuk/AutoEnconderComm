# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, concatenate, LSTM, Add, Multiply, Layer
from keras.layers.merge import subtract, add
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

input_signal1 = Input(shape=(M,), name="input1")

encoder1 = create_encoder([M, n_channel], name="encoder1", activations=["tanh", "linear"])
signal_input1 = create_inputs(R=R, H=H, t=0, k=0, ebno=ebno, name="transmit1")
decoder1 = create_decoder([M, M, M], name="decoder1", activation="relu")

x1 = encoder1(input_signal1)
x1 = signal_input1(x1)
out1 = decoder1(x1)

model1 = Model(inputs=input_signal1, outputs=out1)

model1.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", BER])

train_datas = generate_train_datas(k=1)

model1.fit(x=train_datas, y=train_datas, validation_split=.5, batch_size=10000,
    epochs=2000, callbacks=[
    EarlyStopping(patience=100, restore_best_weights=True, monitor="BER", mode="min"),
    ReduceLROnPlateau(monitor="loss", factor=.7, patience=20),
])

# # %%
input_signal1 = Input(shape=(M,), name="input1")
input_signal2 = Input(shape=(M,), name="input2")
# encoder1 = create_encoder([M, n_channel], name="encoder1", activations=["relu", "linear"])
# encoder2 = create_encoder([M, n_channel], name="encoder2", activations=["relu", "linear"])

# combiner = create_combiner(layer_sizes=[n_channel * 4, n_channel], activations=["relu", "tanh"], name="combiner")

signal_input2 = create_inputs(R=R, H=H, t=0, k=1, ebno=ebno, name="transmit2")
signal_input1 = create_inputs(R=R, H=H, t=0, k=0, ebno=ebno, name="transmit1")

decoder1 = create_decoder([M * 4, M * 2, M], name="decoder1", activation="relu")

decoder2 = create_decoder([M * 4, M * 2, M], name="decoder2", activation="relu")

# x1 = encoder1(input_signal1)
# x2 = encoder1(input_signal2)
# x = concatenate([x1, x2], axis=1)
# x = combiner(x)
# x1 = signal_input1(x)
# x2 = signal_input2(x)
# out1 = decoder1(x1)
# out2 = decoder2(x2)


# # %%
# model = Model(inputs=[input_signal1, input_signal2], outputs=[out1, out2])
# model.summary()
# alpha = K.variable(.5)

# optim = SGD(momentum=.9)

# model.compile(optimizer="rmsprop", loss=["categorical_crossentropy", "categorical_crossentropy"],
#             loss_weights=[alpha, (1 - alpha)], metrics=["accuracy", BER])


# # %%
# train_datas = generate_train_datas(k=2)

# model.fit(x=train_datas, y=train_datas, validation_split=.5, batch_size=40000,
#     epochs=2000, callbacks=[
#     EarlyStopping(patience=20, restore_best_weights=True, monitor="loss", mode="min"),
#     ReduceLROnPlateau(monitor="loss", factor=.5, patience=20),
#     AlphaCallback(alpha),
# ])


# %%
oscilation = 5


normalizer = Lambda(lambda x: x)
# encoder1.trainable = False
# encoder2.trainable = False

input_snr1 = Input(shape=(1,), name="snr_input1")
input_snr2 = Input(shape=(1,), name="snr_input2")
#diff_snr = Input(shape=(1,), name="snr_input2")

combiner = create_combiner(layer_sizes=[n_channel*4, n_channel*2, n_channel], dropout_prob=0, activations=["relu", "linear"], name="combiner")

x1 = encoder1(input_signal1)
x2 = encoder1(input_signal2)
diff_snr = subtract([input_snr2, input_snr1])
summed = add([x1, x2])
subtracted = subtract([x2, x1])
x = concatenate([x1, x2, summed, subtracted, normalizer(diff_snr)], axis=1)
x = combiner(x)
x1 = signal_input1([x, input_snr1])
x2 = signal_input2([x, input_snr2])
out1 = decoder1(x1)
out2 = decoder2(x2)

# %%
model = Model(inputs=[input_signal1, input_signal2, input_snr1, input_snr2], outputs=[out1, out2])
model.summary()
alpha = K.variable(.5)

optim = Adam(learning_rate=0.0001)

model.compile(optimizer="rmsprop", loss=["categorical_crossentropy", "categorical_crossentropy"], loss_weights=[.7, .3],
            metrics=["accuracy", BER])

# %%

train_datas = generate_train_datas(k=k, N=2800000, include_snr=True, oscilation=oscilation)

model.fit(x=train_datas, y=train_datas[:k], validation_split=.5, batch_size=16000,
    epochs=2000, shuffle=False, callbacks=[
    EarlyStopping(patience=3, restore_best_weights=True, monitor="loss", mode="min"),
    ReduceLROnPlateau(monitor="loss", factor=.5, patience=20),
    # AlphaCallback(alpha),
])

# %%

scatter_plot = []
for i in range(M):
    temp = np.zeros(M)
    temp[i] = 1
    out1 = encoder1.predict(np.expand_dims(temp, axis=0))
    for j in range(M):
        temp2 = np.zeros(M)
        temp2[j] = 1
        out2 = encoder1.predict(np.expand_dims(temp2, axis=0))
        comb_in = np.zeros(9)
        comb_in[:2] = out1.squeeze(0)
        comb_in[2:4] = out2.squeeze(0)
        comb_in[4:6] = out1 + out2
        comb_in[6:8] = out2 - out1
        comb_in[9:] = 0/10 #/ (7 + oscilation)#, 7 / (7 + oscilation)#, 6 / (10 + oscilation)
        # out = np.concatenate(comb_in)
        out = combiner.predict(np.expand_dims(comb_in, axis=0))
        scatter_plot.append(out.squeeze(0))

scatter_plot = np.array(scatter_plot)

scatter_plot = scatter_plot.reshape(-1, 2, 1)

plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1])
# %%
