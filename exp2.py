from keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, concatenate, LSTM, Add, Multiply, Layer
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

def trial(seed_number):
    tf.set_random_seed(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number)


    # H = K.random_normal(shape=(1,2), mean=0.0, stddev=1.0) 
    H = K.constant([[1, 1]])
    print(K.get_value(H))


    SNR1 = 7
    SNR2 = 7
    SNRs = [SNR1, SNR2]


    k = 2
    n_channel = 2
    M = 2 ** k
    k = int(k)
    R = k / n_channel
    calc_ebno = lambda SNR: 10 ** (SNR / 10)
    ebno = [calc_ebno(SNR) for SNR in SNRs]
    ebno1 = calc_ebno(SNR1)
    ebno2 = calc_ebno(SNR2)


    def generate_data(M, N=100000, n_receiver=2):
        label = np.random.randint(M, size=N)

        data = np.zeros((N, M))
        for i in range(N):
            data[i, label[i]] = 1
        np.random.shuffle(data)
        return data

    def TransmissionLayer(x, t, k):
        signal = H[t, k] * x

        for i in range(t):
            if i == k:
                continue
            interference = H[i, k] * x
            signal = signal + interference
        
        
        # noise = K.random_normal(K.shape(signal),
        #                     mean=0,
        #                     stddev=np.sqrt( 1/ (2 * R * ebno[k])))
        return signal 


    # Layer definitions
    input_signal1 = Input(shape=(M,), name="input1")
    input_signal2 = Input(shape=(M,), name="input2")
    encoder1 = Sequential([
        Dense(M, activation="relu"),
        Dense(n_channel, activation="linear")
    ], name="encoder1")
    encoder2 = Sequential([
        Dense(M, activation="relu"),
        Dense(n_channel, activation="linear")
    ], name="encoder2")

    combiner = Sequential([
        Dense(n_channel, activation="relu"),
        Dense(n_channel, activation="linear"),
        BatchNormalization(center=False, scale=False),
        # Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x,axis=1), name="normalization")
        # BatchNormalization(),
        # Lambda(lambda x: x / K.sqrt(K.mean(x**2))) #average power constraint
    ], name="combiner")

    signal_input1 = Sequential(
        [Lambda(TransmissionLayer, arguments={"t":0, "k":0}),
        GaussianNoise(np.sqrt(1 / (2 * R * ebno1)))], name="transmit1")

    signal_input2 = Sequential(
        [Lambda(TransmissionLayer, arguments={"t":0, "k":1}),
        GaussianNoise(np.sqrt(1 / (2 * R * ebno2)))], name="transmit2")

    decoder1 = Sequential([
        Dense(M, activation="relu"),
        Dense(M, activation="relu"),
        Dense(M, activation="softmax")
    ], name="decoder1")
    decoder2 = Sequential([
        Dense(M, activation="relu"),
        Dense(M, activation="relu"),
        Dense(M, activation="softmax")
    ], name="decoder2")

    # Computation graph
    x1 = encoder1(input_signal1)
    x2 = encoder2(input_signal2)
    x = concatenate([x1, x2], axis=1)
    x = combiner(x)
    x1 = signal_input1(x)
    x2 = signal_input2(x)
    out1 = decoder1(x1)
    out2 = decoder2(x2)


    model = Model(inputs=[input_signal1, input_signal2], outputs=[out1, out2])
    model.summary()
    alpha = K.variable(.5)

    model.compile(optimizer="adam", loss=["categorical_crossentropy", "categorical_crossentropy"],
                loss_weights=[alpha, (1 - alpha)], metrics=["accuracy"])


    plot_model(model, "hebe.png")


    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    early_stop = EarlyStopping(patience=100, restore_best_weights=True, monitor="loss")
    # ckpt_callback = ModelCheckpoint(filepath="checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor="val_loss")
    ckpt_callback = ModelCheckpoint(filepath="checkpoints/weights.hdf5", monitor="val_loss")


    class AlphaCallback(Callback):
        def __init__(self, alpha):
            super().__init__()
            self.alpha = alpha

        def on_batch_end(self, batch, logs=None):
            loss1, loss2 = logs["decoder1_loss"], logs["decoder2_loss"]
            K.set_value(self.alpha, loss1 / (loss1 + loss2))
            # self.alpha = loss1 / (loss1 + loss2)


    train_data_1 = generate_data(M, N=40000)
    train_data_2 = generate_data(M, N=40000)


    model.fit(x=[train_data_1, train_data_2], y=[train_data_1, train_data_2], validation_split=.5, batch_size=1024,
            epochs=2000, callbacks=[
            tensorboard_callback,
            early_stop,
            # ckpt_callback,
            ReduceLROnPlateau(monitor="loss", factor=.1, patience=20),
            # SGDRScheduler(min_lr=1e-5, max_lr=1e-1, steps_per_epoch=np.ceil(train_data_1.shape[0]/1024)),
            AlphaCallback(alpha)
        ])


    scatter_plot = []
    for i in range(M):
        temp = np.zeros(M)
        temp[i] = 1
        out1 = encoder1.predict(np.expand_dims(temp, axis=0))
        for j in range(M):
            temp2 = np.zeros(M)
            temp2[j] = 1
            out2 = encoder2.predict(np.expand_dims(temp2, axis=0))
            out = np.concatenate([out1.squeeze(0), out2.squeeze(0)])
            out = combiner.predict(np.expand_dims(out, axis=0))
            scatter_plot.append(out.squeeze(0))

    scatter_plot = np.array(scatter_plot)

    scatter_plot = scatter_plot.reshape(-1, 2, 1)

    import matplotlib.pyplot as plt


    SNR_range = np.arange(0, 31, 1) - 10
    ebno_range1 = calc_ebno(SNR_range)
    ebno_range2 = calc_ebno(SNR_range)
    ber1, ber2 = [], []
    test_data_1 = generate_data(M=M)
    test_data_2 = generate_data(M=M)
    for i in range(len(SNR_range)):
        ebno1 = ebno_range1[i]
        ebno2 = ebno_range2[i]
        out1 = encoder1.predict(test_data_1)
        out2 = encoder2.predict(test_data_2)
        concatenated = np.concatenate([out1, out2], axis=1)
        out = combiner.predict(concatenated)
        noise1 = np.sqrt(1 / (2 * R * ebno1)) * np.random.randn(*out.shape)
        noise2 = np.sqrt(1 / (2 * R * ebno2)) * np.random.randn(*out.shape)
        out1 = decoder1.predict(out + noise1)
        out2 = decoder2.predict(out + noise2)

        pred1 = np.argmax(out1, axis=1)
        pred2 = np.argmax(out2, axis=1)

        errors1 = np.asarray((pred1 != np.argmax(test_data_1, axis=1))).astype(int).mean()
        errors2 = np.asarray((pred2 != np.argmax(test_data_2, axis=1))).astype(int).mean()

        ber1.append(errors1)
        ber2.append(errors2)

    return (1 - np.array(ber1)) * R + (1 - np.array(ber2)) * R

file_name = "rates.txt"
for seed in range(50):
    rate = trial(seed)
    with open(file_name, "a+") as f:
        f.write("{} seed - rate {}".format(seed, rate))
