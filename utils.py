import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, concatenate, LSTM, Add, Multiply, Layer
from keras.models import Model, Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback

def plot_scatter_constellation(encoder, show = True, M = 4):
    scatter_plot = []
    for i in range(M):
        temp = np.zeros(M)
        temp[i] = 1
        out1 = encoder.predict(np.expand_dims(temp, axis=0))
        scatter_plot.append(out1.squeeze(0))

    scatter_plot = np.array(scatter_plot)

    scatter_plot = scatter_plot.reshape(-1, 2, 1)

    if show:
        plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1])

    else:
        return scatter_plot

def plot_scatter_duo(encoders, combiner, show=True, M=4):
    encoder1, encoder2 = encoders
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

    if show:
        plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1])

    else:
        return scatter_plot

def calc_ebno(SNR):
    return 10**(SNR/10)

def generate_data(M, N=100000, n_receiver=2):
    label = np.random.randint(M, size=N)

    data = np.zeros((N, M))
    for i in range(N):
        data[i, label[i]] = 1
    np.random.shuffle(data)
    return data

def generate_qam(M = 4, N=100000):
    label = np.random.randint(M, size=N)

    data = np.ones((N, 2))
    label_data = np.zeros((N, M))
    for i in range(N):
        label_data[i, label[i]] = 1
        if label[i] == 1:
            data[i, :] = [-1, 1]
        elif label[i] == 2:
            data[i, :] = [-1, -1]
        elif label[i] == 3:
            data[i, :] = [1, -1]
    
    return data, label_data

def ber_curve_combined(encoders, decoders, combiner, R = 1, SNR_range = [-10, 20], M=4):
    test_datas = [generate_data(M=M) for i in range(len(decoders))]
    ebno_range = calc_ebno(np.arange(SNR_range[0], SNR_range[1]))
    bers = [[]] * len(decoders)
    for ebno in ebno_range:
        outs = [encoders[i].predict(test_datas[i]) for i in range(len(encoders))]
        concatenated = np.concatenate(outs, axis=1)
        out = combiner.predict(concatenated)
        receiveds = [out + np.sqrt(1 / (2 * R * ebno)) * np.random.randn(*out.shape) for i in range(len(outs))]
        outs = [decoders[i].predict(receiveds[i]) for i in range(len(decoders))]
        preds = [np.argmax(out, axis=1) for out in outs]
        errors = [np.asarray((preds[i] != np.argmax(test_datas[i], axis=1))).astype(int).mean() for i in range(len(preds))]
        
        for i, error in enumerate(errors):
            bers[i].append(error)

    return bers

def ber_curve(encoder, decoder, R = 1, SNR_range = [-10, 20], M=4):
    test_datas = generate_data(M=M)
    ebno_range = calc_ebno(np.arange(SNR_range[0], SNR_range[1]))
    ber = []
    for ebno in ebno_range:
        out = encoder.predict(test_datas)
        received = out + np.sqrt(1 / (2 * R * ebno)) * np.random.randn(*out.shape)
        out = decoder.predict(received)
        preds = np.argmax(out, axis=1)
        errors = np.asarray((preds != np.argmax(test_datas, axis=1))).astype(int).mean()
        
        ber.append(errors)

    return ber

def rate_curve(bers, R = 1):
    rates = [1 - np.array(ber) * R for ber in bers]
    return np.sum(np.array(rates), axis=0)

def seed_everything(seed_number):
    tf.set_random_seed(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number)

normalize_layer = Lambda(lambda inputs: inputs[0] / inputs[1])

def generate_train_datas(M = 4, N = 400000, k=2, include_snr=False, oscilation=5):
    if include_snr:
        # snrs = [np.random.randint(0, 10, (N, 1)) for i in range(k)]
        # snrs = np.array(snrs)
        # snrs = np.sort(snrs, 0)
        snrs = np.ones((N, k + 1))
        step_size = int(N / (2 * oscilation))

        for i in range(oscilation):
            snrs[i * step_size: (i + 1) * step_size, 0] *= 7 - i
            snrs[i * step_size: (i + 1) * step_size, 1] *= 7 + i
        for i in range(oscilation, 2 * oscilation):
            snrs[i * step_size: (i + 1) * step_size, 0] *= 7 - i
            snrs[i * step_size: (i + 1) * step_size, 1] *= 7 + i

        # snrs[:, 2] = snrs[:, 1] - snrs[:, 0]

        np.random.shuffle(snrs)

        snrs = [snrs[:, i].squeeze() for i in range(k)]
        train_data = [generate_data(M=M, N=N) for i in range(k)]
        train_data.extend(snrs)
        return train_data
    return [generate_data(M=M, N=N) for i in range(k)]

def BER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)  

def create_decoder(layer_sizes, name="decoder1", activation="relu"):
    layers = [
        Dense(layer_size, activation=activation) for layer_size in layer_sizes[:-1]
    ]
    layers.append(Dense(layer_sizes[-1], activation="softmax"))
    return Sequential(layers, name=name)

def create_encoder(layer_sizes, name="encoder1", activations=["relu", "linear"]):
    layers = [
        Dense(layer_size, activation=activations[0]) for layer_size in layer_sizes[:-1]
    ]
    layers.append(Dense(layer_sizes[-1], activation=activations[1]))
    layers.append(BatchNormalization(center=False, scale=False))
    return Sequential(layers, name=name)

def create_combiner(layer_sizes, activations=["relu", "linear"], dropout_prob=0, name="combiner"):
    layers = [
        Dense(layer_size, activation=activations[0]) for layer_size in layer_sizes[:-1]
    ]
    layers.append(Dropout(dropout_prob))
    layers.append(Dense(layer_sizes[-1], activation=activations[1]))
    layers.append(BatchNormalization(center=False, scale=False))
    # layers.append(Lambda(lambda x: np.sqrt(layer_sizes[-1])*K.l2_normalize(x,axis=1)))
    return Sequential(layers, name=name)

def create_inputs(R, H, t, k, ebno, name="transmit1"):
    # return Sequential(
    #             [Lambda(TransmissionLayer, arguments={"H": H,"R": R, "ebno": ebno, "t":t, "k":k}),
    #             GaussianNoise(np.sqrt(1 / (2 * R * ebno)))], name=name)
    return Lambda(TransmissionLayer, arguments={"H": H,"R": R, "ebno": ebno, "t":t, "k":k}, name=name)

def measure_sig_power(sig):
    sig = np.array(sig)
    sig = np.array(sig[:, 0] + 1j * sig[:, 1])
    sig_power = np.sum(np.abs(sig**2)) / len(sig)
    return sig_power

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

class AlphaCallback(Callback):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def on_batch_end(self, batch, logs=None):
        loss1, loss2 = logs["decoder1_loss"], logs["decoder2_loss"]
        K.set_value(self.alpha, loss1 / (loss1 + loss2))