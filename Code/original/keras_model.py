from keras.layers import BatchNormalization, GRU, Bidirectional, Conv2D, MaxPooling2D, Input, TimeDistributed, Dense, Activation, Dropout, Reshape, Permute
from keras.models import Model
from keras.optimizers import Adam
import keras

# 'channel_first' has occured error
# https://stackoverflow.com/questions/68036975/valueerror-shape-must-be-at-least-rank-3-but-is-rank-2-for-node-biasadd
data_in_shape = None, 128, 1024, 8
def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, weights):
    # data_in shape is
    # batch_size, seq_len, feat_len, 2_nb_ch
    # None, 128, 1024, 8
    data_in = data_in_shape
    spec_start = Input(shape=(tuple(data_in[1:])))

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # before permute, shape of spec_cnn is [None, 64, 128, 4]
    # channel은 8 -> 64,
    # 주파수, data_in의 height, 128 유지,
    # 시간축, data_in의 width, 1024 -> 4로 MaxPooling
    # spec_cnn = Permute((3, 2, 1))(spec_cnn)

    # RNN
    # data_in[-2] : 주파수 축
    spec_rnn = Reshape((data_in[-3], -1))(spec_cnn)
    # spec_rnn shape is [None, 128, 256]
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('linear', name='doa_out')(doa)

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    return model