# -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, GRU, Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.merge import Multiply

def grna_preprocess(lines):
    length = 23
    data_n = len(lines)
    seq = np.zeros((data_n, length, 4), dtype=int)
    for l in range(data_n):
        data = lines[l]
        seq_temp = data
        for i in range(length):
            if seq_temp[i] in "Aa":
                seq[l, i, 0] = 1
            elif seq_temp[i] in "Cc":
                seq[l, i, 1] = 1
            elif seq_temp[i] in "Gg":
                seq[l, i, 2] = 1
            elif seq_temp[i] in "Tt":
                seq[l, i, 3] = 1
    return seq


def epi_preprocess(lines):
    length = 23
    data_n = len(lines)
    epi = np.zeros((data_n, length), dtype=int)
    for l in range(data_n):
        data = lines[l]
        epi_temp = data
        for i in range(length):
            if epi_temp[i] in "A":
                epi[l, i] = 1
            elif epi_temp[i] in "N":
                epi[l, i] = 0
    return epi


def preprocess(file_path, usecols):
    data = pd.read_csv(file_path, usecols=usecols)
    data = np.array(data)
    epi_1, epi_2, epi_3, epi_4 = epi_preprocess(data[:, 0]), epi_preprocess(data[:, 1]), epi_preprocess(data[:, 2]), epi_preprocess(data[:, 3])
    epi = []
    for i in range(len(data)):
        epi_1_temp, epi_2_temp, epi_3_temp, epi_4_temp = pd.DataFrame(epi_1[i]), pd.DataFrame(epi_2[i]), pd.DataFrame(
            epi_3[i]), pd.DataFrame(epi_4[i])
        epi_temp = pd.concat([epi_1_temp, epi_2_temp, epi_3_temp, epi_4_temp], axis=1)
        epi_temp = np.array(epi_temp)
        epi.append(epi_temp)
    epi = np.array(epi)
    return epi


def load_data(test_file):
    test_data = pd.read_csv(test_file, usecols=[4, 9])
    test_data = np.array(test_data)
    x_test, y_test = test_data[:, 0], test_data[:, 1]
    x_test = grna_preprocess(x_test)
    epi_test = preprocess(test_file, [5, 6, 7, 8])
    y_test = y_test.reshape(len(y_test), -1)
    return x_test, epi_test, y_test

def main():
    seq_input = Input(shape=(23, 4))
    seq_conv1 = Convolution1D(256, 5, kernel_initializer='random_uniform', name='seq_conv1')(seq_input)
    seq_act1 = Activation('relu')(seq_conv1)
    seq_pool1 = MaxPooling1D(2)(seq_act1)
    seq_drop1 = Dropout(0.2)(seq_pool1)
    gru1 = Bidirectional(GRU(256, kernel_initializer='he_normal', dropout=0.3, recurrent_dropout=0.2), name='gru1')(seq_drop1)
    seq_dense1 = Dense(256, name='seq_dense1')(gru1)
    seq_act2 = Activation('relu')(seq_dense1)
    seq_drop2 = Dropout(0.3)(seq_act2)
    seq_dense2 = Dense(128, name='seq_dense2')(seq_drop2)
    seq_act3 = Activation('relu')(seq_dense2)
    seq_drop3 = Dropout(0.2)(seq_act3)
    seq_dense3 = Dense(64, name='seq_dense3')(seq_drop3)
    seq_act4 = Activation('relu')(seq_dense3)
    seq_drop4 = Dropout(0.2)(seq_act4)
    seq_dense4 = Dense(40, name='seq_dense4')(seq_drop4)
    seq_act5 = Activation('relu')(seq_dense4)
    seq_drop5 = Dropout(0.2)(seq_act5)

    epi_input = Input(shape=(23, 4))
    epi_conv1 = Convolution1D(256, 5, name='epi_conv1')(epi_input)
    epi_act1 = Activation('relu')(epi_conv1)
    epi_pool1 = MaxPooling1D(2)(epi_act1)
    epi_drop1 = Dropout(0.3)(epi_pool1)
    epi_dense1 = Dense(256, name='epi_dense1')(epi_drop1)
    epi_act2 = Activation('relu')(epi_dense1)
    epi_drop2 = Dropout(0.2)(epi_act2)
    epi_dense2 = Dense(128, name='epi_dense2')(epi_drop2)
    epi_act3 = Activation('relu')(epi_dense2)
    epi_drop3 = Dropout(0.3)(epi_act3)
    epi_dense3 = Dense(64, name='epi_dense3')(epi_drop3)
    epi_act4 = Activation('relu')(epi_dense3)
    epi_drop4 = Dropout(0.3)(epi_act4)
    epi_act5 = Dense(40, name='epi_dense4')(epi_drop4)
    epi_out = Activation('relu')(epi_act5)

    seq_epi_m = Multiply()([seq_drop5, epi_out])
    seq_epi_drop = Dropout(0.2)(seq_epi_m)
    seq_epi_flat = Flatten()(seq_epi_drop)
    seq_epi_output = Dense(1, activation='linear')(seq_epi_flat)

    model = Model(inputs=[seq_input, epi_input], outputs=[seq_epi_output])

    print("Loading weights for the models")
    model.load_weights('weights/C_RNNCrispr_weights.h5')

    print("Loading test data")
    x_test, epi_test, y_test = load_data(test_file)

    print("Predicting on test data")
    y_test = pd.DataFrame(y_test)
    y_pred = model.predict([x_test, epi_test], batch_size=256, verbose=2)
    y_pred = pd.DataFrame(y_pred)

    result = pd.concat([y_test, y_pred], axis=1)
    result.to_csv(result_file, index=False, sep=',', header=['y_test', 'y_pred'])


if __name__ == '__main__':
    test_file = "data/input_example.csv"
    result_file = "result/output_example.csv"
    main()

