import h5py
import numpy as np
from tensorflow import keras
import tensorflow as tf
import kerasncp1 as kncp
from kerasncp1.tf import LTCCell
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GaussianNoise
train={}
infodic={}
def train_module():
    model.load_weights('train_1.h5')
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error",metrics=['accuracy'])
    print('Start training...........')
    # Train the model
    tf.compat.v1.experimental.output_all_intermediates(True)

    model.fit(
        x=data, y=y_valid, batch_size=30, epochs=1, validation_data=(data, y_valid)
    )
    model.save_weights('train_1.h5')


def getMartrix_info():
    train.clear()
    fdata = h5py.File('train_1.h5', 'r+')
    train['rnn-cm'] = fdata['rnn']['rnn']['ltc_cell']['cm:0'].value
    train['rnn-erev'] = fdata['rnn']['rnn']['ltc_cell']['erev:0'].value
    train['rnn-gleak'] = fdata['rnn']['rnn']['ltc_cell']['gleak:0'].value
    train['rnn-input_b'] = fdata['rnn']['rnn']['ltc_cell']['input_b:0'].value
    train['rnn-input_w'] = fdata['rnn']['rnn']['ltc_cell']['input_w:0'].value
    train['rnn-mu'] = fdata['rnn']['rnn']['ltc_cell']['mu:0'].value
    train['rnn-output_b'] = fdata['rnn']['rnn']['ltc_cell']['output_b:0'].value
    train['rnn-output_w'] = fdata['rnn']['rnn']['ltc_cell']['output_w:0'].value
    train['rnn-sensory_erev'] = fdata['rnn']['rnn']['ltc_cell']['sensory_erev:0'].value
    train['rnn-sensory_mu'] = fdata['rnn']['rnn']['ltc_cell']['sensory_mu:0'].value
    train['rnn-sensory_sigma'] = fdata['rnn']['rnn']['ltc_cell']['sensory_sigma:0'].value
    train['rnn-sensory_w'] = fdata['rnn']['rnn']['ltc_cell']['sensory_w:0'].value
    train['rnn-sigma'] = fdata['rnn']['rnn']['ltc_cell']['sigma:0'].value
    train['rnn-vleak'] = fdata['rnn']['rnn']['ltc_cell']['vleak:0'].value
    train['rnn-w'] = fdata['rnn']['rnn']['ltc_cell']['w:0'].value
    train['time_distributed-timedistributed-kernel'] = fdata['time_distributed']['time_distributed'][
        'kernel:0'].value
    train['time_distributed_1-timedistributed_1-kernel'] = fdata['time_distributed_1']['time_distributed_1'][
        'kernel:0'].value
    train['time_distributed_3-timedistributed_3-kernel'] = fdata['time_distributed_3']['time_distributed_3'][
        'kernel:0'].value
    train['time_distributed_5-timedistributed_5-kernel'] = fdata['time_distributed_5']['time_distributed_5'][
        'kernel:0'].value
    train['time_distributed_7-timedistributed_7-kernel'] = fdata['time_distributed_7']['time_distributed_7'][
        'kernel:0'].value
    train['time_distributed'] = fdata['time_distributed']['time_distributed']['bias:0'].value
    train['time_distributed_1'] = fdata['time_distributed_1']['time_distributed_1']['bias:0'].value
    train['time_distributed_3'] = fdata['time_distributed_3']['time_distributed_3']['bias:0'].value
    train['time_distributed_5'] = fdata['time_distributed_5']['time_distributed_5']['bias:0'].value
    train['time_distributed_7'] = fdata['time_distributed_7']['time_distributed_7']['bias:0'].value

def deteoverange(num):
    if num>3:
        return 3
    elif num<-3:
        return -3
    else:
        return num

def write_data_into_file(dic=train):
    f = h5py.File('train_1.h5', 'r+')
    randvalue=1
    # rnn-cm:0
    # while f['rnn']['rnn']['ltc_cell']['cm:0'].value[randvalue]!=dic.get('rnn-cm')[randvalue]:
    del f['rnn']['rnn']['ltc_cell']['cm:0']
    dcm=f.create_dataset('rnn/rnn/ltc_cell/cm:0', (21,), dtype='float32')
    dcm[...]=dic.get('rnn-cm')
    #randvalue=random.randint(1,9)

    # rnn-erev
    del f['rnn']['rnn']['ltc_cell']['erev:0']
    derev = f.create_dataset('rnn/rnn/ltc_cell/erev:0', (21, 21), dtype='float32')
    derev[...]=dic.get('rnn-erev')

    # rnn-gleak
    del f['rnn']['rnn']['ltc_cell']['gleak:0']
    dgleak = f.create_dataset('rnn/rnn/ltc_cell/gleak:0', (21,), dtype='float32')
    dgleak[...]=dic.get('rnn-gleak')

    # rnn-input_b
    del f['rnn']['rnn']['ltc_cell']['input_b:0']
    dinput_b = f.create_dataset('rnn/rnn/ltc_cell/input_b:0', (32,), dtype='float32')
    dinput_b[...]=dic.get('rnn-input_b')

    # rnn-input_w
    del f['rnn']['rnn']['ltc_cell']['input_w:0']
    dinput_w = f.create_dataset('rnn/rnn/ltc_cell/input_w:0', (32,), dtype='float32')
    dinput_w[...]=dic.get('rnn-input_w')

    # rnn-mu
    del f['rnn']['rnn']['ltc_cell']['mu:0']
    dmu = f.create_dataset('rnn/rnn/ltc_cell/mu:0', (21, 21), dtype='float32')
    dmu[...]=dic.get('rnn-mu')

    # rnn-output_b
    del f['rnn']['rnn']['ltc_cell']['output_b:0']
    doutput_b = f.create_dataset('rnn/rnn/ltc_cell/output_b:0', (1,), dtype='float32')
    doutput_b[...]=dic.get('rnn-output_b')

    # rnn-output_w
    del f['rnn']['rnn']['ltc_cell']['output_w:0']
    doutput_w = f.create_dataset('rnn/rnn/ltc_cell/output_w:0', (1,), dtype='float32')
    doutput_w[...]=dic.get('rnn-output_w')

    # rnn-sensory_ever
    del f['rnn']['rnn']['ltc_cell']['sensory_erev:0']
    dsensory_erev = f.create_dataset('rnn/rnn/ltc_cell/sensory_erev:0', (32, 21), dtype='float32')
    dsensory_erev[...]=dic.get('rnn-sensory_erev')

    # rnn-sensory_mu
    del f['rnn']['rnn']['ltc_cell']['sensory_mu:0']
    dsensory_mu = f.create_dataset('rnn/rnn/ltc_cell/sensory_mu:0', (32, 21), dtype='float32')
    dsensory_mu[...]=dic.get('rnn-sensory_mu')

    # rnn-sensory_sigma
    del f['rnn']['rnn']['ltc_cell']['sensory_sigma:0']
    dsessory_sigma = f.create_dataset('rnn/rnn/ltc_cell/sensory_sigma:0', (32, 21), dtype='float32')
    dsessory_sigma[...]=dic.get('rnn-sensory_sigma')

    # rnn-sensory_w
    del f['rnn']['rnn']['ltc_cell']['sensory_w:0']
    dsensory_w = f.create_dataset('rnn/rnn/ltc_cell/sensory_w:0', (32, 21), dtype='float32')
    dsensory_w[...]=dic.get('rnn-sensory_w')

    #rnn-sigma
    del f['rnn']['rnn']['ltc_cell']['sigma:0']
    dsigma = f.create_dataset('rnn/rnn/ltc_cell/sigma:0', (21, 21), dtype='float32')
    dsigma[...]=dic.get('rnn-sigma')

    # rnn-vleak
    del f['rnn']['rnn']['ltc_cell']['vleak:0']
    dvleak = f.create_dataset('rnn/rnn/ltc_cell/vleak:0', (21,), dtype='float32')
    dvleak[...]=dic.get('rnn-vleak')

    # rnn-w
    del f['rnn']['rnn']['ltc_cell']['w:0']
    dw = f.create_dataset('rnn/rnn/ltc_cell/w:0', (21, 21), dtype='float32')
    dw[...]=dic.get('rnn-w')

    #time_distributed-timedistributed-kernel
    del f['time_distributed']['time_distributed']['kernel:0']
    dtimedistributed = f.create_dataset('time_distributed/time_distributed/kernel:0', (5, 1, 18), dtype='float32')
    dtimedistributed[...]=dic.get('time_distributed-timedistributed-kernel')
    del f['time_distributed']['time_distributed']['bias:0']
    dtimedistributed_bias=f.create_dataset('time_distributed/time_distributed/bias:0', (18,), dtype='float32')
    dtimedistributed_bias[...]=dic.get('time_distributed')

    #time_distributed_1-timedistributed_1-kernel
    del f['time_distributed_1']['time_distributed_1']['kernel:0']
    dtimedistributed_1 = f.create_dataset('time_distributed_1/time_distributed_1/kernel:0', (5, 18, 20), dtype='float32')
    dtimedistributed_1[...]=dic.get('time_distributed_1-timedistributed_1-kernel')
    del f['time_distributed_1']['time_distributed_1']['bias:0']
    dtimedistributed1_bias=f.create_dataset('time_distributed_1/time_distributed_1/bias:0', (20,), dtype='float32')
    dtimedistributed1_bias[...]=dic.get('time_distributed_1')

    # time_distributed_3-timedistributed_3-kernel
    del f['time_distributed_3']['time_distributed_3']['kernel:0']
    dtimedistributed_3 = f.create_dataset('time_distributed_3/time_distributed_3/kernel:0', (5, 20, 22), dtype='float32')
    dtimedistributed_3[...]=dic.get('time_distributed_3-timedistributed_3-kernel')
    del f['time_distributed_3']['time_distributed_3']['bias:0']
    dtimedistributed3_bias=f.create_dataset('time_distributed_3/time_distributed_3/bias:0', (22,), dtype='float32')
    dtimedistributed3_bias[...]=dic.get('time_distributed_3')

    # time_distributed_5-timedistributed_5-kernel
    del f['time_distributed_5']['time_distributed_5']['kernel:0']
    dtimedistributed_5 = f.create_dataset('time_distributed_5/time_distributed_5/kernel:0', (5, 22, 24), dtype='float32')
    dtimedistributed_5[...]=dic.get('time_distributed_5-timedistributed_5-kernel')
    del f['time_distributed_5']['time_distributed_5']['bias:0']
    dtimedistributed5_bias=f.create_dataset('time_distributed_5/time_distributed_5/bias:0', (24,), dtype='float32')
    dtimedistributed5_bias[...]=dic.get('time_distributed_5')

    # time_distributed_7-timedistributed_7-kernel
    del f['time_distributed_7']['time_distributed_7']['kernel:0']
    dtimedistributed_7 = f.create_dataset('time_distributed_7/time_distributed_7/kernel:0', (384, 32), dtype='float32')
    dtimedistributed_7[...]=dic.get('time_distributed_7-timedistributed_7-kernel')
    del f['time_distributed_7']['time_distributed_7']['bias:0']
    dtimedistributed7_bias=f.create_dataset('time_distributed_7/time_distributed_7/bias:0', (32,), dtype='float32')
    dtimedistributed7_bias[...]=dic.get('time_distributed_7')

def iteraldic():
    # rnn-cm
    for i in range(21):
       train.get('rnn-cm')[i]=deteoverange(train.get('rnn-cm')[i])

    # rnn-erev
    for i in range(21):
        for j in range(21):
            train.get('rnn-erev')[i][j] = deteoverange(train.get('rnn-erev')[i][j])
    # rnn-gleak
    for i in range(21):
        train.get('rnn-gleak')[i] =deteoverange(train.get('rnn-gleak')[i])

    # rnn-input_b
    for i in range(32):
        train.get('rnn-input_b')[i] =deteoverange(train.get('rnn-input_b')[i])

    # rnn-input_w
    for i in range(32):
       train.get('rnn-input_w')[i] = deteoverange(train.get('rnn-input_w')[i])

    # rnn-mu
    for i in range(21):
        for j in range(21):
            train.get('rnn-mu')[i][j] = deteoverange(train.get('rnn-mu')[i][j])

    # rnn-output_b
    for i in range(1):
        train.get('rnn-output_b')[i] = deteoverange(train.get('rnn-output_b')[i])

    # rnn-output_w
    for i in range(1):
        train.get('rnn-output_w')[i] = deteoverange(train.get('rnn-output_w')[i])

    # rnn-sensory_erev
    for i in range(32):
        for j in range(21):
            train.get('rnn-sensory_erev')[i][j] = deteoverange(train.get('rnn-sensory_erev')[i][j])

    # rnn-sensory_mu
    for i in range(32):
        for j in range(21):
           train.get('rnn-sensory_mu')[i][j] = deteoverange(train.get('rnn-sensory_mu')[i][j])


    # rnn-sensory_sigma
    for i in range(32):
        for j in range(21):
            train.get('rnn-sensory_sigma')[i][j] = deteoverange(train.get('rnn-sensory_sigma')[i][j])

    # rnn-sensory_w
    for i in range(32):
        for j in range(21):
            train.get('rnn-sensory_w')[i][j] = deteoverange(train.get('rnn-sensory_w')[i][j])

    # rnn-sigma
    for i in range(21):
        for j in range(21):
            train.get('rnn-sigma')[i][j] = deteoverange(train.get('rnn-sigma')[i][j])

    # rnn-vleak
    for i in range(21):
        train.get('rnn-vleak')[i] = deteoverange(train.get('rnn-vleak')[i])

    # rnn-w
    for i in range(21):
        for j in range(21):
            train.get('rnn-w')[i][j] = deteoverange(train.get('rnn-w')[i][j])

    # time_distributed-timedistributed-kernel
    for i in range(5):
        for j in range(1):
            for k in range(18):
                train.get('time_distributed-timedistributed-kernel')[i][j][k]=deteoverange(train.get('time_distributed-timedistributed-kernel')[i][j][k])


    # time_distributed - bias
    for i in range(18):
        train.get('time_distributed')[i] = deteoverange(train.get('time_distributed')[i])

    # time_distributed_1-timedistributed_1-kernel
    for i in range(5):
        for j in range(18):
            for k in range(20):
                train.get('time_distributed_1-timedistributed_1-kernel')[i][j][k]=deteoverange(train.get('time_distributed_1-timedistributed_1-kernel')[i][j][k])

    # time_distributed_1 - bias
    for i in range(20):
        train.get('time_distributed_1')[i] = deteoverange(train.get('time_distributed_1')[i])

    # time_distributed_3-timedistributed_3-kernel
    for i in range(5):
        for j in range(20):
            for k in range(22):
                train.get('time_distributed_3-timedistributed_3-kernel')[i][j][k]=deteoverange(train.get('time_distributed_3-timedistributed_3-kernel')[i][j][k])

        # time_distributed_3 - bias
        for i in range(22):
           train.get('time_distributed_3')[i] =deteoverange(train.get('time_distributed_3')[i])

    # time_distributed_5-timedistributed_5-kernel
    for i in range(5):
        for j in range(22):
            for k in range(24):
                train.get('time_distributed_5-timedistributed_5-kernel')[i][j][k]=deteoverange(train.get('time_distributed_5-timedistributed_5-kernel')[i][j][k])

    # time_distributed_5 - bias
    for i in range(24):
       train.get('time_distributed_5')[i]=deteoverange(train.get('time_distributed_5')[i])

    # time_distributed_7-timedistributed_7-kernel
    for i in range(384):
        for j in range(32):
            train.get('time_distributed_7-timedistributed_7-kernel')[i][j] =deteoverange(train.get('time_distributed_7-timedistributed_7-kernel')[i][j])

    # time_distributed_7 - bias
    for i in range(32):
        train.get('time_distributed_7')[i] = deteoverange(train.get('time_distributed_7')[i])
def jaccard(num1,num2):
    if num1>num2:
        return num2/num1
    else:
        return num1/num2

if __name__ == '__main__':
    # Download the dataset (already implemented in keras-ncp)
    (
        (x_train, y_train),
        (x_valid, y_valid),
    ) = kncp.datasets.icra2020_lidar_collision_avoidance.load_data()
    layer = tf.keras.layers.GaussianNoise(0)
    data = layer(x_valid, training=True)
    N = x_train.shape[2]
    channels = x_train.shape[3]

    wiring = kncp.wirings.NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incomming syanpses has each motor neuron
    )
    rnn_cell = LTCCell(wiring)

    # We need to use the TimeDistributed layer to independently apply the
    # Conv1D/MaxPool1D/Dense over each time-step of the input time-series.

    model = keras.models.Sequential(
        [

            keras.layers.InputLayer(input_shape=(None, N, channels)),
            keras.layers.TimeDistributed(keras.layers.Conv1D(18, 5, strides=3, activation="relu")),
            keras.layers.TimeDistributed(keras.layers.Conv1D(20, 5, strides=2, activation="relu")),
            keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
            keras.layers.TimeDistributed(keras.layers.Conv1D(22, 5, activation="relu")),
            keras.layers.TimeDistributed(keras.layers.MaxPool1D()),
            keras.layers.TimeDistributed(keras.layers.Conv1D(24, 5, activation="relu")),
            keras.layers.TimeDistributed(keras.layers.Flatten()),
            keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu")),
            keras.layers.RNN(rnn_cell, return_sequences=True),
        ]
    )

    for i in range(50):
        train_module()
        getMartrix_info()
        iteraldic()
        write_data_into_file()
        model.load_weights('train_1.h5')
        pre = model.predict(x_valid, batch_size=30)
        n = 0
        a = 0
        b = 0
        acc = 0
        for x in range(250):
            for y in range(32):
                for z in range(1):
                    predict = pre[x, y, z]
                    target = y_valid[x, y, z]
                    acc = acc + jaccard(math.exp(predict), math.exp(target))
                    n = n + 1
        print(str(i) + " : " + str(acc / n))


