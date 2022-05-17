from tensorflow import keras
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import kerasncp1 as kncp
from shapely.geometry import LineString
from shapely.geometry import Point
from kerasncp1.tf import LTCCell
from tensorflow.keras.constraints import min_max_norm

def GeneralEquation(first_x,first_y,second_x,second_y):
    # 一般式 Ax+By+C=0
    A = second_y-first_y
    B = first_x-second_x
    C = second_x*first_y-first_x*second_y
    if B==0:
        return 0,0
    else:
        k = -1 * A / B
        b = -1 * C / B
        return k, b

def LineIntersectCircle(x0,y0,r0,x1,y1,x2,y2):
    p = Point(x0, y0)
    c = p.buffer(r0).boundary
    l = LineString([(x1, y1), (x2, y2)])
    i = c.intersection(l)
    return i.geoms[0].coords[0],i.geoms[1].coords[0]
# 绕pointx,pointy逆时针旋转
def Nrotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return nRotatex, nRotatey


# 绕pointx,pointy顺时针旋转
def Srotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return sRotatex, sRotatey

# 线段按某个角度旋转
def ret(disAngle,x,y):
    if disAngle < 0:
        rotateX, rotateY = Nrotate(math.radians(abs(disAngle)), x, y, x1[-1], y1[-1])
    else:
        rotateX, rotateY = Srotate(math.radians(abs(disAngle)), x, y, x1[-1], y1[-1])
    return rotateX,rotateY

def drwa_line(angle):
    # TODO 第一步，做出行走方向延长线
    p1x = x1[-1]
    p1y = y1[-1]
    p2x = x1[-2]
    p2y = y1[-2]
    k, b = GeneralEquation(p1x, p1y, p2x, p2y)
    if k == 0 and b == 0:
        x = x1[-1]
        y = y1[-1] + 1
    else:
        a1 = p1x + 2
        b1 = (k * a1) + b
        a2= p1x - 2
        b2=(k * a2) + b
        # TODO 第二步，最后一个点为圆心，直径为1的圆与延长线相交，取得交点
        point1,point2=LineIntersectCircle(p1x, p1y, 1, a1, b1, a2, b2)
        if p1x > p2x:
            if point1[0]>point2[0]:
                x=point1[0]
                y=point1[1]
            else:
                x=point2[0]
                y=point2[1]
        if p1x<p2x:
            if point1[0]<point2[0]:
                x=point1[0]
                y=point1[1]
            else:
                x=point2[0]
                y=point2[1]

    # TODO 第三步，根据角度，以最后一个点为旋转中心，对延长线进行旋转并得到新点，即小车前进的下一个点
    if angle - 0 == 0:
        x = x
        y = y
    else:
        x, y = ret(angle,x,y)
    x1.append(x)
    y1.append(y)

def jaccard(num1,num2):
    if num1>num2:
        return num2/num1
    else:
        return num1/num2

if __name__ == '__main__':
    anglist=[]
    perslist=[]
    x1 = [0, 0]
    y1 = [-1, 0]
    li=[]
    ((x_train, y_train), (x_valid, y_valid),) = kncp.datasets.icra2020_lidar_collision_avoidance.load_data()
    num=0
    n=0
    a=0
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
    N = x_train.shape[2]
    channels = x_train.shape[3]
    model = keras.models.Sequential(
        [

            keras.layers.InputLayer(input_shape=(None, N, channels)),
            # tf.keras.layers.GaussianDropout(1),
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

    # model.load_weights('Matrix1(0).h5')
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error", metrics=['accuracy'])
    # model.load_weights('Matrix1.h5')
    # # model.fit(
    # #     x=x_train, y=y_train, batch_size=30, epochs=30, validation_data=(x_valid, y_valid)
    # # )
    #
    # pre = model.predict(x_valid, batch_size=30)
    # acc=0
    # n=0
    # num=0
    # ang=0
    # for x in range(250):
    #     for y in range(32):
    #         for z in range(1):
    #             predict=pre[x,y,z]
    #             target=y_valid[x,y,z]
    #             if n>1568 and n<2111:
    #                 if num<31:
    #                     ang=ang+predict*90
    #                     num=num+1
    #                 else:
    #                     ang=ang/31
    #                     drwa_line(ang)
    #                     num=0
    #                     ang=0
    #             acc = acc + jaccard(math.exp(predict),math.exp(target))
    #             n=n+1
    #
    # print(acc/n)
    # x1.pop(0)
    # y1.pop(0)
    # print(x1)
    # print(y1)
    # plt.plot(x1, y1, color='green')
    # plt.scatter(x1[0], y1[0], color='black')
    # plt.scatter(x1[-1], y1[-1], color='blue')
    # plt.show()

    #
