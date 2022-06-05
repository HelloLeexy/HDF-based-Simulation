import h5py
import numpy as np
from tensorflow import keras
import tensorflow as tf
import kerasncp1 as kncp
from kerasncp1.tf import LTCCell
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GaussianNoise

Matrix1={}
Gplus_T_Info={}
Gmius_T_Info={}
targetinfo={}
fileinfo={}
pushinfo={}
G2info={}
loss=[]
loss_va=[]
accuracylist=[]
val_accuracylist=[]
bias_info={}
# G Matrix updating information
LTPGmax=304.367/100
LTPGmin=0.972985/100
LTPap=21.07/100
# 参数
LTPbp=5.048

LTDGmax=304.367/100
LTDGmin=0.957552/100
LTDad=118.522/100
# 参数
LTDbd=1.8224

# 根据神经形态器件的特征对权重进行更新
def GposI(gpos):
    a = gpos-LTPGmin
    b = LTPGmax-LTPGmin
    gpos = float(LTPap*math.exp(-LTPbp*(a/b))+gpos)
    if gpos>LTPGmax:
        gpos = LTPGmax
    return gpos

def GposD(gpos):
    a = LTDGmax - gpos
    b = LTDGmax - LTDGmin
    gpos = float(-LTDad * math.exp(-LTDbd * (a/b)) +gpos)
    if gpos < LTDGmin:
        gpos=LTDGmin
    return gpos

def GnegI(gneg):
    a = gneg-LTPGmin
    b = LTPGmax-LTPGmin
    gneg = float(LTPap*math.exp(-LTPbp*(a/b))+gneg)
    if gneg>LTPGmax:
        gneg=LTPGmax
    return gneg

def GnegD(gneg):
    a = LTDGmax - gneg
    b = LTDGmax - LTDGmin
    gneg = float(-LTDad * math.exp(-LTDbd * (a/b)) + gneg)
    if gneg < LTDGmin:
        gneg=LTDGmin
    return gneg


# TODO 获得G+和的实时信息
def getMartrix_info(dicName,FileName):
    dicName.clear()
    fdata = h5py.File(FileName, 'r+')
    dicName['rnn-cm'] = fdata['rnn']['rnn']['ltc_cell']['cm:0'].value
    dicName['rnn-erev'] = fdata['rnn']['rnn']['ltc_cell']['erev:0'].value
    dicName['rnn-gleak'] = fdata['rnn']['rnn']['ltc_cell']['gleak:0'].value
    dicName['rnn-input_b'] = fdata['rnn']['rnn']['ltc_cell']['input_b:0'].value
    dicName['rnn-input_w'] = fdata['rnn']['rnn']['ltc_cell']['input_w:0'].value
    dicName['rnn-mu'] = fdata['rnn']['rnn']['ltc_cell']['mu:0'].value
    dicName['rnn-output_b'] = fdata['rnn']['rnn']['ltc_cell']['output_b:0'].value
    dicName['rnn-output_w'] = fdata['rnn']['rnn']['ltc_cell']['output_w:0'].value
    dicName['rnn-sensory_erev'] = fdata['rnn']['rnn']['ltc_cell']['sensory_erev:0'].value
    dicName['rnn-sensory_mu'] = fdata['rnn']['rnn']['ltc_cell']['sensory_mu:0'].value
    dicName['rnn-sensory_sigma'] = fdata['rnn']['rnn']['ltc_cell']['sensory_sigma:0'].value
    dicName['rnn-sensory_w'] = fdata['rnn']['rnn']['ltc_cell']['sensory_w:0'].value
    dicName['rnn-sigma'] = fdata['rnn']['rnn']['ltc_cell']['sigma:0'].value
    dicName['rnn-vleak'] = fdata['rnn']['rnn']['ltc_cell']['vleak:0'].value
    dicName['rnn-w'] = fdata['rnn']['rnn']['ltc_cell']['w:0'].value
    dicName['time_distributed-timedistributed-kernel'] = fdata['time_distributed']['time_distributed'][
        'kernel:0'].value
    dicName['time_distributed_1-timedistributed_1-kernel'] = fdata['time_distributed_1']['time_distributed_1'][
        'kernel:0'].value
    dicName['time_distributed_3-timedistributed_3-kernel'] = fdata['time_distributed_3']['time_distributed_3'][
        'kernel:0'].value
    dicName['time_distributed_5-timedistributed_5-kernel'] = fdata['time_distributed_5']['time_distributed_5'][
        'kernel:0'].value
    dicName['time_distributed_7-timedistributed_7-kernel'] = fdata['time_distributed_7']['time_distributed_7'][
        'kernel:0'].value
    dicName['time_distributed'] = fdata['time_distributed']['time_distributed']['bias:0'].value
    dicName['time_distributed_1'] = fdata['time_distributed_1']['time_distributed_1']['bias:0'].value
    dicName['time_distributed_3'] = fdata['time_distributed_3']['time_distributed_3']['bias:0'].value
    dicName['time_distributed_5'] = fdata['time_distributed_5']['time_distributed_5']['bias:0'].value
    dicName['time_distributed_7'] = fdata['time_distributed_7']['time_distributed_7']['bias:0'].value

# TODO 判断更新是否有效 ：
# TODO diff1=(target-Martrix1) diff2=(target-(G+.updated-G-.updated)) 1
# TODO iff : diff1 < diff2 2
# TODO do Matrix3.thisNode = 0 (这个时候Matrix3被放在pushinfo里) 3

def isValid(NvalidNum,validNum):
        info = []
    # rnn-cm
        for i in range(21):
            diff1= targetinfo.get('rnn-cm')[i]-Matrix1.get('rnn-cm')[i]
            diff2= targetinfo.get('rnn-cm')[i]-(Gplus_T_Info.get('rnn-cm')[i]-Gmius_T_Info.get('rnn-cm')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-cm'][i]=0
                if diff1!=0.0:
                    info.append("target: "+str((targetinfo.get('rnn-cm')[i]).astype(float))+ " old: "+str((Matrix1.get('rnn-cm')[i]).astype(float)) + " wanted: "+str((Gplus_T_Info.get('rnn-cm')[i]-Gmius_T_Info.get('rnn-cm')[i]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" cm"+"\n")
                NvalidNum=NvalidNum+1
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1


    # rnn-erev
        for i in range(21):
            for j in range(21):
                diff1 = targetinfo.get('rnn-erev')[i][j] - Matrix1.get('rnn-erev')[i][j]
                diff2 = targetinfo.get('rnn-erev')[i][j] - (Gplus_T_Info.get('rnn-erev')[i][j] - Gmius_T_Info.get('rnn-erev')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-erev'][i][j] = 0
                    NvalidNum = NvalidNum + 1
                    if diff1 != 0.0:
                        info.append(
                        "target: " + str((targetinfo.get('rnn-erev')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-erev')[i][j]).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-erev')[i][j] - Gmius_T_Info.get('rnn-erev')[i][j]).astype(float)) +"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" erev"+ "\n")
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1


    # rnn-gleak
        for i in range(21):
            diff1 = targetinfo.get('rnn-gleak')[i] - Matrix1.get('rnn-gleak')[i]
            diff2 = targetinfo.get('rnn-gleak')[i]- (Gplus_T_Info.get('rnn-gleak')[i] - Gmius_T_Info.get('rnn-gleak')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-gleak'][i] = 0
                NvalidNum=NvalidNum+1
                if diff1!=0.0:
                    info.append(
                        "target: " + str((targetinfo.get('rnn-gleak')[i]).astype(float)) + " old: " + str((Matrix1.get('rnn-gleak')[i]).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-gleak')[i] - Gmius_T_Info.get('rnn-gleak')[i]).astype(float)) +"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" gleak"+ "\n")
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1

    # rnn-input_b
        for i in range(32):
            diff1 = targetinfo.get('rnn-input_b')[i] - Matrix1.get('rnn-input_b')[i]
            diff2 = targetinfo.get('rnn-input_b')[i] - (Gplus_T_Info.get('rnn-input_b')[i] - Gmius_T_Info.get('rnn-input_b')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-input_b'][i] = 0
                NvalidNum=NvalidNum+1
                if diff1!=0.0:
                    info.append(
                        "target: " + str((targetinfo.get('rnn-input_b')[i]).astype(float)) + " old: " + str((Matrix1.get('rnn-input_b')[i]).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-input_b')[i] - Gmius_T_Info.get('rnn-input_b')[i]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" input_b" + "\n")
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1


    # rnn-input_w
        for i in range(32):
            diff1 = targetinfo.get('rnn-input_w')[i] - Matrix1.get('rnn-input_w')[i]
            diff2 = targetinfo.get('rnn-input_w')[i] - (Gplus_T_Info.get('rnn-input_w')[i] - Gmius_T_Info.get('rnn-input_w')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-input_w'][i] = 0
                NvalidNum=NvalidNum+1
                if diff1!=0.0:
                    info.append(
                        "target: " +str((targetinfo.get('rnn-input_w')[i]).astype(float)) + " old: " + str((Matrix1.get('rnn-input_w')[
                            i]).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-input_w')[i] - Gmius_T_Info.get('rnn-input_w')[i]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" input_w"+ "\n")
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1


    # rnn-mu
        for i in range(21):
            for j in range(21):
                diff1 = targetinfo.get('rnn-mu')[i][j] - Matrix1.get('rnn-mu')[i][j]
                diff2 = targetinfo.get('rnn-mu')[i][j] - (Gplus_T_Info.get('rnn-mu')[i][j] - Gmius_T_Info.get('rnn-mu')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-mu'][i][j] = 0
                    NvalidNum=NvalidNum+1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-mu')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-mu')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-mu')[i][j] - Gmius_T_Info.get('rnn-mu')[i][j]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" mu"+ "\n")
                if (abs(diff1)>abs(diff2)):
                    validNum=validNum+1


    # rnn-output_b
        for i in range(1):
            diff1 = targetinfo.get('rnn-output_b')[i] - Matrix1.get('rnn-output_b')[i]
            diff2 = targetinfo.get('rnn-output_b')[i] - (Gplus_T_Info.get('rnn-output_b')[i] - Gmius_T_Info.get('rnn-output_b')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-output_b'][i] = 0
                NvalidNum=NvalidNum+1
                if diff1!=0.0:
                    info.append(
                        "target: " +str((targetinfo.get('rnn-output_b')[i]).astype(float))+ " old: " + str(((Matrix1.get('rnn-output_b')[i])).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-output_b')[i] - Gmius_T_Info.get('rnn-output_b')[i]).astype(float)) +"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" output_b"+ "\n")
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1


    # rnn-output_w
        for i in range(1):
            diff1 = targetinfo.get('rnn-output_w')[i] - Matrix1.get('rnn-output_w')[i]
            diff2 = targetinfo.get('rnn-output_w')[i] - (Gplus_T_Info.get('rnn-output_w')[i] - Gmius_T_Info.get('rnn-output_w')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-output_w'][i] = 0
                NvalidNum=NvalidNum+1
                if diff1!=0.0:
                    info.append(
                        "target: " + str((targetinfo.get('rnn-output_w')[i]).astype(float)) + " old: " + str((Matrix1.get('rnn-output_w')[i]).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-output_w')[i] - Gmius_T_Info.get('rnn-output_w')[i]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" output_w"+ "\n")
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1


    # rnn-sensory_erev
        for i in range(32):
            for j in range(21):
                diff1 = targetinfo.get('rnn-sensory_erev')[i][j] - Matrix1.get('rnn-sensory_erev')[i][j]
                diff2 = targetinfo.get('rnn-sensory_erev')[i][j] - (Gplus_T_Info.get('rnn-sensory_erev')[i][j] - Gmius_T_Info.get('rnn-sensory_erev')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-sensory_erev'][i][j] = 0
                    NvalidNum=NvalidNum+1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-sensory_erev')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-sensory_erev')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-sensory_erev')[i][j] - Gmius_T_Info.get('rnn-sensory_erev')[i][j]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float)) +" sensory_erev"+ "\n")
                if (abs(diff1)>abs(diff2)):
                    validNum=validNum+1


    # rnn-sensory_mu
        for i in range(32):
            for j in range(21):
                diff1 = targetinfo.get('rnn-sensory_mu')[i][j] - Matrix1.get('rnn-sensory_mu')[i][j]
                diff2 = targetinfo.get('rnn-sensory_mu')[i][j] - (
                            Gplus_T_Info.get('rnn-sensory_mu')[i][j] - Gmius_T_Info.get('rnn-sensory_mu')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-sensory_mu'][i][j] = 0
                    NvalidNum=NvalidNum+1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-sensory_mu')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-sensory_mu')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-sensory_mu')[i][j] - Gmius_T_Info.get('rnn-sensory_mu')[i][j]).astype(float)) +"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" sensory_mu"+ "\n")
                if (abs(diff1)>abs(diff2)):
                    validNum=validNum+1


    # rnn-sensory_sigma
        for i in range(32):
            for j in range(21):
                diff1 = targetinfo.get('rnn-sensory_sigma')[i][j] - Matrix1.get('rnn-sensory_sigma')[i][j]
                diff2 = targetinfo.get('rnn-sensory_sigma')[i][j] - (
                        Gplus_T_Info.get('rnn-sensory_sigma')[i][j] - Gmius_T_Info.get('rnn-sensory_sigma')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-sensory_sigma'][i][j] = 0
                    NvalidNum = NvalidNum + 1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-sensory_sigma')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-sensory_sigma')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-sensory_sigma')[i][j] - Gmius_T_Info.get('rnn-sensory_sigma')[i][j]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" sensory_sigma" + "\n")
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1


    # rnn-sensory_w
        for i in range(32):
            for j in range(21):
                diff1 = targetinfo.get('rnn-sensory_w')[i][j] - Matrix1.get('rnn-sensory_w')[i][j]
                diff2 = targetinfo.get('rnn-sensory_w')[i][j] - (
                        Gplus_T_Info.get('rnn-sensory_w')[i][j] - Gmius_T_Info.get('rnn-sensory_w')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-sensory_w'][i][j] = 0
                    NvalidNum = NvalidNum + 1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-sensory_w')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-sensory_w')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-sensory_w')[i][j] - Gmius_T_Info.get('rnn-sensory_w')[i][j]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" sensory_w" + "\n")
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1


    # rnn-sigma
        for i in range(21):
            for j in range(21):
                diff1 = targetinfo.get('rnn-sigma')[i][j] - Matrix1.get('rnn-sigma')[i][j]
                diff2 = targetinfo.get('rnn-sigma')[i][j] - (
                        Gplus_T_Info.get('rnn-sigma')[i][j] - Gmius_T_Info.get('rnn-sigma')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-sigma'][i][j] = 0
                    NvalidNum = NvalidNum + 1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-sigma')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-sigma')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-sigma')[i][j] - Gmius_T_Info.get('rnn-sigma')[i][j]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" sigma" + "\n")
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1


    # rnn-vleak
        for i in range(21):
            diff1 = targetinfo.get('rnn-vleak')[i] - Matrix1.get('rnn-vleak')[i]
            diff2 = targetinfo.get('rnn-vleak')[i] - (Gplus_T_Info.get('rnn-vleak')[i] - Gmius_T_Info.get('rnn-vleak')[i])
            if (abs(diff1)<abs(diff2)):
                pushinfo['rnn-vleak'][i] = 0
                NvalidNum=NvalidNum+1
                if diff1!=0.0:
                    info.append(
                        "target: " + str((targetinfo.get('rnn-vleak')[i]).astype(float)) + " old: " + str((Matrix1.get('rnn-vleak')[i]).astype(float)) + " wanted: " +
                        str((Gplus_T_Info.get('rnn-vleak')[i] - Gmius_T_Info.get('rnn-vleak')[i]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float)) +" vleak"+ "\n")
            if (abs(diff1)>abs(diff2)):
                validNum=validNum+1


    # rnn-w
        for i in range(21):
            for j in range(21):
                diff1 = targetinfo.get('rnn-w')[i][j] - Matrix1.get('rnn-w')[i][j]
                diff2 = targetinfo.get('rnn-w')[i][j] - (Gplus_T_Info.get('rnn-w')[i][j] - Gmius_T_Info.get('rnn-w')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['rnn-w'][i][j] = 0
                    NvalidNum = NvalidNum + 1
                    if diff1 != 0.0:
                        info.append(
                            "target: " + str((targetinfo.get('rnn-w')[i][j]).astype(float)) + " old: " + str((Matrix1.get('rnn-w')[i][j]).astype(float)) + " wanted: " +
                            str((Gplus_T_Info.get('rnn-w')[i][j] - Gmius_T_Info.get('rnn-w')[i][j]).astype(float))+"|"+str(diff1.astype(float))+"|"+str(diff2.astype(float))+" w" + "\n")
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1


    # time_distributed-timedistributed-kernel
        for i in range(5):
            for j in range(1):
                for k in range(18):
                    diff1 = targetinfo.get('time_distributed-timedistributed-kernel')[i][j][k] - Matrix1.get('time_distributed-timedistributed-kernel')[i][j][k]
                    diff2 = targetinfo.get('time_distributed-timedistributed-kernel')[i][j][k] - (Gplus_T_Info.get('time_distributed-timedistributed-kernel')[i][j][k] - Gmius_T_Info.get('time_distributed-timedistributed-kernel')[i][j][k])
                    if (abs(diff1)<abs(diff2)):
                        pushinfo['time_distributed-timedistributed-kernel'][i][j][k] = 0
                        NvalidNum = NvalidNum + 1
                    if (abs(diff1) > abs(diff2)):
                        validNum = validNum + 1

    # time_distributed - bias
        for i in range(18):
            diff1 = targetinfo.get('time_distributed')[i] - Matrix1.get('time_distributed')[i]
            diff2 = targetinfo.get('time_distributed')[i] - (Gplus_T_Info.get('time_distributed')[i] - Gmius_T_Info.get('time_distributed')[i])
            if (abs(diff1) < abs(diff2)):
                pushinfo['time_distributed'][i] = 0
                NvalidNum = NvalidNum + 1
            if (abs(diff1) > abs(diff2)):
                validNum = validNum + 1


    # time_distributed_1-timedistributed_1-kernel
        for i in range(5):
            for j in range(18):
                for k in range(20):
                    diff1 = targetinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] - Matrix1.get('time_distributed_1-timedistributed_1-kernel')[i][j][k]
                    diff2 = targetinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] - (Gplus_T_Info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] -Gmius_T_Info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k])
                    if (abs(diff1)<abs(diff2)):
                        pushinfo['time_distributed_1-timedistributed_1-kernel'][i][j][k] = 0
                        NvalidNum = NvalidNum + 1
                    if (abs(diff1) > abs(diff2)):
                        validNum = validNum + 1

    # time_distributed_1 - bias
        for i in range(20):
            diff1 = targetinfo.get('time_distributed_1')[i] - Matrix1.get('time_distributed_1')[i]
            diff2 = targetinfo.get('time_distributed_1')[i] - (Gplus_T_Info.get('time_distributed_1')[i] - Gmius_T_Info.get('time_distributed_1')[i])
            if (abs(diff1) < abs(diff2)):
                pushinfo['time_distributed_1'][i] = 0
                NvalidNum = NvalidNum + 1
            if (abs(diff1) > abs(diff2)):
                validNum = validNum + 1

    # time_distributed_3-timedistributed_3-kernel
        for i in range(5):
            for j in range(20):
                for k in range(22):
                    diff1 = targetinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] - Matrix1.get('time_distributed_3-timedistributed_3-kernel')[i][j][k]
                    diff2 = targetinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] - (Gplus_T_Info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] -Gmius_T_Info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k])
                    if (abs(diff1)<abs(diff2)):
                        pushinfo['time_distributed_3-timedistributed_3-kernel'][i][j][k] = 0
                        NvalidNum = NvalidNum + 1
                    if (abs(diff1) > abs(diff2)):
                        validNum = validNum + 1

    # time_distributed_3 - bias
            for i in range(22):
                diff1 = targetinfo.get('time_distributed_3')[i] - Matrix1.get('time_distributed_3')[i]
                diff2 = targetinfo.get('time_distributed_3')[i] - (Gplus_T_Info.get('time_distributed_3')[i] - Gmius_T_Info.get('time_distributed_3')[i])
                if (abs(diff1) < abs(diff2)):
                    pushinfo['time_distributed_3'][i] = 0
                    NvalidNum = NvalidNum + 1
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1

    # time_distributed_5-timedistributed_5-kernel
        for i in range(5):
            for j in range(22):
                for k in range(24):
                    diff1 = targetinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] - Matrix1.get('time_distributed_5-timedistributed_5-kernel')[i][j][k]
                    diff2 = targetinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] - (Gplus_T_Info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] -Gmius_T_Info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k])
                    if (abs(diff1)<abs(diff2)):
                        pushinfo['time_distributed_5-timedistributed_5-kernel'][i][j][k] = 0
                        NvalidNum = NvalidNum + 1
                    if (abs(diff1) > abs(diff2)):
                        validNum = validNum + 1

     # time_distributed_5 - bias
        for i in range(24):
            diff1 = targetinfo.get('time_distributed_5')[i] - Matrix1.get('time_distributed_5')[i]
            diff2 = targetinfo.get('time_distributed_5')[i] - (Gplus_T_Info.get('time_distributed_5')[i] - Gmius_T_Info.get('time_distributed_5')[i])
            if (abs(diff1) < abs(diff2)):
                pushinfo['time_distributed_5'][i] = 0
                NvalidNum = NvalidNum + 1
            if (abs(diff1) > abs(diff2)):
                validNum = validNum + 1


    # time_distributed_7-timedistributed_7-kernel
        for i in range(384):
            for j in range(32):
                diff1 = targetinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j] - Matrix1.get('time_distributed_7-timedistributed_7-kernel')[i][j]
                diff2 = targetinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j] - (Gplus_T_Info.get('time_distributed_7-timedistributed_7-kernel')[i][j] -Gmius_T_Info.get('time_distributed_7-timedistributed_7-kernel')[i][j])
                if (abs(diff1)<abs(diff2)):
                    pushinfo['time_distributed_7-timedistributed_7-kernel'][i][j] = 0
                    NvalidNum = NvalidNum + 1
                if (abs(diff1) > abs(diff2)):
                    validNum = validNum + 1

    # time_distributed_7 - bias
        for i in range(32):
            diff1 = targetinfo.get('time_distributed_7')[i] - Matrix1.get('time_distributed_7')[i]
            diff2 = targetinfo.get('time_distributed_7')[i] - (
                        Gplus_T_Info.get('time_distributed_7')[i] - Gmius_T_Info.get('time_distributed_7')[i])
            if (abs(diff1) < abs(diff2)):
                pushinfo['time_distributed_7'][i] = 0
                NvalidNum = NvalidNum + 1
            if (abs(diff1) > abs(diff2)):
                validNum = validNum + 1
        if validNum==0:
            with open('information.txt', 'a+') as File:
                for any in info:
                    File.write(any)
                File.write("................................................................."+"\n")
        return  NvalidNum,validNum


# TODO 获取目标矩阵的权重信息
def getTargetMar(FileName):
    fdata = h5py.File(FileName, 'r+')
    targetinfo['rnn-cm'] = fdata['rnn']['rnn']['ltc_cell']['cm:0'].value
    targetinfo['rnn-erev'] = fdata['rnn']['rnn']['ltc_cell']['erev:0'].value
    targetinfo['rnn-gleak'] = fdata['rnn']['rnn']['ltc_cell']['gleak:0'].value
    targetinfo['rnn-input_b'] = fdata['rnn']['rnn']['ltc_cell']['input_b:0'].value
    targetinfo['rnn-input_w'] = fdata['rnn']['rnn']['ltc_cell']['input_w:0'].value
    targetinfo['rnn-mu'] = fdata['rnn']['rnn']['ltc_cell']['mu:0'].value
    targetinfo['rnn-output_b'] = fdata['rnn']['rnn']['ltc_cell']['output_b:0'].value
    targetinfo['rnn-output_w'] = fdata['rnn']['rnn']['ltc_cell']['output_w:0'].value
    targetinfo['rnn-sensory_erev'] = fdata['rnn']['rnn']['ltc_cell']['sensory_erev:0'].value
    targetinfo['rnn-sensory_mu'] = fdata['rnn']['rnn']['ltc_cell']['sensory_mu:0'].value
    targetinfo['rnn-sensory_sigma'] = fdata['rnn']['rnn']['ltc_cell']['sensory_sigma:0'].value
    targetinfo['rnn-sensory_w'] = fdata['rnn']['rnn']['ltc_cell']['sensory_w:0'].value
    targetinfo['rnn-sigma'] = fdata['rnn']['rnn']['ltc_cell']['sigma:0'].value
    targetinfo['rnn-vleak'] = fdata['rnn']['rnn']['ltc_cell']['vleak:0'].value
    targetinfo['rnn-w'] = fdata['rnn']['rnn']['ltc_cell']['w:0'].value
    targetinfo['time_distributed-timedistributed-kernel'] = fdata['time_distributed']['time_distributed'][
        'kernel:0'].value
    targetinfo['time_distributed_1-timedistributed_1-kernel'] = fdata['time_distributed_1']['time_distributed_1'][
        'kernel:0'].value
    targetinfo['time_distributed_3-timedistributed_3-kernel'] = fdata['time_distributed_3']['time_distributed_3'][
        'kernel:0'].value
    targetinfo['time_distributed_5-timedistributed_5-kernel'] = fdata['time_distributed_5']['time_distributed_5'][
        'kernel:0'].value
    targetinfo['time_distributed_7-timedistributed_7-kernel'] = fdata['time_distributed_7']['time_distributed_7'][
        'kernel:0'].value
    targetinfo['time_distributed'] = fdata['time_distributed']['time_distributed']['bias:0'].value
    targetinfo['time_distributed_1'] = fdata['time_distributed_1']['time_distributed_1']['bias:0'].value
    targetinfo['time_distributed_3'] = fdata['time_distributed_3']['time_distributed_3']['bias:0'].value
    targetinfo['time_distributed_5'] = fdata['time_distributed_5']['time_distributed_5']['bias:0'].value
    targetinfo['time_distributed_7'] = fdata['time_distributed_7']['time_distributed_7']['bias:0'].value


#TODO 求两个矩阵的差并保存在 pushinfo 中
def generate_diff_to_pushinfo(Matrix_1,Matrix_2,step):
    pushinfo.clear()
    push_file_data_to_infodict(Matrix_1,Matrix_2)
    #rnn-cm
    G1cm=fileinfo.get('rnn-cm')
    G2cm=G2info.get('rnn-cm')
    d = np.empty(shape=(21,), dtype='float32')
    if step==1:
        for i in range(21):
            d[i] = float(G1cm[i]-G2cm[i])
    if step==3:
        for i in range(21):
            if float(G1cm[i]-G2cm[i])>0:
                d[i] = 1
            elif float(G1cm[i]-G2cm[i])<0:
                d[i] = -1
            else:
                d[i] = 0
    pushinfo['rnn-cm']=d

    #rnn-erev:0
    G1erev=fileinfo.get('rnn-erev')
    G2erev=G2info.get('rnn-erev')
    derev=np.empty(shape=(21,21), dtype='float32')
    if step==1:
        for i in range(21):
            for j in range(21):
                derev[i][j] = float(G1erev[i][j]-G2erev[i][j])
    if step == 3:
        for i in range(21):
            for j in range(21):
                if float(G1erev[i][j]-G2erev[i][j])>0:
                    derev[i][j] = 1
                elif float(G1erev[i][j]-G2erev[i][j])<0:
                    derev[i][j] = -1
                else:
                    derev[i][j] = 0
    pushinfo['rnn-erev'] = derev

    #rnn-gleak
    G1gleak = fileinfo.get('rnn-gleak')
    G2gleak = G2info.get('rnn-gleak')
    dgleak = np.empty(shape=(21,), dtype='float32')
    if step == 1:
        for i in range(21):
            dgleak[i] = float(G1gleak[i] - G2gleak[i])
    if step == 3:
        for i in range(21):
            if float(G1gleak[i] - G2gleak[i]) > 0:
                dgleak[i] = 1
            elif float(G1gleak[i] - G2gleak[i]) < 0:
                dgleak[i] = -1
            else :
                dgleak[i] = 0
    pushinfo['rnn-gleak'] = dgleak

    #rnn-input_b
    G1input_b = fileinfo.get('rnn-input_b')
    G2input_b = G2info.get('rnn-input_b')
    dinput_b = np.empty(shape=(32,), dtype='float32')
    if step == 1:
        for i in range(32):
            dinput_b[i] = float(G1input_b[i] - G2input_b[i])
    if step == 3:
        for i in range(32):
            if float(G1input_b[i] - G2input_b[i]) > 0:
                dinput_b[i] = 1
            elif float(G1input_b[i] - G2input_b[i]) < 0:
                dinput_b[i] = -1
            else:
                dinput_b[i] = 0
    pushinfo['rnn-input_b'] = dinput_b

    # rnn-input_w
    G1input_w = fileinfo.get('rnn-input_w')
    G2input_w = G2info.get('rnn-input_w')
    dinput_w = np.empty(shape=(32,), dtype='float32')
    if step == 1:
        for i in range(32):
            dinput_w[i] = float(G1input_w[i] - G2input_w[i])
    if step == 3:
        for i in range(32):
            if float(G1input_w[i] - G2input_w[i]) > 0:
                dinput_w[i] = 1
            elif float(G1input_w[i] - G2input_w[i]) < 0:
                dinput_w[i] = -1
            else:
                dinput_w[i] = 0
    pushinfo['rnn-input_w'] = dinput_w

    # rnn-mu:0
    G1mu = fileinfo.get('rnn-mu')
    G2mu = G2info.get('rnn-mu')
    dmu = np.empty(shape=(21, 21), dtype='float32')
    if step == 1:
        for i in range(21):
            for j in range(21):
                dmu[i][j] = float(G1mu[i][j] - G2mu[i][j])
    if step == 3:
        for i in range(21):
            for j in range(21):
                if float(G1mu[i][j] - G2mu[i][j]) > 0:
                    dmu[i][j] = 1
                elif float(G1mu[i][j] - G2mu[i][j]) < 0:
                    dmu[i][j] = -1
                else:
                    dmu[i][j] = 0
    pushinfo['rnn-mu'] = dmu

    # rnn-output_b
    G1output_b = fileinfo.get('rnn-output_b')
    G2output_b = G2info.get('rnn-output_b')
    doutput_b = np.empty(shape=(1,), dtype='float32')
    if step == 1:
        for i in range(1):
            doutput_b[i] = float(G1output_b[i] - G2output_b[i])
    if step == 3:
        for i in range(1):
            if float(G1output_b[i] - G2output_b[i]) > 0:
                doutput_b[i] = 1
            elif float(G1output_b[i] - G2output_b[i]) < 0:
                doutput_b[i] = -1
            else:
                doutput_b[i] = 0
    pushinfo['rnn-output_b'] = doutput_b

    # rnn-output_w
    G1output_w = fileinfo.get('rnn-output_w')
    G2output_w = G2info.get('rnn-output_w')
    doutput_w = np.empty(shape=(1,), dtype='float32')
    if step == 1:
        for i in range(1):
            doutput_w[i] = float(G1output_w[i] - G2output_w[i])
    if step == 3:
        for i in range(1):
            if float(G1output_w[i] - G2output_w[i]) > 0:
                doutput_w[i] = 1
            elif float(G1output_w[i] - G2output_w[i]) < 0:
                doutput_w[i] = -1
            else:
                doutput_w[i] = 0
    pushinfo['rnn-output_w'] = doutput_w

    # rnn-sensory_erev
    G1sensory_erev = fileinfo.get('rnn-sensory_erev')
    G2sensory_erev = G2info.get('rnn-sensory_erev')
    dsensory_erev = np.empty(shape=(32, 21), dtype='float32')
    if step == 1:
        for i in range(32):
            for j in range(21):
                dsensory_erev[i][j] = float(G1sensory_erev[i][j] - G2sensory_erev[i][j])
    if step == 3:
        for i in range(32):
            for j in range(21):
                if float(G1sensory_erev[i][j] - G2sensory_erev[i][j]) > 0:
                    dsensory_erev[i][j] = 1
                elif float(G1sensory_erev[i][j] - G2sensory_erev[i][j]) < 0:
                    dsensory_erev[i][j] = -1
                else:
                    dsensory_erev[i][j] = 0
    pushinfo['rnn-sensory_erev'] = dsensory_erev

    # rnn-sensory_mu
    G1sensory_mu = fileinfo.get('rnn-sensory_mu')
    G2sensory_mu = G2info.get('rnn-sensory_mu')
    dsensory_mu = np.empty(shape=(32, 21), dtype='float32')
    if step == 1:
        for i in range(32):
            for j in range(21):
                dsensory_mu[i][j] = float(G1sensory_mu[i][j] - G2sensory_mu[i][j])
    if step == 3:
        for i in range(32):
            for j in range(21):
                if float(G1sensory_mu[i][j] - G2sensory_mu[i][j]) > 0:
                    dsensory_mu[i][j] = 1
                elif float(G1sensory_mu[i][j] - G2sensory_mu[i][j]) < 0:
                    dsensory_mu[i][j] = -1
                else:
                    dsensory_mu[i][j] = 0
    pushinfo['rnn-sensory_mu'] = dsensory_mu

    # rnn-sensory_sigma
    G1sensory_sigma = fileinfo.get('rnn-sensory_sigma')
    G2sensory_sigma = G2info.get('rnn-sensory_sigma')
    dsensory_sigma = np.empty(shape=(32, 21), dtype='float32')
    if step == 1:
        for i in range(32):
            for j in range(21):
                dsensory_sigma[i][j] = float(G1sensory_sigma[i][j] - G2sensory_sigma[i][j])
    if step == 3:
        for i in range(32):
            for j in range(21):
                if float(G1sensory_sigma[i][j] - G2sensory_sigma[i][j]) > 0:
                    dsensory_sigma[i][j] = 1
                elif float(G1sensory_sigma[i][j] - G2sensory_sigma[i][j]) < 0:
                    dsensory_sigma[i][j] = -1
                else:
                    dsensory_sigma[i][j] = 0
    pushinfo['rnn-sensory_sigma'] = dsensory_sigma

    # rnn-sensory_w
    G1sensory_w = fileinfo.get('rnn-sensory_w')
    G2sensory_w = G2info.get('rnn-sensory_w')
    dsensory_w = np.empty(shape=(32, 21), dtype='float32')
    if step == 1:
        for i in range(32):
            for j in range(21):
                dsensory_w[i][j] = float(G1sensory_w[i][j] - G2sensory_w[i][j])
    if step == 3:
        for i in range(32):
            for j in range(21):
                if float(G1sensory_w[i][j] - G2sensory_w[i][j]) > 0:
                    dsensory_w[i][j] = 1
                elif float(G1sensory_w[i][j] - G2sensory_w[i][j]) < 0:
                    dsensory_w[i][j] = -1
                else:
                    dsensory_w[i][j] = 0
    pushinfo['rnn-sensory_w'] = dsensory_w

    # rnn-sigma
    G1sigma = fileinfo.get('rnn-sigma')
    G2sigma = G2info.get('rnn-sigma')
    dsigma = np.empty(shape=(21, 21), dtype='float32')
    if step == 1:
        for i in range(21):
            for j in range(21):
                dsigma[i][j] = float(G1sigma[i][j] - G2sigma[i][j])
    if step == 3:
        for i in range(21):
            for j in range(21):
                if float(G1sigma[i][j] - G2sigma[i][j]) > 0:
                    dsigma[i][j] = 1
                elif float(G1sigma[i][j] - G2sigma[i][j]) < 0:
                    dsigma[i][j] = -1
                else:
                    dsigma[i][j] = 0
    pushinfo['rnn-sigma'] = dsigma

    #rnn-vleak
    G1vleak = fileinfo.get('rnn-vleak')
    G2vleak = G2info.get('rnn-vleak')
    dvleak = np.empty(shape=(21,), dtype='float32')
    if step == 1:
        for i in range(21):
            dvleak[i] = float(G1vleak[i] - G2vleak[i])
    if step == 3:
        for i in range(21):
            if float(G1vleak[i] - G2vleak[i]) > 0:
                dvleak[i] = 1
            elif float(G1vleak[i] - G2vleak[i]) < 0:
                dvleak[i] = -1
            else:
                dvleak[i] = 0
    pushinfo['rnn-vleak'] = dvleak

    # rnn-w:0
    G1w = fileinfo.get('rnn-w')
    G2w = G2info.get('rnn-w')
    dw = np.empty(shape=(21, 21), dtype='float32')
    if step == 1:
        for i in range(21):
            for j in range(21):
                dw[i][j] = float(G1w[i][j] - G2w[i][j])
    if step == 3:
        for i in range(21):
            for j in range(21):
                if float(G1w[i][j] - G2w[i][j]) > 0:
                    dw[i][j] = 1
                elif float(G1w[i][j] - G2w[i][j]) < 0:
                    dw[i][j] = -1
                else:
                    dw[i][j] = 0
    pushinfo['rnn-w'] = dw

    # time_distributed-timedistributed-kernel
    G1timedistributed = fileinfo.get('time_distributed-timedistributed-kernel')
    G2timedistributed = G2info.get('time_distributed-timedistributed-kernel')
    dtimedistributed = np.empty(shape=(5,1,18), dtype='float32')
    if step == 1:
        for i in range(5):
            for j in range(1):
                for k in range(18):
                    dtimedistributed[i][j][k] = float(G1timedistributed[i][j][k]-G2timedistributed[i][j][k])
    if step == 3:
        for i in range(5):
            for j in range(1):
                for k in range(18):
                   if float(G1timedistributed[i][j][k]-G2timedistributed[i][j][k])>0:
                       dtimedistributed[i][j][k]=1
                   elif float(G1timedistributed[i][j][k]-G2timedistributed[i][j][k])<0:
                       dtimedistributed[i][j][k]=-1
                   else:
                       dtimedistributed[i][j][k]=0
    pushinfo['time_distributed-timedistributed-kernel']=dtimedistributed

    # time_distributed - bias
    G1time_distributed = fileinfo.get('time_distributed')
    G2time_distributed = G2info.get('time_distributed')
    time_distributed = np.empty(shape=(18,), dtype='float32')
    if step == 1:
        for i in range(18):
            time_distributed[i] = float(G1time_distributed[i]-G2time_distributed[i])
    if step == 3:
        for i in range(18):
            if float(G1time_distributed[i] - G2time_distributed[i]) > 0:
                time_distributed[i] = 1
            elif float(G1time_distributed[i] - G2time_distributed[i]) < 0:
                time_distributed[i] = -1
            else:
                time_distributed[i] = 0
    pushinfo['time_distributed'] = time_distributed



    # time_distributed_1-timedistributed_1-kernel
    G1timedistributed1 = fileinfo.get('time_distributed_1-timedistributed_1-kernel')
    G2timedistributed1 = G2info.get('time_distributed_1-timedistributed_1-kernel')
    dtimedistributed1 = np.empty(shape=(5,18,20), dtype='float32')
    if step == 1:
        for i in range(5):
            for j in range(18):
                for k in range(20):
                    dtimedistributed1[i][j][k] = float(G1timedistributed1[i][j][k]-G2timedistributed1[i][j][k])
    if step == 3:
        for i in range(5):
            for j in range(18):
                for k in range(20):
                   if float(G1timedistributed1[i][j][k]-G2timedistributed1[i][j][k])>0:
                       dtimedistributed1[i][j][k]=1
                   elif float(G1timedistributed1[i][j][k]-G2timedistributed1[i][j][k])<0:
                       dtimedistributed1[i][j][k]=-1
                   else:
                       dtimedistributed1[i][j][k]=0
    pushinfo['time_distributed_1-timedistributed_1-kernel']=dtimedistributed1

    # time_distributed_1 - bias
    G1time_distributed_1 = fileinfo.get('time_distributed_1')
    G2time_distributed_1= G2info.get('time_distributed_1')
    time_distributed_1 = np.empty(shape=(20,), dtype='float32')
    if step == 1:
        for i in range(20):
            time_distributed_1[i] = float(G1time_distributed_1[i]-G2time_distributed_1[i])
    if step == 3:
        for i in range(20):
            if float(G1time_distributed_1[i]-G2time_distributed_1[i]) > 0:
                time_distributed_1[i] = 1
            elif float(G1time_distributed_1[i]-G2time_distributed_1[i]) < 0:
                time_distributed_1[i] = -1
            else:
                time_distributed_1[i] = 0
    pushinfo['time_distributed_1'] = time_distributed_1

    # time_distributed_3-timedistributed_3-kernel
    G1timedistributed3 = fileinfo.get('time_distributed_3-timedistributed_3-kernel')
    G2timedistributed3 = G2info.get('time_distributed_3-timedistributed_3-kernel')
    dtimedistributed3 = np.empty(shape=(5,20,22), dtype='float32')
    if step == 1:
        for i in range(5):
            for j in range(20):
                for k in range(22):
                    dtimedistributed3[i][j][k] = float(G1timedistributed3[i][j][k]-G2timedistributed3[i][j][k])
    if step == 3:
        for i in range(5):
            for j in range(20):
                for k in range(22):
                   if float(G1timedistributed3[i][j][k]-G2timedistributed3[i][j][k])>0:
                       dtimedistributed3[i][j][k]=1
                   elif float(G1timedistributed3[i][j][k]-G2timedistributed3[i][j][k])<0:
                       dtimedistributed3[i][j][k]=-1
                   else:
                       dtimedistributed3[i][j][k]=0
    pushinfo['time_distributed_3-timedistributed_3-kernel']=dtimedistributed3

    # time_distributed_3 - bias
    G1time_distributed_3 = fileinfo.get('time_distributed_3')
    G2time_distributed_3= G2info.get('time_distributed_3')
    time_distributed_3 = np.empty(shape=(22,), dtype='float32')
    if step == 1:
        for i in range(22):
            time_distributed_3[i] = float(G1time_distributed_3[i]-G2time_distributed_3[i])
    if step == 3:
        for i in range(22):
            if float(G1time_distributed_3[i]-G2time_distributed_3[i]) > 0:
                time_distributed_3[i] = 1
            elif float(G1time_distributed_3[i]-G2time_distributed_3[i]) < 0:
                time_distributed_3[i] = -1
            else:
                time_distributed_3[i] = 0
    pushinfo['time_distributed_3'] = time_distributed_3

    # time_distributed_5-timedistributed_5-kernel
    G1timedistributed5 = fileinfo.get('time_distributed_5-timedistributed_5-kernel')
    G2timedistributed5 = G2info.get('time_distributed_5-timedistributed_5-kernel')
    dtimedistributed5 = np.empty(shape=(5,22,24), dtype='float32')
    if step == 1:
        for i in range(5):
            for j in range(22):
                for k in range(24):
                    dtimedistributed5[i][j][k] = float(G1timedistributed5[i][j][k]-G2timedistributed5[i][j][k])
    if step == 3:
        for i in range(5):
            for j in range(22):
                for k in range(24):
                   if float(G1timedistributed5[i][j][k]-G2timedistributed5[i][j][k])>0:
                       dtimedistributed5[i][j][k]=1
                   elif float(G1timedistributed5[i][j][k]-G2timedistributed5[i][j][k])<0:
                       dtimedistributed5[i][j][k]=-1
                   else:
                       dtimedistributed5[i][j][k]=0
    pushinfo['time_distributed_5-timedistributed_5-kernel']=dtimedistributed5

    # time_distributed_5 - bias
    G1time_distributed_5 = fileinfo.get('time_distributed_5')
    G2time_distributed_5= G2info.get('time_distributed_5')
    time_distributed_5 = np.empty(shape=(24,), dtype='float32')
    if step == 1:
        for i in range(24):
            time_distributed_5[i] = float(G1time_distributed_5[i]-G2time_distributed_5[i])
    if step == 3:
        for i in range(24):
            if float(G1time_distributed_5[i]-G2time_distributed_5[i]) > 0:
                time_distributed_5[i] = 1
            elif float(G1time_distributed_5[i]-G2time_distributed_5[i]) < 0:
                time_distributed_5[i] = -1
            else:
                time_distributed_5[i] = 0
    pushinfo['time_distributed_5'] = time_distributed_5

    # time_distributed_7-timedistributed_7-kernel
    G1timedistributed7 = fileinfo.get('time_distributed_7-timedistributed_7-kernel')
    G2timedistributed7 = G2info.get('time_distributed_7-timedistributed_7-kernel')
    dtimedistributed7 = np.empty(shape=(384,32), dtype='float32')
    if step == 1:
        for i in range(384):
            for j in range(32):
                dtimedistributed7[i][j] = float(G1timedistributed7[i][j]-G2timedistributed7[i][j])
    if step == 3:
        for i in range(384):
            for j in range(32):
                if float(G1timedistributed7[i][j]-G2timedistributed7[i][j])>0:
                       dtimedistributed7[i][j]=1
                elif float(G1timedistributed7[i][j]-G2timedistributed7[i][j])<0:
                       dtimedistributed7[i][j]=-1
                else:
                       dtimedistributed7[i][j]=0
    pushinfo['time_distributed_7-timedistributed_7-kernel']=dtimedistributed7

    # time_distributed_7 - bias
    G1time_distributed_7 = fileinfo.get('time_distributed_7')
    G2time_distributed_7= G2info.get('time_distributed_7')
    time_distributed_7 = np.empty(shape=(32,), dtype='float32')
    if step == 1:
        for i in range(32):
            time_distributed_7[i] = float(G1time_distributed_7[i]-G2time_distributed_7[i])
    if step == 3:
        for i in range(32):
            if float(G1time_distributed_7[i]-G2time_distributed_7[i]) > 0:
                time_distributed_7[i] = 1
            elif float(G1time_distributed_7[i]-G2time_distributed_7[i]) < 0:
                time_distributed_7[i] = -1
            else:
                time_distributed_7[i] = 0
    pushinfo['time_distributed_7'] = time_distributed_7

# Plot the data
def plot_lidar(lidar, ax):
    # Helper function for plotting polar-based lidar data
    angles = np.linspace(-2.35, 2.35, len(lidar))
    x = lidar * np.cos(angles)
    y = lidar * np.sin(angles)
    ax.plot(y, x)
    ax.scatter([0], [0], marker="^", color="black")
    ax.set_xlim((-6, 6))
    ax.set_ylim((-2, 6))

#TODO 合成Matrix1 W=G+ - G-
def generate_Matrix1 ():
    generate_diff_to_pushinfo('G+.h5','G-.h5',step=1)
    write_data_into_file('Matrix1.h5',pushinfo)

#TODO 训练Matrix1 得到 Matrix2
def train_module():
    model.load_weights('Matrix1.h5')
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error",metrics=['accuracy'])
    print('Start training...........')
    # Train the model
    tf.compat.v1.experimental.output_all_intermediates(True)

    hist = model.fit(
        x=data, y=y_valid, batch_size=30, epochs=1, validation_data=(data, y_valid)
    )
    model.save_weights('Matrix2.h5')

#TODO 计算Matrix3 Matrix3=Matrix2-Matrix1
def generate_Matrix3():
    generate_diff_to_pushinfo('Matrix2.h5','Matrix1.h5',step=3)
    write_data_into_file('Matrix3.h5',pushinfo)

#TODO 使用忆阻器特性更新 G+和G-
def get_G_Update_info(num):
    pushinfo.clear()

    # rnn-cm
    d = np.empty(shape=(21,), dtype='float32')
    if num==1:
        for i in range(21):
            if G2info.get('rnn-cm')[i]==1:
                d[i] = GposI(fileinfo.get('rnn-cm')[i])
            if G2info.get('rnn-cm')[i]==-1:
                d[i] = GposD(fileinfo.get('rnn-cm')[i])
            if G2info.get('rnn-cm')[i] ==0:
                d[i] = fileinfo.get('rnn-cm')[i]
    if num==-1:
        for i in range(21):
            if G2info.get('rnn-cm')[i]==1:
                d[i]=GnegD(fileinfo.get('rnn-cm')[i])
            if G2info.get('rnn-cm')[i]==-1:
                d[i]= GnegI(fileinfo.get('rnn-cm')[i])
            if G2info.get('rnn-cm')[i]==0:
                d[i] = fileinfo.get('rnn-cm')[i]
    pushinfo['rnn-cm'] = d

    #rnn-erev:0
    derev=np.empty(shape=(21,21), dtype='float32')
    if num== 1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-erev')[i][j]==1:
                    derev[i][j]=GposI(fileinfo.get('rnn-erev')[i][j])
                if G2info.get('rnn-erev')[i][j]==-1:
                    derev[i][j]=GposD(fileinfo.get('rnn-erev')[i][j])
                if G2info.get('rnn-erev')[i][j]==0:
                    derev[i][j]=fileinfo.get('rnn-erev')[i][j]
    if num == -1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-erev')[i][j]==1:
                    derev[i][j]=GnegD(fileinfo.get('rnn-erev')[i][j])
                if G2info.get('rnn-erev')[i][j]==-1:
                    derev[i][j]=GnegI(fileinfo.get('rnn-erev')[i][j])
                if G2info.get('rnn-erev')[i][j]==0:
                    derev[i][j]=fileinfo.get('rnn-erev')[i][j]
    pushinfo['rnn-erev'] = derev

    #rnn-gleak
    dgleak = np.empty(shape=(21,), dtype='float32')
    if num == 1:
        for i in range(21):
            if G2info.get('rnn-gleak')[i] == 1:
                dgleak[i]= GposI(fileinfo.get('rnn-gleak')[i])
            if G2info.get('rnn-gleak')[i] == -1:
                dgleak[i] = GposD(fileinfo.get('rnn-gleak')[i])
            if G2info.get('rnn-gleak')[i] == 0:
                dgleak[i] = fileinfo.get('rnn-gleak')[i]
    if num == -1:
        for i in range(21):
            if G2info.get('rnn-gleak')[i] == 1:
                dgleak[i]= GnegD(fileinfo.get('rnn-gleak')[i])
            if G2info.get('rnn-gleak')[i] == -1:
                dgleak[i] = GnegI(fileinfo.get('rnn-gleak')[i])
            if G2info.get('rnn-gleak')[i] == 0:
                dgleak[i] = fileinfo.get('rnn-gleak')[i]
    pushinfo['rnn-gleak'] = dgleak

    #rnn-input_b
    dinput_b = np.empty(shape=(32,), dtype='float32')
    if num == 1:
        for i in range(32):
            if G2info.get('rnn-input_b')[i] == 1:
                dinput_b[i]= GposI(fileinfo.get('rnn-input_b')[i])
            if G2info.get('rnn-input_b')[i] == -1:
                dinput_b[i] = GposD(fileinfo.get('rnn-input_b')[i])
            if G2info.get('rnn-input_b')[i] == 0:
                dinput_b[i] = fileinfo.get('rnn-input_b')[i]
    if num == -1:
        for i in range(32):
            if G2info.get('rnn-input_b')[i] == 1:
                dinput_b[i]= GnegD(fileinfo.get('rnn-input_b')[i])
            if G2info.get('rnn-input_b')[i] == -1:
                dinput_b[i] = GnegI(fileinfo.get('rnn-input_b')[i])
            if G2info.get('rnn-input_b')[i] == 0:
                dinput_b[i] = fileinfo.get('rnn-input_b')[i]
    pushinfo['rnn-input_b'] = dinput_b

    # rnn-input_w
    dinput_w = np.empty(shape=(32,), dtype='float32')
    if num == 1:
        for i in range(32):
            if G2info.get('rnn-input_w')[i] == 1:
                dinput_w[i]= GposI(fileinfo.get('rnn-input_w')[i])
            if G2info.get('rnn-input_w')[i] == -1:
                dinput_w[i] = GposD(fileinfo.get('rnn-input_w')[i])
            if G2info.get('rnn-input_w')[i] == 0:
                dinput_w[i] = fileinfo.get('rnn-input_w')[i]
    if num == -1:
        for i in range(32):
            if G2info.get('rnn-input_w')[i] == 1:
                dinput_w[i]= GnegD(fileinfo.get('rnn-input_w')[i])
            if G2info.get('rnn-input_w')[i] == -1:
                dinput_w[i] = GnegI(fileinfo.get('rnn-input_w')[i])
            if G2info.get('rnn-input_w')[i] == 0:
                dinput_w[i] = fileinfo.get('rnn-input_w')[i]
    pushinfo['rnn-input_w'] = dinput_w

    # rnn-mu:0
    dmu = np.empty(shape=(21, 21), dtype='float32')
    if num == 1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-mu')[i][j] == 1:
                    dmu[i][j] = GposI(fileinfo.get('rnn-mu')[i][j])
                if G2info.get('rnn-mu')[i][j] == -1:
                    dmu[i][j] = GposD(fileinfo.get('rnn-mu')[i][j])
                if G2info.get('rnn-mu')[i][j] == 0:
                    dmu[i][j] = fileinfo.get('rnn-mu')[i][j]
    if num == -1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-mu')[i][j] == 1:
                    dmu[i][j] = GnegD(fileinfo.get('rnn-mu')[i][j])
                if G2info.get('rnn-mu')[i][j] == -1:
                    dmu[i][j] = GnegI(fileinfo.get('rnn-mu')[i][j])
                if G2info.get('rnn-mu')[i][j] == 0:
                    dmu[i][j] = fileinfo.get('rnn-mu')[i][j]
    pushinfo['rnn-mu'] = dmu

    # rnn-output_b
    doutput_b = np.empty(shape=(1,), dtype='float32')
    if num == 1:
        for i in range(1):
            if G2info.get('rnn-output_b')[i] == 1:
                doutput_b[i] = GposI(fileinfo.get('rnn-output_b')[i])
            if G2info.get('rnn-output_b')[i] == -1:
                doutput_b[i] = GposD(fileinfo.get('rnn-output_b')[i])
            if G2info.get('rnn-output_b')[i] == 0:
                doutput_b[i] = fileinfo.get('rnn-output_b')[i]
    if num == -1:
        for i in range(1):
            if G2info.get('rnn-output_b')[i] == 1:
                doutput_b[i] = GnegD(fileinfo.get('rnn-output_b')[i])
            if G2info.get('rnn-output_b')[i] == -1:
                doutput_b[i] = GnegI(fileinfo.get('rnn-output_b')[i])
            if G2info.get('rnn-output_b')[i] == 0:
                doutput_b[i] = fileinfo.get('rnn-output_b')[i]
    pushinfo['rnn-output_b'] = doutput_b

    # rnn-output_w
    doutput_w = np.empty(shape=(1,), dtype='float32')
    if num == 1:
        for i in range(1):
            if G2info.get('rnn-output_w')[i] == 1:
                doutput_w[i] = GposI(fileinfo.get('rnn-output_w')[i])
            if G2info.get('rnn-output_w')[i] == -1:
                doutput_w[i] = GposD(fileinfo.get('rnn-output_w')[i])
            if G2info.get('rnn-output_w')[i] == 0:
                doutput_w[i] = fileinfo.get('rnn-output_w')[i]
    if num == -1:
        for i in range(1):
            if G2info.get('rnn-output_w')[i] == 1:
                doutput_w[i] = GnegD(fileinfo.get('rnn-output_w')[i])
            if G2info.get('rnn-output_w')[i] == -1:
                doutput_w[i] = GnegI(fileinfo.get('rnn-output_w')[i])
            if G2info.get('rnn-output_w')[i] == 0:
                doutput_w[i] = fileinfo.get('rnn-output_w')[i]
    pushinfo['rnn-output_w'] = doutput_w

    # rnn-sensory_erev
    dsensory_erev = np.empty(shape=(32, 21), dtype='float32')
    if num == 1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_erev')[i][j] == 1:
                    dsensory_erev[i][j] = GposI(fileinfo.get('rnn-sensory_erev')[i][j])
                if G2info.get('rnn-sensory_erev')[i][j] == -1:
                    dsensory_erev[i][j] = GposD(fileinfo.get('rnn-sensory_erev')[i][j])
                if G2info.get('rnn-sensory_erev')[i][j] == 0:
                    dsensory_erev[i][j] = fileinfo.get('rnn-sensory_erev')[i][j]
    if num == -1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_erev')[i][j] == 1:
                    dsensory_erev[i][j] = GnegD(fileinfo.get('rnn-sensory_erev')[i][j])
                if G2info.get('rnn-sensory_erev')[i][j] == -1:
                    dsensory_erev[i][j] = GnegI(fileinfo.get('rnn-sensory_erev')[i][j])
                if G2info.get('rnn-sensory_erev')[i][j] == 0:
                    dsensory_erev[i][j] = fileinfo.get('rnn-sensory_erev')[i][j]
    pushinfo['rnn-sensory_erev'] = dsensory_erev

    # rnn-sensory_mu
    dsensory_mu = np.empty(shape=(32, 21), dtype='float32')
    if num == 1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_mu')[i][j] == 1:
                    dsensory_mu[i][j] = GposI(fileinfo.get('rnn-sensory_mu')[i][j])
                if G2info.get('rnn-sensory_mu')[i][j] == -1:
                    dsensory_mu[i][j] = GposD(fileinfo.get('rnn-sensory_mu')[i][j])
                if G2info.get('rnn-sensory_mu')[i][j] == 0:
                    dsensory_mu[i][j] = fileinfo.get('rnn-sensory_mu')[i][j]
    if num == -1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_mu')[i][j] == 1:
                    dsensory_mu[i][j] = GnegD(fileinfo.get('rnn-sensory_mu')[i][j])
                if G2info.get('rnn-sensory_mu')[i][j] == -1:
                    dsensory_mu[i][j] = GnegI(fileinfo.get('rnn-sensory_mu')[i][j])
                if G2info.get('rnn-sensory_mu')[i][j] == 0:
                    dsensory_mu[i][j] = fileinfo.get('rnn-sensory_mu')[i][j]
    pushinfo['rnn-sensory_mu'] = dsensory_mu

    # rnn-sensory_sigma
    dsensory_sigma = np.empty(shape=(32, 21), dtype='float32')
    if num == 1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_sigma')[i][j] == 1:
                    dsensory_sigma[i][j] = GposI(fileinfo.get('rnn-sensory_sigma')[i][j])
                if G2info.get('rnn-sensory_sigma')[i][j] == -1:
                    dsensory_sigma[i][j] = GposD(fileinfo.get('rnn-sensory_sigma')[i][j])
                if G2info.get('rnn-sensory_sigma')[i][j] == 0:
                    dsensory_sigma[i][j] = fileinfo.get('rnn-sensory_sigma')[i][j]
    if num == -1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_sigma')[i][j] == 1:
                    dsensory_sigma[i][j] = GnegD(fileinfo.get('rnn-sensory_sigma')[i][j])
                if G2info.get('rnn-sensory_sigma')[i][j] == -1:
                    dsensory_sigma[i][j] = GnegI(fileinfo.get('rnn-sensory_sigma')[i][j])
                if G2info.get('rnn-sensory_sigma')[i][j] == 0:
                    dsensory_sigma[i][j] = fileinfo.get('rnn-sensory_sigma')[i][j]
    pushinfo['rnn-sensory_sigma'] = dsensory_sigma

    # rnn-sensory_w
    dsensory_w = np.empty(shape=(32, 21), dtype='float32')
    if num == 1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_w')[i][j] == 1:
                    dsensory_w[i][j] = GposI(fileinfo.get('rnn-sensory_w')[i][j])
                if G2info.get('rnn-sensory_w')[i][j] == -1:
                    dsensory_w[i][j] = GposD(fileinfo.get('rnn-sensory_w')[i][j])
                if G2info.get('rnn-sensory_w')[i][j] == 0:
                    dsensory_w[i][j] = fileinfo.get('rnn-sensory_w')[i][j]
    if num == -1:
        for i in range(32):
            for j in range(21):
                if G2info.get('rnn-sensory_w')[i][j] == 1:
                    dsensory_w[i][j] = GnegD(fileinfo.get('rnn-sensory_w')[i][j])
                if G2info.get('rnn-sensory_w')[i][j] == -1:
                    dsensory_w[i][j] = GnegI(fileinfo.get('rnn-sensory_w')[i][j])
                if G2info.get('rnn-sensory_w')[i][j] == 0:
                    dsensory_w[i][j] = fileinfo.get('rnn-sensory_w')[i][j]
    pushinfo['rnn-sensory_w'] = dsensory_w

    # rnn-sigma
    dsigma = np.empty(shape=(21, 21), dtype='float32')
    if num == 1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-sigma')[i][j] == 1:
                    dsigma[i][j] = GposI(fileinfo.get('rnn-sigma')[i][j])
                if G2info.get('rnn-sigma')[i][j] == -1:
                    dsigma[i][j] = GposD(fileinfo.get('rnn-sigma')[i][j])
                if G2info.get('rnn-sigma')[i][j] == 0:
                    dsigma[i][j] = fileinfo.get('rnn-sigma')[i][j]
    if num == -1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-sigma')[i][j] == 1:
                    dsigma[i][j] = GnegD(fileinfo.get('rnn-sigma')[i][j])
                if G2info.get('rnn-sigma')[i][j] == -1:
                    dsigma[i][j] = GnegI(fileinfo.get('rnn-sigma')[i][j])
                if G2info.get('rnn-sigma')[i][j] == 0:
                    dsigma[i][j] = fileinfo.get('rnn-sigma')[i][j]
    pushinfo['rnn-sigma'] = dsigma

    #rnn-vleak
    dvleak = np.empty(shape=(21,), dtype='float32')
    if num == 1:
        for i in range(21):
            if G2info.get('rnn-vleak')[i] == 1:
                dvleak[i] = GposI(fileinfo.get('rnn-vleak')[i])
            if G2info.get('rnn-vleak')[i] == -1:
                dvleak[i] = GposD(fileinfo.get('rnn-vleak')[i])
            if G2info.get('rnn-vleak')[i] == 0:
                dvleak[i] = fileinfo.get('rnn-vleak')[i]
    if num == -1:
        for i in range(21):
            if G2info.get('rnn-vleak')[i] == 1:
                dvleak[i] = GnegD(fileinfo.get('rnn-vleak')[i])
            if G2info.get('rnn-vleak')[i] == -1:
                dvleak[i] = GnegI(fileinfo.get('rnn-vleak')[i])
            if G2info.get('rnn-vleak')[i] == 0:
                dvleak[i] = fileinfo.get('rnn-vleak')[i]
    pushinfo['rnn-vleak'] = dvleak

    # rnn-w:0
    dw = np.empty(shape=(21, 21), dtype='float32')
    if num == 1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-w')[i][j] == 1:
                    dw[i][j] = GposI(fileinfo.get('rnn-w')[i][j])
                if G2info.get('rnn-w')[i][j] == -1:
                    dw[i][j] = GposD(fileinfo.get('rnn-w')[i][j])
                if G2info.get('rnn-w')[i][j] == 0:
                    dw[i][j] = fileinfo.get('rnn-w')[i][j]
    if num == -1:
        for i in range(21):
            for j in range(21):
                if G2info.get('rnn-w')[i][j] == 1:
                    dw[i][j] = GnegD(fileinfo.get('rnn-w')[i][j])
                if G2info.get('rnn-w')[i][j] == -1:
                    dw[i][j] = GnegI(fileinfo.get('rnn-w')[i][j])
                if G2info.get('rnn-w')[i][j] == 0:
                    dw[i][j] = fileinfo.get('rnn-w')[i][j]
    pushinfo['rnn-w'] = dw

    # time_distributed-timedistributed-kernel
    dtimedistributed = np.empty(shape=(5, 1, 18), dtype='float32')
    if num == 1:
        for i in range(5):
            for j in range(1):
                for k in range(18):
                    if G2info.get('time_distributed-timedistributed-kernel')[i][j][k] == 1:
                        dtimedistributed[i][j][k] = GposI(fileinfo.get('time_distributed-timedistributed-kernel')[i][j][k])
                    if G2info.get('time_distributed-timedistributed-kernel')[i][j][k] == -1:
                        dtimedistributed[i][j][k] = GposD(fileinfo.get('time_distributed-timedistributed-kernel')[i][j][k])
                    if G2info.get('time_distributed-timedistributed-kernel')[i][j][k] == 0:
                        dtimedistributed[i][j][k] = fileinfo.get('time_distributed-timedistributed-kernel')[i][j][k]
    if num == -1:
        for i in range(5):
            for j in range(1):
                for k in range(18):
                    if G2info.get('time_distributed-timedistributed-kernel')[i][j][k] == 1:
                        dtimedistributed[i][j][k] = GnegD(fileinfo.get('time_distributed-timedistributed-kernel')[i][j][k])
                    if G2info.get('time_distributed-timedistributed-kernel')[i][j][k] == -1:
                        dtimedistributed[i][j][k] = GnegI(fileinfo.get('time_distributed-timedistributed-kernel')[i][j][k])
                    if G2info.get('time_distributed-timedistributed-kernel')[i][j][k] == 0:
                        dtimedistributed[i][j][k] = fileinfo.get('time_distributed-timedistributed-kernel')[i][j][k]
    pushinfo['time_distributed-timedistributed-kernel'] = dtimedistributed

    # time_distributed - bias
    time_distributed = np.empty(shape=(18,), dtype='float32')
    if num == 1:
        for i in range(18):
            if G2info.get('time_distributed')[i] == 1:
                time_distributed[i] = GposI(fileinfo.get('time_distributed')[i])
            if G2info.get('time_distributed')[i] == -1:
                time_distributed[i] = GposD(fileinfo.get('time_distributed')[i])
            if G2info.get('time_distributed')[i] == 0:
                time_distributed[i] = fileinfo.get('time_distributed')[i]
    if num == -1:
        for i in range(18):
            if G2info.get('time_distributed')[i] == 1:
                time_distributed[i] = GnegD(fileinfo.get('time_distributed')[i])
            if G2info.get('time_distributed')[i] == -1:
                time_distributed[i] = GnegI(fileinfo.get('time_distributed')[i])
            if G2info.get('time_distributed')[i] == 0:
                time_distributed[i] = fileinfo.get('time_distributed')[i]
    pushinfo['time_distributed'] = time_distributed

    # time_distributed_1-timedistributed_1-kernel
    dtimedistributed1 = np.empty(shape=(5, 18, 20), dtype='float32')
    if num == 1:
        for i in range(5):
            for j in range(18):
                for k in range(20):
                    if G2info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] == 1:
                        dtimedistributed1[i][j][k] = GposI(
                            fileinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k])
                    if G2info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] == -1:
                        dtimedistributed1[i][j][k] = GposD(
                            fileinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k])
                    if G2info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] == 0:
                        dtimedistributed1[i][j][k] = fileinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k]
    if num == -1:
        for i in range(5):
            for j in range(18):
                for k in range(20):
                    if G2info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] == 1:
                        dtimedistributed1[i][j][k] = GnegD(
                            fileinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k])
                    if G2info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] == -1:
                        dtimedistributed1[i][j][k] = GnegI(
                            fileinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][k])
                    if G2info.get('time_distributed_1-timedistributed_1-kernel')[i][j][k] == 0:
                        dtimedistributed1[i][j][k] = fileinfo.get('time_distributed_1-timedistributed_1-kernel')[i][j][
                            k]
    pushinfo['time_distributed_1-timedistributed_1-kernel'] = dtimedistributed1

    # time_distributed_1 - bias
    time_distributed_1 = np.empty(shape=(20,), dtype='float32')
    if num == 1:
        for i in range(20):
            if G2info.get('time_distributed_1')[i] == 1:
                time_distributed_1[i] = GposI(fileinfo.get('time_distributed_1')[i])
            if G2info.get('time_distributed_1')[i] == -1:
                time_distributed_1[i] = GposD(fileinfo.get('time_distributed_1')[i])
            if G2info.get('time_distributed_1')[i] == 0:
                time_distributed_1[i] = fileinfo.get('time_distributed_1')[i]
    if num == -1:
        for i in range(20):
            if G2info.get('time_distributed_1')[i] == 1:
                time_distributed_1[i] = GnegD(fileinfo.get('time_distributed_1')[i])
            if G2info.get('time_distributed_1')[i] == -1:
                time_distributed_1[i] = GnegI(fileinfo.get('time_distributed_1')[i])
            if G2info.get('time_distributed_1')[i] == 0:
                time_distributed_1[i] = fileinfo.get('time_distributed_1')[i]
    pushinfo['time_distributed_1'] = time_distributed_1

    # time_distributed_3-timedistributed_3-kernel
    dtimedistributed3 = np.empty(shape=(5,20,22), dtype='float32')
    if num == 1:
        for i in range(5):
            for j in range(20):
                for k in range(22):
                    if G2info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] == 1:
                        dtimedistributed3[i][j][k] = GposI(
                            fileinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][k])
                    if G2info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] == -1:
                        dtimedistributed3[i][j][k] = GposD(
                            fileinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][k])
                    if G2info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] == 0:
                        dtimedistributed3[i][j][k] = fileinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][
                            k]
    if num == -1:
        for i in range(5):
            for j in range(20):
                for k in range(22):
                    if G2info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] == 1:
                        dtimedistributed3[i][j][k] = GnegD(
                            fileinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][k])
                    if G2info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] == -1:
                        dtimedistributed3[i][j][k] = GnegI(
                            fileinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][k])
                    if G2info.get('time_distributed_3-timedistributed_3-kernel')[i][j][k] == 0:
                        dtimedistributed3[i][j][k] = fileinfo.get('time_distributed_3-timedistributed_3-kernel')[i][j][
                            k]
    pushinfo['time_distributed_3-timedistributed_3-kernel']=dtimedistributed3

    # time_distributed_3 - bias
    time_distributed_3 = np.empty(shape=(22,), dtype='float32')
    if num == 1:
        for i in range(22):
            if G2info.get('time_distributed_3')[i] == 1:
                time_distributed_3[i] = GposI(fileinfo.get('time_distributed_3')[i])
            if G2info.get('time_distributed_3')[i] == -1:
                time_distributed_3[i] = GposD(fileinfo.get('time_distributed_3')[i])
            if G2info.get('time_distributed_3')[i] == 0:
                time_distributed_3[i] = fileinfo.get('time_distributed_3')[i]
    if num == -1:
        for i in range(22):
            if G2info.get('time_distributed_3')[i] == 1:
                time_distributed_3[i] = GnegD(fileinfo.get('time_distributed_3')[i])
            if G2info.get('time_distributed_3')[i] == -1:
                time_distributed_3[i] = GnegI(fileinfo.get('time_distributed_3')[i])
            if G2info.get('time_distributed_3')[i] == 0:
                time_distributed_3[i] = fileinfo.get('time_distributed_3')[i]
    pushinfo['time_distributed_3'] = time_distributed_3

    # time_distributed_5-timedistributed_5-kernel
    dtimedistributed5 = np.empty(shape=(5,22,24), dtype='float32')
    if num == 1:
        for i in range(5):
            for j in range(22):
                for k in range(24):
                    if G2info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] == 1:
                        dtimedistributed5[i][j][k] = GposI(
                            fileinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][k])
                    if G2info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] == -1:
                        dtimedistributed5[i][j][k] = GposD(
                            fileinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][k])
                    if G2info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] == 0:
                        dtimedistributed5[i][j][k] = fileinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][
                            k]
    if num == -1:
        for i in range(5):
            for j in range(22):
                for k in range(24):
                    if G2info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] == 1:
                        dtimedistributed5[i][j][k] = GnegD(
                            fileinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][k])
                    if G2info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] == -1:
                        dtimedistributed5[i][j][k] = GnegI(
                            fileinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][k])
                    if G2info.get('time_distributed_5-timedistributed_5-kernel')[i][j][k] == 0:
                        dtimedistributed5[i][j][k] = fileinfo.get('time_distributed_5-timedistributed_5-kernel')[i][j][
                            k]
    pushinfo['time_distributed_5-timedistributed_5-kernel']=dtimedistributed5

    # time_distributed_5 - bias
    time_distributed_5 = np.empty(shape=(24,), dtype='float32')
    if num == 1:
        for i in range(24):
            if G2info.get('time_distributed_5')[i] == 1:
                time_distributed_5[i] = GposI(fileinfo.get('time_distributed_5')[i])
            if G2info.get('time_distributed_5')[i] == -1:
                time_distributed_5[i] = GposD(fileinfo.get('time_distributed_5')[i])
            if G2info.get('time_distributed_5')[i] == 0:
                time_distributed_5[i] = fileinfo.get('time_distributed_5')[i]
    if num == -1:
        for i in range(24):
            if G2info.get('time_distributed_5')[i] == 1:
                time_distributed_5[i] = GnegD(fileinfo.get('time_distributed_5')[i])
            if G2info.get('time_distributed_5')[i] == -1:
                time_distributed_5[i] = GnegI(fileinfo.get('time_distributed_5')[i])
            if G2info.get('time_distributed_5')[i] == 0:
                time_distributed_5[i] = fileinfo.get('time_distributed_5')[i]
    pushinfo['time_distributed_5'] = time_distributed_5

    # time_distributed_7-timedistributed_7-kernel
    dtimedistributed7 = np.empty(shape=(384,32), dtype='float32')
    if num == 1:
        for i in range(384):
            for j in range(32):
                if G2info.get('time_distributed_7-timedistributed_7-kernel')[i][j]== 1:
                    dtimedistributed7[i][j] = GposI(
                        fileinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j])
                if G2info.get('time_distributed_7-timedistributed_7-kernel')[i][j] == -1:
                    dtimedistributed7[i][j] = GposD(
                        fileinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j])
                if G2info.get('time_distributed_7-timedistributed_7-kernel')[i][j] == 0:
                    dtimedistributed7[i][j] = fileinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j]
    if num == -1:
        for i in range(384):
            for j in range(32):
                if G2info.get('time_distributed_7-timedistributed_7-kernel')[i][j] == 1:
                    dtimedistributed7[i][j] = GnegD(
                        fileinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j])
                if G2info.get('time_distributed_7-timedistributed_7-kernel')[i][j] == -1:
                    dtimedistributed7[i][j] = GnegI(
                        fileinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j])
                if G2info.get('time_distributed_7-timedistributed_7-kernel')[i][j] == 0:
                    dtimedistributed7[i][j] = fileinfo.get('time_distributed_7-timedistributed_7-kernel')[i][j]
    pushinfo['time_distributed_7-timedistributed_7-kernel']=dtimedistributed7

    # time_distributed_7 - bias
    time_distributed_7 = np.empty(shape=(32,), dtype='float32')
    if num == 1:
        for i in range(32):
            if G2info.get('time_distributed_7')[i] == 1:
                time_distributed_7[i] = GposI(fileinfo.get('time_distributed_7')[i])
            if G2info.get('time_distributed_7')[i] == -1:
                time_distributed_7[i] = GposD(fileinfo.get('time_distributed_7')[i])
            if G2info.get('time_distributed_7')[i] == 0:
                time_distributed_7[i] = fileinfo.get('time_distributed_7')[i]
    if num == -1:
        for i in range(32):
            if G2info.get('time_distributed_7')[i] == 1:
                time_distributed_7[i] = GnegD(fileinfo.get('time_distributed_7')[i])
            if G2info.get('time_distributed_7')[i] == -1:
                time_distributed_7[i] = GnegI(fileinfo.get('time_distributed_7')[i])
            if G2info.get('time_distributed_7')[i] == 0:
                time_distributed_7[i] = fileinfo.get('time_distributed_7')[i]
    pushinfo['time_distributed_7'] = time_distributed_7



#TODO G+_temp管理
def update_Gplus_temp_Matrix():
    push_file_data_to_infodict('G+.h5','Matrix3.h5')
    get_G_Update_info(num=1)
    write_data_into_file('G+temp.h5',pushinfo)

#TODO G-_temp管理
def update_Gmin_temp_Matrix():
    push_file_data_to_infodict('G-.h5','Matrix3.h5')
    get_G_Update_info(num=-1)
    write_data_into_file('G-temp.h5', pushinfo)


# TODO G+管理
def update_Gplus_Matrix():
    push_file_data_to_infodict('G+.h5', 'Matrix3_temp.h5')
    get_G_Update_info(num=1)
    write_data_into_file('G+.h5', pushinfo)


# TODO G-管理
def update_Gmin_Matrix():
    push_file_data_to_infodict('G-.h5', 'Matrix3_temp.h5')
    get_G_Update_info(num=-1)
    write_data_into_file('G-.h5', pushinfo)


#TODO 遍历文件获取数据
def push_file_data_to_infodict(FileName1,FileName2):
    fileinfo.clear()
    fdata = h5py.File(FileName1, 'r+')
    fileinfo['rnn-cm']=fdata['rnn']['rnn']['ltc_cell']['cm:0'].value
    fileinfo['rnn-erev'] = fdata['rnn']['rnn']['ltc_cell']['erev:0'].value
    fileinfo['rnn-gleak'] = fdata['rnn']['rnn']['ltc_cell']['gleak:0'].value
    fileinfo['rnn-input_b'] = fdata['rnn']['rnn']['ltc_cell']['input_b:0'].value
    fileinfo['rnn-input_w'] = fdata['rnn']['rnn']['ltc_cell']['input_w:0'].value
    fileinfo['rnn-mu'] = fdata['rnn']['rnn']['ltc_cell']['mu:0'].value
    fileinfo['rnn-output_b'] = fdata['rnn']['rnn']['ltc_cell']['output_b:0'].value
    fileinfo['rnn-output_w'] = fdata['rnn']['rnn']['ltc_cell']['output_w:0'].value
    fileinfo['rnn-sensory_erev'] = fdata['rnn']['rnn']['ltc_cell']['sensory_erev:0'].value
    fileinfo['rnn-sensory_mu'] = fdata['rnn']['rnn']['ltc_cell']['sensory_mu:0'].value
    fileinfo['rnn-sensory_sigma'] = fdata['rnn']['rnn']['ltc_cell']['sensory_sigma:0'].value
    fileinfo['rnn-sensory_w']=fdata['rnn']['rnn']['ltc_cell']['sensory_w:0'].value
    fileinfo['rnn-sigma'] = fdata['rnn']['rnn']['ltc_cell']['sigma:0'].value
    fileinfo['rnn-vleak'] = fdata['rnn']['rnn']['ltc_cell']['vleak:0'].value
    fileinfo['rnn-w'] = fdata['rnn']['rnn']['ltc_cell']['w:0'].value
    fileinfo['time_distributed-timedistributed-kernel']=fdata['time_distributed']['time_distributed']['kernel:0'].value
    fileinfo['time_distributed_1-timedistributed_1-kernel'] = fdata['time_distributed_1']['time_distributed_1']['kernel:0'].value
    fileinfo['time_distributed_3-timedistributed_3-kernel'] = fdata['time_distributed_3']['time_distributed_3']['kernel:0'].value
    fileinfo['time_distributed_5-timedistributed_5-kernel'] = fdata['time_distributed_5']['time_distributed_5']['kernel:0'].value
    fileinfo['time_distributed_7-timedistributed_7-kernel'] = fdata['time_distributed_7']['time_distributed_7']['kernel:0'].value
    fileinfo['time_distributed'] = fdata['time_distributed']['time_distributed']['bias:0'].value
    fileinfo['time_distributed_1'] = fdata['time_distributed_1']['time_distributed_1']['bias:0'].value
    fileinfo['time_distributed_3'] = fdata['time_distributed_3']['time_distributed_3']['bias:0'].value
    fileinfo['time_distributed_5'] = fdata['time_distributed_5']['time_distributed_5']['bias:0'].value
    fileinfo['time_distributed_7'] = fdata['time_distributed_7']['time_distributed_7']['bias:0'].value

    if FileName2 !='':
        G2info.clear()
        fdata = h5py.File(FileName2, 'r+')
        G2info['rnn-cm'] = fdata['rnn']['rnn']['ltc_cell']['cm:0'].value
        G2info['rnn-erev'] = fdata['rnn']['rnn']['ltc_cell']['erev:0'].value
        G2info['rnn-gleak'] = fdata['rnn']['rnn']['ltc_cell']['gleak:0'].value
        G2info['rnn-input_b'] = fdata['rnn']['rnn']['ltc_cell']['input_b:0'].value
        G2info['rnn-input_w'] = fdata['rnn']['rnn']['ltc_cell']['input_w:0'].value
        G2info['rnn-mu'] = fdata['rnn']['rnn']['ltc_cell']['mu:0'].value
        G2info['rnn-output_b'] = fdata['rnn']['rnn']['ltc_cell']['output_b:0'].value
        G2info['rnn-output_w'] = fdata['rnn']['rnn']['ltc_cell']['output_w:0'].value
        G2info['rnn-sensory_erev'] = fdata['rnn']['rnn']['ltc_cell']['sensory_erev:0'].value
        G2info['rnn-sensory_mu'] = fdata['rnn']['rnn']['ltc_cell']['sensory_mu:0'].value
        G2info['rnn-sensory_sigma'] = fdata['rnn']['rnn']['ltc_cell']['sensory_sigma:0'].value
        G2info['rnn-sensory_w'] = fdata['rnn']['rnn']['ltc_cell']['sensory_w:0'].value
        G2info['rnn-sigma'] = fdata['rnn']['rnn']['ltc_cell']['sigma:0'].value
        G2info['rnn-vleak'] = fdata['rnn']['rnn']['ltc_cell']['vleak:0'].value
        G2info['rnn-w'] = fdata['rnn']['rnn']['ltc_cell']['w:0'].value
        G2info['time_distributed-timedistributed-kernel'] = fdata['time_distributed']['time_distributed'][
            'kernel:0'].value
        G2info['time_distributed_1-timedistributed_1-kernel'] = fdata['time_distributed_1']['time_distributed_1'][
            'kernel:0'].value
        G2info['time_distributed_3-timedistributed_3-kernel'] = fdata['time_distributed_3']['time_distributed_3'][
            'kernel:0'].value
        G2info['time_distributed_5-timedistributed_5-kernel'] = fdata['time_distributed_5']['time_distributed_5'][
            'kernel:0'].value
        G2info['time_distributed_7-timedistributed_7-kernel'] = fdata['time_distributed_7']['time_distributed_7'][
            'kernel:0'].value
        G2info['time_distributed'] = fdata['time_distributed']['time_distributed']['bias:0'].value
        G2info['time_distributed_1'] = fdata['time_distributed_1']['time_distributed_1']['bias:0'].value
        G2info['time_distributed_3'] = fdata['time_distributed_3']['time_distributed_3']['bias:0'].value
        G2info['time_distributed_5'] = fdata['time_distributed_5']['time_distributed_5']['bias:0'].value
        G2info['time_distributed_7'] = fdata['time_distributed_7']['time_distributed_7']['bias:0'].value

#TODO 遍历字典写入数据
def write_data_into_file(FileName,dic):
    f = h5py.File(FileName, 'r+')
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
    layer=tf.keras.layers.GaussianNoise(0)
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
    getTargetMar('targetm7.h5')
    for i in range(50):
        print('this is the epoch '+str(i))
        generate_Matrix1()
        train_module()
        getMartrix_info(Matrix1,'Matrix1.h5')
        generate_Matrix3()
        update_Gplus_temp_Matrix()
        update_Gmin_temp_Matrix()
        getMartrix_info(Gplus_T_Info, 'G+temp.h5')
        getMartrix_info(Gmius_T_Info, 'G-temp.h5')
        getMartrix_info(pushinfo, 'Matrix3.h5')
        NvalidNum,validNum=isValid(NvalidNum=0,validNum=0)
        print('选择不更新节点数 : '+str(NvalidNum)+"|"+'更新节点数 : '+str(validNum))
        with open('case5.1.txt', 'a+') as File:
            File.write(str(validNum)+"\n")
        write_data_into_file('Matrix3_temp.h5',pushinfo)
        update_Gplus_Matrix()
        update_Gmin_Matrix()
        print('Simulation complete,already generate new G+ and G-')
        model.load_weights('Matrix1.h5')
        pre = model.predict(data, batch_size=30)
        n = 0
        a = 0
        b = 0
        acc = 0
        for x in range(250):
            for y in range(32):
                for z in range(1):
                    predict = pre[x, y, z]
                    target = y_valid[x, y, z]
                    acc = acc + jaccard(math.exp(predict),math.exp(target))
                    n = n + 1
        acc=abs(acc)
        print(str(i)+" : "+str(acc / n))
        with open('case5.txt', 'a+') as File:
            File.write(str(acc / n)+"\n")
