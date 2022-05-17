import h5py
import numpy as np
Gmax=303.394/100
Gmin=0.972985/100
pushinfo={}
fileinfo={}
Gp={}
Gn={}
#TODO 遍历文件获取数据
def push_file_data_to_infodict(FileName1):
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
    fileinfo['time_distributed'] = fdata['time_distributed']['time_distributed']['bias:0'].value
    fileinfo['time_distributed_1-timedistributed_1-kernel'] = fdata['time_distributed_1']['time_distributed_1']['kernel:0'].value
    fileinfo['time_distributed_1'] = fdata['time_distributed_1']['time_distributed_1']['bias:0'].value
    fileinfo['time_distributed_3-timedistributed_3-kernel'] = fdata['time_distributed_3']['time_distributed_3']['kernel:0'].value
    fileinfo['time_distributed_3'] = fdata['time_distributed_3']['time_distributed_3']['bias:0'].value
    fileinfo['time_distributed_5-timedistributed_5-kernel'] = fdata['time_distributed_5']['time_distributed_5']['kernel:0'].value
    fileinfo['time_distributed_5'] = fdata['time_distributed_5']['time_distributed_5']['bias:0'].value
    fileinfo['time_distributed_7-timedistributed_7-kernel'] = fdata['time_distributed_7']['time_distributed_7']['kernel:0'].value
    fileinfo['time_distributed_7'] = fdata['time_distributed_7']['time_distributed_7']['bias:0'].value


#TODO G+的值保存在gz中，G-的值保存在gf中
def generate_diff_pushinfo(Matrix_1):
    pushinfo.clear()
    push_file_data_to_infodict(Matrix_1)
    #rnn-cm
    G1cm=fileinfo.get('rnn-cm')
    gz = np.empty(shape=(21,), dtype='float32')
    gf = np.empty(shape=(21,), dtype='float32')
    for i in range(21):
        data=G1cm[i]
        if data>0:
            if Gmin+data >Gmax:
                print('52')
                gz[i]=Gmax
            else:
                gz[i]=Gmin+data
            gf[i]=Gmin
        elif data<0:
            if abs(data)+Gmin>Gmax:
                print('59')
                gf[i]=Gmax
            else:
                gf[i]=abs(data)+Gmin
            gz[i]=Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-cm']=gz
    Gn['rnn-cm']=gf


    #rnn-erev:0
    G1erev=fileinfo.get('rnn-erev')
    gz=np.empty(shape=(21,21), dtype='float32')
    gf = np.empty(shape=(21, 21), dtype='float32')
    for i in range(21):
        for j in range(21):
            data=G1erev[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('80')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('87')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin

    Gp['rnn-erev']=gz
    Gn['rnn-erev']=gf
    #rnn-gleak
    G1gleak = fileinfo.get('rnn-gleak')
    gz = np.empty(shape=(21,), dtype='float32')
    gf = np.empty(shape=(21,), dtype='float32')
    for i in range(21):
        data =G1gleak[i]
        if data > 0:
            if Gmin + data > Gmax:
                print('106')
                gz[i] = Gmax
            else:
                gz[i] = Gmin + data
            gf[i] = Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                print('113')
                gf[i] = Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-gleak']=gz
    Gn['rnn-gleak']=gf

    #rnn-input_b
    G1input_b = fileinfo.get('rnn-input_b')
    gz = np.empty(shape=(32,), dtype='float32')
    gf = np.empty(shape=(32,), dtype='float32')
    for i in range(32):
        data=G1input_b[i]
        if data > 0:
            if Gmin + data > Gmax:
                print('132')
                gz[i] = Gmax
            else:
                gz[i] = Gmin + data
            gf[i] = Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                print('139')
                gf[i] = Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-input_b'] = gz
    Gn['rnn-input_b'] = gf

    # rnn-input_w
    G1input_w = fileinfo.get('rnn-input_w')
    gz = np.empty(shape=(32,), dtype='float32')
    gf = np.empty(shape=(32,), dtype='float32')
    for i in range(32):
        data = G1input_w[i]
        if data > 0:
            if Gmin + data > Gmax:
                print('158')
                gz[i] = Gmax
            else:
                gz[i] = Gmin + data
            gf[i] = Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                print('165')
                gf[i] = Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-input_w'] = gz
    Gn['rnn-input_w'] = gf

    # rnn-mu:0
    G1mu = fileinfo.get('rnn-mu')
    gz = np.empty(shape=(21, 21), dtype='float32')
    gf = np.empty(shape=(21, 21), dtype='float32')
    for i in range(21):
        for j in range(21):
            data = G1mu[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('185')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('192')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-mu'] = gz
    Gn['rnn-mu'] = gf

    # rnn-output_b
    G1output_b = fileinfo.get('rnn-output_b')
    gz = np.empty(shape=(1,), dtype='float32')
    gf = np.empty(shape=(1,), dtype='float32')
    for i in range(1):
        data=G1output_b[i]
        if data>0:
            if Gmin+data >Gmax:
                print('211')
                gz[i]=Gmax
            else:
                gz[i]=Gmin+data
            gf[i]=Gmin
        elif data<0:
            if abs(data)+Gmin>Gmax:
                print('218')
                gf[i]=Gmax
            else:
                gf[i]=abs(data)+Gmin
            gz[i]=Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-output_b'] = gz
    Gn['rnn-output_b'] = gf

    # rnn-output_w
    G1output_w = fileinfo.get('rnn-output_w')
    gz = np.empty(shape=(1,), dtype='float32')
    gf = np.empty(shape=(1,), dtype='float32')
    for i in range(1):
        data = G1output_w[i]
        if data > 0:
            if Gmin + data > Gmax:
                print('237')
                gz[i] = Gmax
            else:
                gz[i] = Gmin + data
            gf[i] = Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                print('244')
                gf[i] = Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-output_w'] = gz
    Gn['rnn-output_w'] = gf

    # rnn-sensory_erev
    G1sensory_erev = fileinfo.get('rnn-sensory_erev')
    gz = np.empty(shape=(32, 21), dtype='float32')
    gf = np.empty(shape=(32, 21), dtype='float32')
    for i in range(32):
        for j in range(21):
            data = G1sensory_erev[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('264')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('271')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-sensory_erev'] = gz
    Gn['rnn-sensory_erev'] = gf

    # rnn-sensory_mu
    G1sensory_mu = fileinfo.get('rnn-sensory_mu')
    gz = np.empty(shape=(32, 21), dtype='float32')
    gf = np.empty(shape=(32, 21), dtype='float32')
    for i in range(32):
        for j in range(21):
            data = G1sensory_mu[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('291')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('298')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-sensory_mu'] = gz
    Gn['rnn-sensory_mu'] = gf

    # rnn-sensory_sigma
    G1sensory_sigma = fileinfo.get('rnn-sensory_sigma')
    gz = np.empty(shape=(32, 21), dtype='float32')
    gf = np.empty(shape=(32, 21), dtype='float32')
    for i in range(32):
        for j in range(21):
            data = G1sensory_sigma[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('318')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('325')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-sensory_sigma'] = gz
    Gn['rnn-sensory_sigma'] = gf

    # rnn-sensory_w
    G1sensory_w = fileinfo.get('rnn-sensory_w')
    gz = np.empty(shape=(32, 21), dtype='float32')
    gf = np.empty(shape=(32, 21), dtype='float32')
    for i in range(32):
        for j in range(21):
            data = G1sensory_w[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('345')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('352')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-sensory_w'] = gz
    Gn['rnn-sensory_w'] = gf

    # rnn-sigma
    G1sigma = fileinfo.get('rnn-sigma')
    gz = np.empty(shape=(21, 21), dtype='float32')
    gf = np.empty(shape=(21, 21), dtype='float32')
    for i in range(21):
        for j in range(21):
            data=G1sigma[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('372')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('379')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-sigma'] = gz
    Gn['rnn-sigma'] = gf

    #rnn-vleak
    G1vleak = fileinfo.get('rnn-vleak')
    gz = np.empty(shape=(21,), dtype='float32')
    gf = np.empty(shape=(21,), dtype='float32')
    for i in range(21):
        data=G1vleak[i]
        if data > 0:
            if Gmin + data > Gmax:
                print('398')
                gz[i] = Gmax
            else:
                gz[i] = Gmin + data
            gf[i] = Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                print('405')
                gf[i] = Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['rnn-vleak'] = gz
    Gn['rnn-vleak'] = gf

    # rnn-w:0
    G1w = fileinfo.get('rnn-w')
    gz = np.empty(shape=(21, 21), dtype='float32')
    gf = np.empty(shape=(21, 21), dtype='float32')
    for i in range(21):
        for j in range(21):
            data = G1w[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    print('425')
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    print('432')
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['rnn-w'] = gz
    Gn['rnn-w'] = gf

    # time_distributed-timedistributed-kernel
    G1timedistributed = fileinfo.get('time_distributed-timedistributed-kernel')
    gz = np.empty(shape=(5,1,18), dtype='float32')
    gf = np.empty(shape=(5,1,18), dtype='float32')
    for i in range(5):
        for j in range(1):
            for k in range(18):
                data = G1timedistributed[i][j][k]
                if data > 0:
                    if Gmin + data > Gmax:
                        print('453')
                        gz[i][j][k] = Gmax
                    else:
                        gz[i][j][k] = Gmin + data
                    gf[i][j][k] = Gmin
                elif data < 0:
                    if abs(data) + Gmin > Gmax:
                        print('460')
                        gf[i][j][k] = Gmax
                    else:
                        gf[i][j][k] = abs(data) + Gmin
                    gz[i][j][k] = Gmin
                else:
                    gz[i][j][k] = Gmin
                    gf[i][j][k] = Gmin
    Gp['time_distributed-timedistributed-kernel'] = gz
    Gn['time_distributed-timedistributed-kernel'] = gf

    # time_distributed - bias
    G1time_distributed=fileinfo.get('time_distributed')
    gz = np.empty(shape=(18,), dtype='float32')
    gf = np.empty(shape=(18,), dtype='float32')
    for i in range(18):
        data = G1time_distributed[i]
        if data > 0:
            if Gmin + data > Gmax:
                gz[i] = Gmax
            else:
                gz[i]= Gmin + data
            gf[i]= Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                gf[i]= Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['time_distributed'] = gz
    Gn['time_distributed'] = gf

    # time_distributed_1-timedistributed_1-kernel
    G1timedistributed1 = fileinfo.get('time_distributed_1-timedistributed_1-kernel')
    gz = np.empty(shape=(5,18,20), dtype='float32')
    gf = np.empty(shape=(5,18,20), dtype='float32')
    for i in range(5):
        for j in range(18):
            for k in range(20):
                data =G1timedistributed1[i][j][k]
                if data > 0:
                    if Gmin + data > Gmax:
                        print('505')
                        gz[i][j][k] = Gmax
                    else:
                        gz[i][j][k] = Gmin + data
                    gf[i][j][k] = Gmin
                elif data < 0:
                    if abs(data) + Gmin > Gmax:
                        print('512')
                        gf[i][j][k] = Gmax
                    else:
                        gf[i][j][k] = abs(data) + Gmin
                    gz[i][j][k] = Gmin
                else:
                    gz[i][j][k] = Gmin
                    gf[i][j][k] = Gmin
    Gp['time_distributed_1-timedistributed_1-kernel'] = gz
    Gn['time_distributed_1-timedistributed_1-kernel'] = gf

    # time_distributed_1 - bias
    G1time_distributed_1=fileinfo.get('time_distributed_1')
    gz = np.empty(shape=(20,), dtype='float32')
    gf = np.empty(shape=(20,), dtype='float32')
    for i in range(20):
        data = G1time_distributed_1[i]
        if data > 0:
            if Gmin + data > Gmax:
                gz[i] = Gmax
            else:
                gz[i]= Gmin + data
            gf[i]= Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                gf[i]= Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['time_distributed_1'] = gz
    Gn['time_distributed_1'] = gf

    # time_distributed_3-timedistributed_3-kernel
    G1timedistributed3 = fileinfo.get('time_distributed_3-timedistributed_3-kernel')
    gz = np.empty(shape=(5,20,22), dtype='float32')
    gf = np.empty(shape=(5,20,22), dtype='float32')
    for i in range(5):
        for j in range(20):
            for k in range(22):
                data =G1timedistributed3[i][j][k]
                if data > 0:
                    if Gmin + data > Gmax:
                        print('557')
                        gz[i][j][k] = Gmax
                    else:
                        gz[i][j][k] = Gmin + data
                    gf[i][j][k] = Gmin
                elif data < 0:
                    if abs(data) + Gmin > Gmax:
                        print('564')
                        gf[i][j][k] = Gmax
                    else:
                        gf[i][j][k] = abs(data) + Gmin
                    gz[i][j][k] = Gmin
                else:
                    gz[i][j][k] = Gmin
                    gf[i][j][k] = Gmin
    Gp['time_distributed_3-timedistributed_3-kernel'] = gz
    Gn['time_distributed_3-timedistributed_3-kernel'] = gf

    # time_distributed_3 - bias
    G1time_distributed_3=fileinfo.get('time_distributed_3')
    gz = np.empty(shape=(22,), dtype='float32')
    gf = np.empty(shape=(22,), dtype='float32')
    for i in range(22):
        data = G1time_distributed_3[i]
        if data > 0:
            if Gmin + data > Gmax:
                gz[i] = Gmax
            else:
                gz[i]= Gmin + data
            gf[i]= Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                gf[i]= Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['time_distributed_3'] = gz
    Gn['time_distributed_3'] = gf

    # time_distributed_5-timedistributed_5-kernel
    G1timedistributed5 = fileinfo.get('time_distributed_5-timedistributed_5-kernel')
    gz = np.empty(shape=(5,22,24), dtype='float32')
    gf = np.empty(shape=(5,22,24), dtype='float32')
    for i in range(5):
        for j in range(22):
            for k in range(24):
                data = G1timedistributed5[i][j][k]
                if data > 0:
                    if Gmin + data > Gmax:
                        print('609')
                        gz[i][j][k] = Gmax
                    else:
                        gz[i][j][k] = Gmin + data
                    gf[i][j][k] = Gmin
                elif data < 0:
                    if abs(data) + Gmin > Gmax:
                        print('616')
                        gf[i][j][k] = Gmax
                    else:
                        gf[i][j][k] = abs(data) + Gmin
                    gz[i][j][k] = Gmin
                else:
                    gz[i][j][k] = Gmin
                    gf[i][j][k] = Gmin
    Gp['time_distributed_5-timedistributed_5-kernel'] = gz
    Gn['time_distributed_5-timedistributed_5-kernel'] = gf

    # time_distributed_5 - bias
    G1time_distributed_5=fileinfo.get('time_distributed_5')
    gz = np.empty(shape=(24,), dtype='float32')
    gf = np.empty(shape=(24,), dtype='float32')
    for i in range(24):
        data = G1time_distributed_5[i]
        if data > 0:
            if Gmin + data > Gmax:
                gz[i] = Gmax
            else:
                gz[i]= Gmin + data
            gf[i]= Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                gf[i]= Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['time_distributed_5'] = gz
    Gn['time_distributed_5'] = gf

    # time_distributed_7-timedistributed_7-kernel
    G1timedistributed7 = fileinfo.get('time_distributed_7-timedistributed_7-kernel')
    gz = np.empty(shape=(384,32), dtype='float32')
    gf = np.empty(shape=(384,32), dtype='float32')
    for i in range(384):
        for j in range(32):
            data=G1timedistributed7[i][j]
            if data > 0:
                if Gmin + data > Gmax:
                    gz[i][j] = Gmax
                else:
                    gz[i][j] = Gmin + data
                gf[i][j] = Gmin
            elif data < 0:
                if abs(data) + Gmin > Gmax:
                    gf[i][j] = Gmax
                else:
                    gf[i][j] = abs(data) + Gmin
                gz[i][j] = Gmin
            else:
                gz[i][j] = Gmin
                gf[i][j] = Gmin
    Gp['time_distributed_7-timedistributed_7-kernel'] = gz
    Gn['time_distributed_7-timedistributed_7-kernel'] = gf

    # time_distributed_7 - bias
    G1time_distributed_7=fileinfo.get('time_distributed_7')
    gz = np.empty(shape=(32,), dtype='float32')
    gf = np.empty(shape=(32,), dtype='float32')
    for i in range(32):
        data = G1time_distributed_7[i]
        if data > 0:
            if Gmin + data > Gmax:
                gz[i] = Gmax
            else:
                gz[i]= Gmin + data
            gf[i]= Gmin
        elif data < 0:
            if abs(data) + Gmin > Gmax:
                gf[i]= Gmax
            else:
                gf[i] = abs(data) + Gmin
            gz[i] = Gmin
        else:
            gz[i] = Gmin
            gf[i] = Gmin
    Gp['time_distributed_7'] = gz
    Gn['time_distributed_7'] = gf


#TODO 遍历字典写入数据
def write_data_into_file(FileName,dic):
    f = h5py.File(FileName, 'r+')
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

if __name__ == '__main__':
    generate_diff_pushinfo('train_1.h5')
    write_data_into_file('G+.h5',Gp)
    write_data_into_file('G-.h5', Gn)