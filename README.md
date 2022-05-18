# HDF-based-Simulation
这是一个根据更改HDF文件为基础对神经形态器件进行仿真的方法
1.需要随机生成初始模型 可在pathway.py中，随机初始化模型后，使用module.save_weight('???.h5')
2.使用1步中的模型生成G+和G-，使用generate_G+andG-.py,更改里面的.h5的文件接口即可
3.如果需要裁剪网络可更改new_training.py 的deteoverange()函数
4.仿真训练结束后，权重矩阵会保存在Matriix1.h5中，可运行pathway.py获得路径预测结果
5.HDFExplorer.zip是打开.5h文件的工具
case5.txt储存ACCURACY
case5.1.txt储存每次更新的节点数目
