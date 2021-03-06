import chainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chainer.links as L
import chainer.functions as F

from chainer import Sequential
from chainer import serializers

# モデルの形を学習させた時と同じ形で設定する
class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=1):
        super().__init__()

        with self.init_scope():
            self.fc1 = L.Linear(None, n_mid_units)
            self.fc2 = L.Linear(n_mid_units, n_mid_units)
            self.fc3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h = self.fc3(h2)
        return h



from chainer import optimizers
from chainer import training

# ネットワークを作成
loaded_net = MLP()

# 学習済みモデルの読み込み
chainer.serializers.load_npz('Output_net/mlp_questionnaire_matsuda.net', loaded_net)

df = pd.read_csv('matsuda_questionare_test.csv', header=0)
x = df.loc[:, ["Column_num","Color_white","Color_black", "Icon_A", "Icon_B", "Icon_C", "Layout_thin", "Layout_thick", "Layout_ratio", "Menu_rectangle", "Menu_float", "Menu_convex", "Shape_corner", "Shape_round", "Shape_chamfer", "Header_none", "Header_half", "Header_full", "Char_none", "Char_half", "Char_full"]]
x = np.array(x,np.float32)



with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y = loaded_net(x)
# print(y)
#print(y[0,0])
score_list = []
for i in range(8):
    value = float(y[i,0].data)
    value = round(value,2)
    score_list.append(value)

print(score_list)

# x = np.array([4.07, 6.16, 10.42, 7.08, 5.92, 1.78, 6.16, 7.37]).astype(np.float32)
# y = np.array([9,3,5,4,7,2,6,5]).astype(np.float32)
# loss = F.mean_absolute_error(x,y)
# print(loss)
