### 設計変数とニーズ満足度(アンケート)の関係を学習したモデルを用いて予測を行う
### 全ての設計解の出力値の算出
### 回帰ニューラルネットワーク(MLP)

import chainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
from chainer import serializers
# ------------------------------------------------------------------------------
# ネットワークの定義
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

# Output_netというフォルダに保存された学習済みモデルの読み込み
chainer.serializers.load_npz('Output_net/mlp_questionnaire_matsuda.net', loaded_net)

df = pd.read_csv('Input_data/ui_input_all.csv', header=0)
x = df.loc[:, ["Column_num","Color_white","Color_black", "Icon_A", "Icon_B", "Icon_C", "Layout_thin", "Layout_thick", "Layout_ratio", "Menu_rectangle", "Menu_float", "Menu_convex", "Shape_corner", "Shape_round", "Shape_chamfer", "Header_none", "Header_half", "Header_full", "Char_none", "Char_half", "Char_full"]]
x = np.array(x,np.float32)

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y = loaded_net(x)

import csv

count = 0
total_score = []
like = []

for i in range(4374): # 4374全て通り
    value = float(y[i,0].data)
    value = round(value,2)
    like.append(value)
    value = [value]
    total_score.append(value)

# print(max(total_score)) # 最大値出力
# print(min(total_score)) # 最小値出力


# 結果の行列をCSVファイルに書き出し
with open("Output_data/mlp_questionnare_allresult.csv", "w") as f:
    # header を設定
    fieldnames = ["Total"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # データの書き込み
    writer = csv.writer(f)
    writer.writerows(total_score)
