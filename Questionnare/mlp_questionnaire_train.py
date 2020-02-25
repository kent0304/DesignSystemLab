### 設計変数とニーズ満足度(アンケート)の関係を学習しモデルを作成する
### 回帰ニューラルネットワーク(MLP)

import chainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_data = pd.read_csv('matsuda_questionare.csv', header=0) #header=0は省略可。見出しがあるので行番号を0スタートとする。
x = input_data.loc[:, ["Column_num","Color_white","Color_black", "Icon_A", "Icon_B", "Icon_C", "Layout_thin", "Layout_thick", "Layout_ratio", "Menu_rectangle", "Menu_float", "Menu_convex", "Shape_corner", "Shape_round", "Shape_chamfer", "Header_none", "Header_half", "Header_full", "Char_none", "Char_half", "Char_full"]] #df.loc[:, ['col_1','col_2']]で烈ラベル指定
t = input_data.loc[:,["Total"]]

#Chainerがデフォルトで用いるfloat32型に変換
x = np.array(x,np.float32)
t = np.array(t,np.float32) # 分類型のときはint, 回帰型のときはfloat

from chainer.datasets import TupleDataset
dataset = TupleDataset(x,t) #TupleDatasetはchainer.datasetsというモジュール内のクラス。

from chainer.datasets import split_dataset_random
train, valid = split_dataset_random(dataset, int(len(dataset) * 0.8), seed=0) # 抽出するデータは固定

from chainer import iterators
batchsize = 128
train_iter = iterators.SerialIterator(train, batchsize, shuffle=True, repeat=True)
valid_iter = iterators.SerialIterator(valid, batchsize, shuffle=True, repeat=True)
# 何周も何周もデータを繰り返し読み出す必要がある場合はrepeat引数をTrue
# 1周が終わったらそれ以上データを取り出したくない場合はこれをFalse
# デフォルトではTrueなので本当は書かなくてもいい


import chainer.links as L
import chainer.functions as F

# ネットワークの定義
class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=1):
        super().__init__()

        with self.init_scope():
            self.fc1 = L.Linear(None, n_mid_units)
            self.fc2 = L.Linear(n_mid_units, n_mid_units)
            self.fc3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

net = MLP() #インスタンス化

from chainer import optimizers
from chainer.optimizer_hooks import WeightDecay

# 最適化手法の選択
optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)  # 学習率を 0.01 に設定
optimizer.setup(net)

for param in net.params():
    if param.name != 'b':  # バイアス以外だったら
        param.update_rule.add_hook(WeightDecay(0.0001))  # 重み減衰を適用

# エポック数
n_epoch = 401

# ログ
results_train, results_valid = {}, {}
results_train['loss'], results_train['accuracy'] = [], []
results_valid['loss'], results_valid['accuracy'] = [], []

count = 1

train_batch = train_iter.next()
x_train, t_train = chainer.dataset.concat_examples(train_batch)

for epoch in range(n_epoch):

    while True:
        # ミニバッチの取得
        train_batch = train_iter.next()

        # x と t に分割
        x_train, t_train = chainer.dataset.concat_examples(train_batch)

        # 予測値と目的関数の計算
        y_train = net(x_train)
        loss_train = F.mean_absolute_error(y_train, t_train)

        # 勾配の初期化と勾配の計算
        net.cleargrads()
        loss_train.backward()

        # パラメータの更新
        optimizer.update()

        # カウントアップ
        count += 1

        # 1エポック終えたら、valid データで評価する
        if train_iter.is_new_epoch:

            # 検証用データに対する結果の確認
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x_valid, t_valid = chainer.dataset.concat_examples(valid)
                y_valid = net(x_valid)
                loss_valid = F.mean_absolute_error(y_valid, t_valid)


            # 結果の表示
            print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(epoch, count, loss_train.array.mean(), loss_valid.array.mean()))

            # 可視化用に保存
            results_train['loss'].append(loss_train.array)
            results_valid['loss'].append(loss_valid.array)

            break


#目的関数(損失)の可視化
# 損失 (loss)
plt.plot(results_train['loss'], label='train')  # label で凡例の設定
plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
plt.title("Loss Function(Mean Absolute Error)")
plt.xlabel("Epocks")
plt.ylabel("Loss")
plt.legend()  # 凡例の表示

plt.show()

# ネットワークの保存
chainer.serializers.save_npz('mlp_questionnaire_matsuda.net', net)
