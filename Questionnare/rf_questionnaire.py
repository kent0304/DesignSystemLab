### 設計変数とニーズ満足度(アンケート)の関係を学習しモデルを作成する
### ランダムフォレスト回帰（RF）

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import chainer
import chainer.functions as F

## 学習データCSVファイル読み込み
input_data = pd.read_csv('Input_data/matsuda_questionare.csv', header=0)
#header=0は省略可。見出しがあるので行番号を0スタート
xtrain = input_data.loc[:, ["Column_num","Color_white","Color_black", "Icon_A", "Icon_B", "Icon_C", "Layout_thin", "Layout_thick", "Layout_ratio", "Menu_rectangle", "Menu_float", "Menu_convex", "Shape_corner", "Shape_round", "Shape_chamfer", "Header_none", "Header_half", "Header_full", "Char_none", "Char_half", "Char_full"]] #df.loc[:, ['col_1','col_2']]で烈ラベル指定
ttrain = input_data.loc[:,"Total"]
x_train = np.array(xtrain,np.float32)
t_train = np.array(ttrain,np.float32)

## テスト用データCSV読み込み
test_data = pd.read_csv('Input_data/matsuda_questionare_test.csv', header=0)
#header=0は省略可。見出しがあるので行番号を0スタート
xtest = test_data.loc[:, ["Column_num","Color_white","Color_black", "Icon_A", "Icon_B", "Icon_C", "Layout_thin", "Layout_thick", "Layout_ratio", "Menu_rectangle", "Menu_float", "Menu_convex", "Shape_corner", "Shape_round", "Shape_chamfer", "Header_none", "Header_half", "Header_full", "Char_none", "Char_half", "Char_full"]] #df.loc[:, ['col_1','col_2']]で烈ラベル指定
ttest = test_data.loc[:,"Total"]
x_test = np.array(xtest,np.float32)
t_test = np.array(ttest,np.float64)


## ランダムフォレスト回帰オブジェクト生成
rfr = RandomForestRegressor(n_estimators=120)
## 学習の実行
rfr.fit(x_train, t_train)
## テストデータで予測実行
predict_t = rfr.predict(x_test)
## R2決定係数で評価
r2_score = r2_score(t_test, predict_t)
print("r2 score:")
print(r2_score)
loss = F.mean_absolute_error(t_test, predict_t)
print("loss:")
print(loss)

## テスト用データの実測値と予測値出力
print("t_test")
print(t_test)
print("predict_t")
print(predict_t)


## 特徴量の重要度を取得
feature = rfr.feature_importances_
## 特徴量の名前ラベルを取得
label = xtrain.columns[0:]
print(label)
print(feature)
# 特徴量の重要度順（降順）に並べて表示
indices = np.argsort(feature)[::-1]
for i in range(len(feature)):
    print(str(i + 1) + "   " +
          str(label[indices[i]]) + "   " + str(feature[indices[i]]))


## 新規設計解導出
## 全データCSV読み込み
all_data = pd.read_csv('Input_data/ui_input_all.csv', header=0) #header=0は省略可。見出しがあるので行番号を0スタートとする。
xall = all_data.loc[:, ["Column_num","Color_white","Color_black", "Icon_A", "Icon_B", "Icon_C", "Layout_thin", "Layout_thick", "Layout_ratio", "Menu_rectangle", "Menu_float", "Menu_convex", "Shape_corner", "Shape_round", "Shape_chamfer", "Header_none", "Header_half", "Header_full", "Char_none", "Char_half", "Char_full"]] #df.loc[:, ['col_1','col_2']]で烈ラベル指定
x_all = np.array(xall,np.float32)
## 全データで予測実行
predict_all = rfr.predict(x_all)
predict_all = list(predict_all)
new_predict_all = []
for ele in predict_all:
    ele = [ele]
    new_predict_all.append(ele)
## 結果の行列をCSVファイルに書き出し
import csv
with open("Output_data/rf_questionnare_allresult.csv", "w") as f:
    # header を設定
    fieldnames = ["Total"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # データの書き込み
    writer = csv.writer(f)
    writer.writerows(new_predict_all)
