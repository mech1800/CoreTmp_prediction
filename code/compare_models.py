import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ディレクトリ名
directory1_paths = ['LightGBM/',
                    'SVM/',
                    'LSTM/',
                    '1DCNN/',
                    'TCN/']

directory2_paths = ['all_sequence/single_feature/',
                    'all_sequence/multi_feature/',
                    'all_sequence/normalized_single_feature/',
                    'all_sequence/normalized_multi_feature/',
                    '10_sequence/single_feature/',
                    '10_sequence/multi_feature/',
                    '10_sequence/normalized_single_feature/',
                    '10_sequence/normalized_multi_feature/']

# モデルの数と仮の出力データ
model_mae = []
model_std = []
models = []
count = 0

# 各モデル×特徴量のmaeとstdを取得する
for directory1_path in directory1_paths:
    for directory2_path in directory2_paths:

        # pathが存在しない場合はスキップ
        directory_path = directory1_path + directory2_path
        if not os.path.isdir(directory_path):
            continue

        # jsonファイルを読み込む
        file_path = directory_path + 'best_model_performance.json'
        with open(file_path, 'r') as file:
            data = json.load(file)

        # maeとstdをリストに追加する
        model_mae.append(data["Test MAE"])
        model_std.append(data["Test STD"])

        # model名をリストに追加
        count += 1
        models.append('model'+str(count))

# 棒グラフを作成
plt.figure(figsize=(30, 15))
bars = plt.bar(models, model_mae, color='skyblue', capsize=5)

# 各モデルに対してH型バーで±2SDだけを表す
for i in range(count):
    # ±2SDのH型バー（点線）
    plt.hlines(y=model_mae[i] - 2 * model_std[i], xmin=i - 0.1, xmax=i + 0.1, colors='black', lw=1.5)
    plt.hlines(y=model_mae[i] + 2 * model_std[i], xmin=i - 0.1, xmax=i + 0.1, colors='black', lw=1.5)
    plt.vlines(x=i, ymin=model_mae[i] - 2 * model_std[i], ymax=model_mae[i] + 2 * model_std[i], colors='black', lw=1.5)

# タイトルと軸ラベルの設定
plt.title('Model MAE with 2SD', fontsize=18)
plt.ylabel('MAE', fontsize=14)

# y軸の範囲を設定
plt.ylim(-1.5, 2.5)

# x軸を点線で表示
plt.axhline(0, linestyle='--', color='gray')  # x軸を点線で描画

# 余白を削除
plt.tight_layout()

# グラフの保存
plt.savefig("MAE_with_2SD.png", dpi=300)