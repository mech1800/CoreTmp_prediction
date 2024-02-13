import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import matplotlib.pyplot as plt

# mean_relative_errorを定義します。これはscikit-learnには含まれていないため、カスタム関数が必要です。
def mean_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 処理対象のディレクトリ
configs = [['all_sequence/single_feature'],
           ['all_sequence/multi_feature'],
           ['all_sequence/normalized_single_feature'],
           ['all_sequence/normalized_multi_feature'],
           ['10_sequence/single_feature'],
           ['10_sequence/multi_feature'],
           ['10_sequence/normalized_single_feature'],
           ['10_sequence/normalized_multi_feature']]

# 各ディレクトリに対して処理を実行
for i, config in enumerate(configs):
    directory = config[0]

    # データをロード
    train_T_input = np.load(directory + '/train_T_input.npy')
    train_T_core = np.load(directory + '/train_T_core.npy')
    val_T_input = np.load(directory + '/val_T_input.npy')
    val_T_core = np.load(directory + '/val_T_core.npy')
    test_T_input = np.load(directory + '/test_T_input.npy')
    test_T_core = np.load(directory + '/test_T_core.npy')

    # 機械学習モデルをロード
    with open(directory + '/best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # 各データセットに対する学習済みモデルの出力
    train_pred = model.predict(train_T_input)
    val_pred = model.predict(val_T_input)
    test_pred = model.predict(test_T_input)

    # データを目視確認ができるようにcsvファイルに出力
    reshaped_pred = train_pred.reshape(-1,1)
    reshaped_core = train_T_core.reshape(-1,1)
    train_data = np.concatenate([train_T_input, reshaped_pred, reshaped_core], axis=1)
    reshaped_pred = val_pred.reshape(-1,1)
    reshaped_core = val_T_core.reshape(-1,1)
    val_data = np.concatenate([val_T_input, reshaped_pred, reshaped_core], axis=1)
    reshaped_pred = test_pred.reshape(-1,1)
    reshaped_core = test_T_core.reshape(-1,1)
    test_data = np.concatenate([test_T_input, reshaped_pred, reshaped_core], axis=1)
    data = [train_data, val_data, test_data]
    names = ['/train_data.csv', '/val_data.csv', '/test_data.csv']
    for j in range(3):
        columns = ['input'] + ['' for _ in range(data[j].shape[1]-3)] + ['output', 'label']
        df = pd.DataFrame(data[j], columns=columns)
        csv_file_path = directory + names[j]
        df.to_csv(csv_file_path, index=True, index_label='No')

    # 各データセットに対するMSE、MAE、MRE、STDの計算
    performance_metrics = {
        'Train MSE': mean_squared_error(train_T_core, train_pred),
        'Train MAE': mean_absolute_error(train_T_core, train_pred),
        'Train MRE': mean_relative_error(train_T_core, train_pred),
        'Train STD': np.std(train_T_core - train_pred),
        'Train MAE+2STD': mean_absolute_error(train_T_core, train_pred) + 2*np.std(train_T_core - train_pred),
        'Val MSE': mean_squared_error(val_T_core, val_pred),
        'Val MAE': mean_absolute_error(val_T_core, val_pred),
        'Val MRE': mean_relative_error(val_T_core, val_pred),
        'Val STD': np.std(val_T_core - val_pred),
        'Val MAE+2STD': mean_absolute_error(val_T_core, val_pred) + 2*np.std(val_T_core - val_pred),
        'Test MSE': mean_squared_error(test_T_core, test_pred),
        'Test MAE': mean_absolute_error(test_T_core, test_pred),
        'Test MRE': mean_relative_error(test_T_core, test_pred),
        'Test STD': np.std(test_T_core - test_pred),
        'Test MAE+2STD': mean_absolute_error(test_T_core, test_pred) + 2*np.std(test_T_core - test_pred)
    }

    # 結果をJSONファイルに書き出す
    with open(directory + '/best_model_performance.json', 'w') as file:
        json.dump(performance_metrics, file, indent=4)

    # Bland Altman Plotの作成
    means = np.mean([test_pred, test_T_core], axis=0)   # test_T_coreとtest_predは1次元のnumpy配列を想定
    differences = test_pred - test_T_core
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    plt.figure(figsize=(10, 6))
    plt.scatter(means, differences, alpha=0.5)
    plt.axhline(mean_diff, color='blue', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--')
    plt.text(np.max(means), mean_diff, 'Mean', verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=12)
    plt.text(np.max(means), mean_diff + 1.96 * std_diff, '+1.96 SD', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=12)
    plt.text(np.max(means), mean_diff - 1.96 * std_diff, '-1.96 SD', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=12)
    plt.xlabel("Mean of Model's prediction and True label", fontsize=14)
    plt.ylabel("Difference between Model's prediction and True label", fontsize=14)
    plt.title('Bland-Altman-Plot LightGBM-'+str(i+1), fontsize=16)
    plt.grid(True)

    plt.savefig(directory + '/Bland_Altman_Plot.png')