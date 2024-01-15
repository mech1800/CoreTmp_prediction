import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 評価関数を定義する
MSE = nn.MSELoss()
MAE = nn.L1Loss()
def torch_mean_relative_error(y_true, y_pred):
    relative_errors = torch.abs((y_true - y_pred) / y_true)
    return torch.mean(relative_errors)


# 処理対象のディレクトリ
configs = [[2,181,'all_sequence/single_feature'],
           [2,181,'all_sequence/normalized_single_feature'],
           [6,181,'all_sequence/normalized_multi_feature'],
           [2,10,'10_sequence/single_feature'],
           [2,10,'10_sequence/normalized_single_feature'],
           [6,10,'10_sequence/normalized_multi_feature']]

# 各ディレクトリに対して処理を実行
for i, config in enumerate(configs):
    # input_size = config[0]
    # sequence_length = config[1]
    directory = config[2]

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")

    # データをロード
    with open(directory + '/train_T_input.pkl', 'rb') as file:
        train_dataset = pickle.load(file)
    with open(directory + '/val_T_input.pkl', 'rb') as file:
        val_dataset = pickle.load(file)
    with open(directory + '/test_T_input.pkl', 'rb') as file:
        test_dataset = pickle.load(file)

    # モデルのパラメータをロード
    with open(directory + '/best_params.pkl', 'rb') as file:
        best_params = pickle.load(file)

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # 機械学習モデルをロード
    model = torch.load(directory + '/best_model.pth')
    model.eval()


    # 各データセットに対するMSE、MAE、MRE、STDの計算
    with torch.no_grad():
        # 学習
        train_mse_loss = 0
        train_mae_loss = 0
        train_mre_loss = 0
        all_train_outputs = []
        all_train_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            train_mse_loss += MSE(outputs, labels).item()
            train_mae_loss += MAE(outputs, labels).item()
            train_mre_loss += torch_mean_relative_error(outputs, labels).item()
            all_train_outputs.append(outputs.detach())
            all_train_labels.append(labels.detach())
        train_mse_loss /= len(train_loader)
        train_mae_loss /= len(train_loader)
        train_mre_loss /= len(train_loader)
        all_train_outputs = torch.cat(all_train_outputs)
        train_std = torch.std(all_train_outputs).item()
        train_mae_2std = train_mae_loss + 2*train_std

        all_train_outputs = all_train_outputs.view(-1).cpu().numpy()
        all_train_labels = torch.cat(all_train_labels)
        all_train_labels = all_train_labels.view(-1).cpu().numpy()

        # 検証
        val_mse_loss = 0
        val_mae_loss = 0
        val_mre_loss = 0
        all_val_outputs = []
        all_val_labels = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_mse_loss += MSE(outputs, labels).item()
            val_mae_loss += MAE(outputs, labels).item()
            val_mre_loss += torch_mean_relative_error(outputs, labels).item()
            all_val_outputs.append(outputs.detach())
            all_val_labels.append(labels.detach())
        val_mse_loss /= len(val_loader)
        val_mae_loss /= len(val_loader)
        val_mre_loss /= len(val_loader)
        all_val_outputs = torch.cat(all_val_outputs)
        val_std = torch.std(all_val_outputs).item()
        val_mae_2std = val_mae_loss + 2*val_std

        all_val_outputs = all_val_outputs.view(-1).cpu().numpy()
        all_val_labels = torch.cat(all_val_labels)
        all_val_labels = all_val_labels.view(-1).cpu().numpy()

        # テスト
        test_mse_loss = 0
        test_mae_loss = 0
        test_mre_loss = 0
        all_test_outputs = []
        all_test_labels = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_mse_loss += MSE(outputs, labels).item()
            test_mae_loss += MAE(outputs, labels).item()
            test_mre_loss += torch_mean_relative_error(outputs, labels).item()
            all_test_outputs.append(outputs.detach())
            all_test_labels.append(labels.detach())
        test_mse_loss /= len(test_loader)
        test_mae_loss /= len(test_loader)
        test_mre_loss /= len(test_loader)
        all_test_outputs = torch.cat(all_test_outputs)
        test_std = torch.std(all_test_outputs).item()
        test_mae_2std = test_mae_loss + 2*test_std

        all_test_outputs = all_test_outputs.view(-1).cpu().numpy()
        all_test_labels = torch.cat(all_test_labels)
        all_test_labels = all_test_labels.view(-1).cpu().numpy()

    # 各データセットに対するMSE、MAE、MRE、STDの集計
    performance_metrics = {
        'Train MSE': train_mse_loss,
        'Train MAE': train_mae_loss,
        'Train MRE': train_mre_loss,
        'Train STD': train_std,
        'Train MAE+2STD': train_mae_2std,
        'Val MSE': train_mse_loss,
        'Val MAE': train_mae_loss,
        'Val MRE': train_mre_loss,
        'Val STD': train_std,
        'Val MAE+2STD': train_mae_2std,
        'Test MSE': train_mse_loss,
        'Test MAE': train_mae_loss,
        'Test MRE': train_mre_loss,
        'Test STD': train_std,
        'Test MAE+2STD': train_mae_2std,
    }

    # 結果をJSONファイルに書き出す
    with open(directory + '/best_model_performance.json', 'w') as file:
        json.dump(performance_metrics, file, indent=4)


    # Bland Altman Plotの作成
    means = np.mean([all_test_outputs, all_test_labels], axis=0)   # all_test_outputsとall_test_labelsは1次元のnumpy配列を想定
    differences = all_test_outputs - all_test_labels
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
    plt.title('Bland-Altman-Plot LightGBM×feature'+str(i), fontsize=16)
    plt.grid(True)

    plt.savefig(directory + '/Bland_Altman_Plot.png')