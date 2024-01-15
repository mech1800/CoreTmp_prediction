import torch
import torch.nn as nn
from dataset import MyDataset
from model import DynamicTCN
from torch.utils.data import DataLoader, random_split
import numpy as np
import optuna
import matplotlib.pyplot as plt
import pickle

def torch_mean_relative_error(y_true, y_pred):
    relative_errors = torch.abs((y_true - y_pred) / y_true)
    return torch.mean(relative_errors)

configs = [[2,181,'all_sequence/single_feature'],
           [2,181,'all_sequence/normalized_single_feature'],
           [6,180,'all_sequence/normalized_multi_feature'],
           [2,10,'10_sequence/single_feature'],
           [2,10,'10_sequence/normalized_single_feature'],
           [6,10,'10_sequence/normalized_multi_feature']]

for config in configs:
    channel = config[0]
    sequence_length = config[1]
    directory = config[2]

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 20, 100)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)

        model = DynamicTCN(channel, num_layers, hidden_dim, kernel_size, sequence_length, dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # データローダーの設定
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(50):  # 試行回数は少なめに設定
            # 学習
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)   # Conv1dの場合は[N,C_in,L_in]→[n_sample=2660, channel=2 or 6, sequence_length=10 or 60]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 検証
            model.eval()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

        return loss


    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")

    # データセットの取得
    T_input = np.load('../../data/CNN/'+directory+'/T_input.npy')
    T_core = np.load('../../data/CNN/'+directory+'/T_core.npy')
    n_samples = T_input.shape[0]

    # MyDatasetインスタンスを作成
    dataset = MyDataset(T_input, T_core)

    # データセットを学習、検証、テストに分割
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 学習データ、検証データ、テストデータを保存しておく
    with open(directory + '/train_T_input.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(directory + '/val_T_input.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(directory + '/test_T_input.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    # Optunaでのハイパーパラメータチューニング実行
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    with open(directory + '/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)


    # 最適なパラメータでモデルの再トレーニング
    model = DynamicTCN(channel, best_params['num_layers'], best_params['hidden_dim'], best_params['kernel_size'], sequence_length, best_params['dropout_rate']).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    tr_loss = []
    va_loss = []
    for epoch in range(50):  # 試行回数は少なめに設定
        # 学習
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()

        tr_loss.append(train_loss/len(train_loader))

        # 検証
        model.eval()
        val_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        va_loss.append(val_loss/len(val_loader))

    else:
        torch.save(model, directory + '/best_model.pth')


    # lossの推移をグラフにする
    x = [i for i in range(50)]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, tr_loss, label='tr_loss')
    ax.plot(x, va_loss, label='va_loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE_loss')
    ax.legend(loc='upper right')
    fig.savefig(directory+'/best_model_loss.png')
    plt.show()


    # テストデータにも当てはめる
    model = torch.load(directory+'/best_model.pth')
    model.eval()
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()

    # 最適なモデルの評価指標を確認
    with torch.no_grad():
        # 学習
        train_mse_loss = 0
        train_mae_loss = 0
        train_mre_loss = 0
        all_outputs = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            train_mse_loss += MSE(outputs, labels).item()
            train_mae_loss += MAE(outputs, labels).item()
            train_mre_loss += torch_mean_relative_error(outputs, labels).item()
            all_outputs.append(outputs.detach())
        train_mse_loss /= len(train_loader)
        train_mae_loss /= len(train_loader)
        train_mre_loss /= len(train_loader)
        all_outputs = torch.cat(all_outputs)
        train_std = torch.std(all_outputs)

        # 検証
        val_mse_loss = 0
        val_mae_loss = 0
        val_mre_loss = 0
        all_outputs = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_mse_loss += MSE(outputs, labels).item()
            val_mae_loss += MAE(outputs, labels).item()
            val_mre_loss += torch_mean_relative_error(outputs, labels).item()
            all_outputs.append(outputs.detach())
        val_mse_loss /= len(val_loader)
        val_mae_loss /= len(val_loader)
        val_mre_loss /= len(val_loader)
        all_outputs = torch.cat(all_outputs)
        val_std = torch.std(all_outputs)

        # テスト
        test_mse_loss = 0
        test_mae_loss = 0
        test_mre_loss = 0
        all_outputs = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_mse_loss += MSE(outputs, labels).item()
            test_mae_loss += MAE(outputs, labels).item()
            test_mre_loss += torch_mean_relative_error(outputs, labels).item()
            all_outputs.append(outputs.detach())
        test_mse_loss /= len(test_loader)
        test_mae_loss /= len(test_loader)
        test_mre_loss /= len(test_loader)
        all_outputs = torch.cat(all_outputs)
        test_std = torch.std(all_outputs)

    # 損失をテキストファイルに書き出す
    with open(directory+'/best_model_performance.txt', 'w') as file:
        file.write(f'Train MSE: {train_mse_loss}\n')
        file.write(f'Train MAE: {train_mae_loss}\n')
        file.write(f'Train MRE: {train_mre_loss}\n')
        file.write(f'Train STD: {train_std}\n')
        file.write(f'Val MSE: {val_mse_loss}\n')
        file.write(f'Val MAE: {val_mae_loss}\n')
        file.write(f'Val MRE: {val_mre_loss}\n')
        file.write(f'Val STD: {val_std}\n')
        file.write(f'Test MSE: {test_mse_loss}\n')
        file.write(f'Test MAE: {test_mae_loss}\n')
        file.write(f'Test MRE: {test_mre_loss}\n')
        file.write(f'Test STD: {test_std}\n')