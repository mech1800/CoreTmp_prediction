import torch
import torch.nn as nn
from dataset import MyDataset
from model import LSTMModel
from torch.utils.data import DataLoader, random_split
import numpy as np
import optuna
import matplotlib.pyplot as plt

def torch_mean_relative_error(y_true, y_pred):
    relative_errors = torch.abs((y_true - y_pred) / y_true)
    return torch.mean(relative_errors)

configs = [[2,'all_sequence/single_feature'],
           [2,'all_sequence/normalized_single_feature'],
           [6,'all_sequence/normalized_multi_feature'],
           [2,'10_sequence/single_feature'],
           [2,'10_sequence/normalized_single_feature'],
           [6,'10_sequence/normalized_multi_feature']]

for config in configs:
    input_size = config[0]
    directory = config[1]

    def objective(trial):
        # ハイパーパラメータの選択
        hidden_dim = trial.suggest_int('hidden_dim', 20, 100)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.2, 0.5, 0.8])
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)

        model = LSTMModel(input_size, hidden_dim, 1, num_layers, dropout_rate, device=device).to(device)
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
                outputs = model(inputs)   # LSTMの場合は[N,L,H_in]→[n_sample=2660, sequence_length=10 or 60, input_size=2 or 6]
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

    # データセットの取得
    T_input = np.load('../../data/RNN/'+directory+'/T_input.npy')
    T_core = np.load('../../data/RNN/'+directory+'/T_core.npy')
    n_samples = T_input.shape[0]

    # MyDatasetインスタンスを作成
    dataset = MyDataset(T_input, T_core)

    # データセットを学習、検証、テストに分割
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Optunaでのハイパーパラメータチューニング実行
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params


    # 最適なパラメータでモデルの再トレーニング
    model = LSTMModel(input_size, best_params['hidden_dim'], 1, best_params['num_layers'], best_params['dropout_rate'], device=device).to(device)
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
        torch.save(model.state_dict(), directory+'/best_model_weight.pth')
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
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            train_mse_loss += MSE(outputs, labels).item()
            train_mae_loss += MAE(outputs, labels).item()
            train_mre_loss += torch_mean_relative_error(outputs, labels).item()
        train_mse_loss /= len(train_loader)
        train_mae_loss /= len(train_loader)
        train_mre_loss /= len(train_loader)

        # 検証
        val_mse_loss = 0
        val_mae_loss = 0
        val_mre_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_mse_loss += MSE(outputs, labels).item()
            val_mae_loss += MAE(outputs, labels).item()
            val_mre_loss += torch_mean_relative_error(outputs, labels).item()
        val_mse_loss /= len(val_loader)
        val_mae_loss /= len(val_loader)
        val_mre_loss /= len(val_loader)

        # テスト
        test_mse_loss = 0
        test_mae_loss = 0
        test_mre_loss = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_mse_loss += MSE(outputs, labels).item()
            test_mae_loss += MAE(outputs, labels).item()
            test_mre_loss += torch_mean_relative_error(outputs, labels).item()
        test_mse_loss /= len(test_loader)
        test_mae_loss /= len(test_loader)
        test_mre_loss /= len(test_loader)

    # 損失をテキストファイルに書き出す
    with open(directory+'/best_model_performance.txt', 'w') as file:
        file.write(f'Train MSE: {train_mse_loss}\n')
        file.write(f'Train MAE: {train_mae_loss}\n')
        file.write(f'Train MRE: {train_mre_loss}\n')
        file.write(f'Val MSE: {val_mse_loss}\n')
        file.write(f'Val MAE: {val_mae_loss}\n')
        file.write(f'Val MRE: {val_mre_loss}\n')
        file.write(f'Test MSE: {test_mse_loss}\n')
        file.write(f'Test MAE: {test_mae_loss}\n')
        file.write(f'Test MRE: {test_mre_loss}\n')