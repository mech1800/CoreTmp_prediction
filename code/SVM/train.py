import numpy as np
from sklearn.svm import SVR
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def mean_relative_error(y_true, y_pred):
    relative_errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(relative_errors)

configs = [['60_sequence/normalized_single_feature'],
           ['60_sequence/normalized_multi_feature'],
           ['10_sequence/normalized_single_feature'],
           ['10_sequence/normalized_multi_feature']]

for config in configs:
    directory = config[0]

    def objective(trial):
        # SVMのハイパーパラメータ
        param = {
            'C': trial.suggest_loguniform('C', 1e-2, 1e3),
            'epsilon': trial.suggest_loguniform('epsilon', 1e-4, 1e1),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        }

        model = SVR(**param)
        model.fit(train_T_input, train_T_core)
        pred = model.predict(val_T_input)
        mse = mean_squared_error(val_T_core, pred)

        return mse


    # データセットの取得
    T_input = np.load('../../data/ML/'+directory+'/T_input.npy')
    T_core = np.load('../../data/ML/'+directory+'/T_core.npy')
    n_samples = T_input.shape[0]

    # 特徴量とラベルを[8:1:1]=[学習データ(tr):検証データ(va):テストデータ(te)]の割合で分割
    train_T_input, val_T_input, train_T_core, val_T_core = train_test_split(T_input, T_core, test_size=0.2, random_state=1, shuffle=True)
    val_T_input, test_T_input, val_T_core, test_T_core = train_test_split(val_T_input, val_T_core, test_size=0.5, random_state=1, shuffle=True)

    # Optunaによるハイパーパラメータチューニング
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)


    # 最適なパラメータでモデルの再トレーニング
    best_params = study.best_params
    model = SVR(**best_params)
    model.fit(train_T_input, train_T_core)

    # モデルを保存
    with open(directory + '/best_model.pth', 'wb') as file:
        pickle.dump(model, file)

    # 各データセットに対するMSEとMAEの計算
    train_pred = model.predict(train_T_input)
    val_pred = model.predict(val_T_input)
    test_pred = model.predict(test_T_input)

    train_mse = mean_squared_error(train_T_core, train_pred)
    val_mse = mean_squared_error(val_T_core, val_pred)
    test_mse = mean_squared_error(test_T_core, test_pred)

    train_mae = mean_absolute_error(train_T_core, train_pred)
    val_mae = mean_absolute_error(val_T_core, val_pred)
    test_mae = mean_absolute_error(test_T_core, test_pred)

    train_mre = mean_relative_error(train_T_core, train_pred)
    val_mre = mean_relative_error(val_T_core, val_pred)
    test_mre = mean_relative_error(test_T_core, test_pred)

    # 結果をファイルに書き出す
    with open(directory+'/best_model_performance.txt', 'w') as file:
        file.write(f'Train MSE: {train_mse}\n')
        file.write(f'Train MAE: {train_mae}\n')
        file.write(f'Train MRE: {train_mre}\n')
        file.write(f'Val MSE: {val_mse}\n')
        file.write(f'Val MAE: {val_mae}\n')
        file.write(f'Val MRE: {val_mre}\n')
        file.write(f'Test MSE: {test_mse}\n')
        file.write(f'Test MAE: {test_mae}\n')
        file.write(f'Test MRE: {test_mre}\n')