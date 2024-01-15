import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def mean_relative_error(y_true, y_pred):
    relative_errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(relative_errors)

configs = [['all_sequence/single_feature'],
           ['all_sequence/multi_feature'],
           ['all_sequence/normalized_single_feature'],
           ['all_sequence/normalized_multi_feature'],
           ['10_sequence/single_feature'],
           ['10_sequence/multi_feature'],
           ['10_sequence/normalized_single_feature'],
           ['10_sequence/normalized_multi_feature']]

for config in configs:
    directory = config[0]

    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        }

        model = lgb.LGBMRegressor(**param)
        model.fit(
            train_T_input,
            train_T_core,
            eval_set=[(val_T_input, val_T_core)],
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )
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

    # 学習データ、検証データ、テストデータを保存しておく
    np.save(directory + '/train_T_input.npy', train_T_input)
    np.save(directory + '/train_T_core.npy', train_T_core)
    np.save(directory + '/val_T_input.npy', val_T_input)
    np.save(directory + '/val_T_core.npy', val_T_core)
    np.save(directory + '/test_T_input.npy', test_T_input)
    np.save(directory + '/test_T_core.npy', test_T_core)

    # Optunaによるハイパーパラメータチューニング
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)


    # 最適なパラメータでモデルの再トレーニング
    best_params = study.best_params
    model = lgb.LGBMRegressor(**best_params)
    model.fit(train_T_input, train_T_core)

    # モデルを保存
    with open(directory + '/best_model.pkl', 'wb') as file:
        pickle.dump(model, file)