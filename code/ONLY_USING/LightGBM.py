import pickle
import numpy as np

# データを用意する
T_skin = np.random.rand(181)   # 0~3600sまで20s間隔のT_skinデータをnumpyの1次元配列として用意してください ex.[27.1,27.9,...,37]
T_sen = np.random.rand(181)   # 0~3600sまで20s間隔のT_senデータをnumpyの1次元配列として用意してください ex.[25.1,26.2,...,37]

########## ここから下は変更する必要はありません ##########

# データの前処理
T_input = np.concatenate((T_skin, T_sen))
T_input = T_input.reshape(1,362)

# 機械学習モデルをロード
with open('../LightGBM/all_sequence/single_feature/best_model.pkl', 'rb') as file:
   model = pickle.load(file)

# 機械学習モデルに入力する
pred_T_core = model.predict(T_input)
print(f'モデルの予測する深部体温は {pred_T_core[0]}℃ です')