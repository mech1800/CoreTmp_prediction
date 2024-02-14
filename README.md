# 概要
このリポジトリはCOMSOLの解析データをもとに機械学習用データセットの作成と複数の機械学習モデルによる学習および評価を行うものです。

# ディレクトリ構造
リポジトリは以下の主要なディレクトリから構成されます。
-'code/':複数の機械学習による学習を行うためディレクトリ。
　-'\*':対象の機械学習に対する学習、評価など全ての処理が含まれます。
-'data/':COMSOLの解析データをもとに機械学習用のデータセットの作成を行うためのディレクトリ。
　-'ML':古典的な機械学習モデルの入出力に合わせたデータセット形式を作成します。
　-'CNN':CNN系ディープラーニングモデルの入出力に合わせたデータセット形式を作成します。
　-'RNN':RNN系ディープラーニングモデルの入出力に合わせたデータセット形式を作成します。

# codeディレクトリの主要なファイルの説明
-'code/compare_models.py':機械学習モデル、使用したデータセットごとの学習結果の誤差をswarmプロットとして表示して比較する。
-'code/swarm_plot.png':swarmプロット。
-'code/\*/train.py':複数種類のデータセットについてモデルを学習し、結果を後続ディレクトリに保存します。
-'code/\*/compute.py':複数種類のデータセットについて学習した各モデルの誤差を計算し、結果を後続ディレクトリに保存します。
-'code/\*/dataset.py':ディープラーニングモデルのカスタムデータセットクラスを定義します。
-'code/\*/model.py':ディープラーニングモデルのアーキテクチャを設計します。
-'code/\*/\*/Bland_Altman_Plot.png':モデルの出力と正解ラベルを対象にしたBland Altman Plot。
-'code/\*/\*/best_model.pth':optunaによるハイパーパラメータ探索で決定した最良モデル。
-'code/\*/\*/best_loss.png':最良モデルの学習の際の損失関数の推移。
-'code/\*/\*/best_model_performance.json':compute.pyで計算した最良モデルに対する誤差。
-'code/\*/\*/best_param.pkl':optunaによるハイパーパラメータ探索で決定した最良モデルのパラメータの値。
-'code/\*/\*/\*_T_input.pkl':学習、検証、テストデータのpickleファイル。
-'code/\*/\*/\*_data.csv':学習、検証、テストデータとそれを入力した際のモデルの出力および対応する正解ラベルが確認できるcsvファイル。

# dataディレクトリの主要なファイルの説明
-'data/dataset_20231205.csv':COMSOLの解析データ。
-'data/\*/mk_*_dataset.py':COMSOLの解析データに特徴量エンジニアリングを行い、複数種類のデータセットを作成します。
-'data/\*/\*/\*/T_inpur.npy':モデルの入力データ。
-'data/\*/\*/\*/T_core.npy':モデルの正解ラベル。

# データセット形式の説明
-'ML'
　-'入力データの次元':[データ数,各データのチャンネル数×各データのデータ長]
　-'正解ラベルの次元':[データ数]
-'CNN'
　-'入力データの次元':[データ数,各データのチャンネル数,各データのデータ長]
　-'正解ラベルの次元':[データ数,1]
-'RNN'
　-'入力データの次元':[データ数,各データのデータ長,各データのチャンネル数]
　-'正解ラベルの次元':[データ数,1]

# 使用方法
-'data/\*/mk_\*_dataset.py'を実行することで、そのモデル(\*)用の複数種類のデータセット(T_inpur.npy、T_core.npy)が作成されます。
-'code/\*/train.py'を実行することで、複数種類のデータセットを用いたそのモデル(\*)の学習が行われ、学習、検証、テストデータ(\*_T_input)および最良モデルの情報(best_model.pth、best_loss.png、best_param.pkl)が後続ディレクトリに保存されます。
-'code/\*/compute.py'を実行することで、最良モデルに対するモデルの入出力および正解ラベルデータ(\*_data.csv)、誤差の評価用のファイル(Bland_Altman_Plot.png、best_model_performance.json)が後続ディレクトリに保存されます。
-'code/compare_models.py'を実行することで、複数の機械学習による学習の誤差をswarmプロット(swarm_plot.png)として可視化することができます。