import torch
import torch.nn as nn

class DynamicTCN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, kernel_size, feature_num, dropout_rate):
        super(DynamicTCN, self).__init__()

        layers = []
        dilation_size = 1
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=dilation_size, dilation=dilation_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim  # 次の層の入力チャネル数を更新
            dilation_size *= 2  # ダイレート係数を増加

        self.tcn_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * feature_num, 1)

    def forward(self, x):
        x = self.tcn_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x