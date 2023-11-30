import torch
import torch.nn as nn

class DynamicCNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, kernel_size, feature_num, dropout_rate):
        super(DynamicCNN, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # 次の層の入力チャネル数を更新

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * feature_num, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x