import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# LSTMモデルの定義
class LSTMModel(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5, device='cpu'):
       super(LSTMModel, self).__init__()
       self.device = device
       self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
       self.fc = nn.Linear(hidden_dim, output_dim)

   def forward(self, x):
       h0, c0 = self.init_hidden(x.size(0))
       out, (hn, cn) = self.lstm(x, (h0, c0))
       out = self.fc(out[:, -1, :])
       return out

   def init_hidden(self, batch_size):
       h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
       c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
       return h0, c0

# 1DCNNモデルの定義
class DynamicCNN(nn.Module):
   def __init__(self, input_dim, num_layers, hidden_dim, kernel_size, feature_num, dropout_rate):
       super(DynamicCNN, self).__init__()

       layers = []
       for _ in range(num_layers):
           layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2))
           layers.append(nn.BatchNorm1d(hidden_dim))
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

# TCNモデルの定義
class Chomp1d(nn.Module):
   def __init__(self, chomp_size):
       super(Chomp1d, self).__init__()
       self.chomp_size = chomp_size

   def forward(self, x):
       return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
   def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
       super(TemporalBlock, self).__init__()
       self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
       self.chomp1 = Chomp1d(padding)
       self.relu1 = nn.ReLU()
       self.dropout1 = nn.Dropout(dropout)

       self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
       self.chomp2 = Chomp1d(padding)
       self.relu2 = nn.ReLU()
       self.dropout2 = nn.Dropout(dropout)

       self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
       self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
       self.relu = nn.ReLU()
       self.init_weights()

   def init_weights(self):
       self.conv1.weight.data.normal_(0, 0.01)
       self.conv2.weight.data.normal_(0, 0.01)
       if self.downsample is not None:
           self.downsample.weight.data.normal_(0, 0.01)

   def forward(self, x):
       out = self.net(x)
       res = x if self.downsample is None else self.downsample(x)
       return self.relu(out + res)


class DynamicTCN(nn.Module):
   def __init__(self, input_dim, num_layers, hidden_dim, kernel_size, feature_num, dropout_rate):
       super(DynamicTCN, self).__init__()

       layers = []
       for i in range(num_layers):
           dilation_size = 2 ** i
           layers.append(TemporalBlock(input_dim, hidden_dim, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout_rate))
           layers.append(nn.BatchNorm1d(hidden_dim))
           input_dim = hidden_dim

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