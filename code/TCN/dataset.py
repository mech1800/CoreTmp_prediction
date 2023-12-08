import torch
from torch.utils.data.dataset import Dataset

# カスタムデータセットクラス
class MyDataset(Dataset):
    def __init__(self, features, y):
        super(MyDataset, self).__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.y[idx]