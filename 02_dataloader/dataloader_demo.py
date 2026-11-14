import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. 准备一些简单数据
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0],
    [12.0]
])

# 2. 把 x 和 y 封装成 TensorDataset
dataset = TensorDataset(x, y)

# 3. 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # shuffle=True 表示每次迭代都会打乱数据顺序

#4. 一批一批读取数据
for batch_x, batch_y in dataloader:
    print("Batch x:", batch_x)
    print("Batch y:", batch_y)
    print("---")