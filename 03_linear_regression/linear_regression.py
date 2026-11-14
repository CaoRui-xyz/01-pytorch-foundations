import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 准备数据
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]])

# 2. Dataste + DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)    

# 3. 定义线性回归模型
modal = nn.Linear(in_features=1, out_features=1)

# 4. 定义损失函数
loss_fn = nn.MSELoss()

# 5. 定义优化器
optimizer = torch.optim.SGD(modal.parameters(), lr=0.01)

# 6. 训练模型
epoch = 100

for epoch in range(epoch):

    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        # 前向传播
        y_pred = modal(batch_x)
        
        # 计算损失
        loss = loss_fn(y_pred, batch_y)
        
        # 反向传播
        optimizer.zero_grad() # 清掉之前的梯度
        loss.backward() # 计算当前的梯度
        
        # 更新参数
        optimizer.step() # 修改参数

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epoch}], Loss: {total_loss:.4f}')

# 7. 看看训练好的模型参数
print(f'Weight: {modal.weight.item():.4f}, Bias: {modal.bias.item():.4f}')

# 8. 测试模型   
test_x = torch.tensor([[7.0]])
test_y_pred = modal(test_x) 

print(f'Prediction for input 7.0: {test_y_pred.item():.4f}')
print("x=7,prediction: ", test_y_pred.item())