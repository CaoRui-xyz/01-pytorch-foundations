import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 准备数据
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]])

# 2. Dataste + DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)    
'''
batch_size = 每一批有几个样本
len(dataloader) = 一共有几个 batch
'''

# 3. 定义线性回归模型
model = nn.Linear(in_features=1, out_features=1)

# 4. 定义损失函数
loss_fn = nn.MSELoss()

# 5. 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 6. 训练模型
epochs = 100
loss_history = []

for epoch in range(epochs):

    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        # 前向传播
        y_pred = model(batch_x)
        
        # 计算损失
        loss = loss_fn(y_pred, batch_y)
        
        # 反向传播
        optimizer.zero_grad() # 清掉之前的梯度
        loss.backward() # 计算当前的梯度
        
        # 更新参数
        optimizer.step() # 修改参数

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) # 一整个epoch 的平均 batch loss

    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

with torch.no_grad():
    y_pred = model(x)

# 7. 看看训练好的模型参数
print(f'Weight: {model.weight.item():.4f}, Bias: {model.bias.item():.4f}')

# 8. 测试模型   
test_x = torch.tensor([[7.0]])

with torch.no_grad():   
    test_y_pred = model(test_x) 

print(f'Prediction for input 7.0: {test_y_pred.item():.4f}')
print("x=7,prediction: ", test_y_pred.item())

# 9. 绘制损失曲线
plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig(
    "03_linear_regression/loss_curve.png"
)

plt.show()

plt.scatter(
    x.numpy(),
    y.numpy(),
    label="Real Data"
)

plt.plot(
    x.numpy(),
    y_pred.numpy(),
    label="Prediction"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")

plt.legend()

plt.savefig(
    "03_linear_regression/fitting_curve.png"
)

plt.show()