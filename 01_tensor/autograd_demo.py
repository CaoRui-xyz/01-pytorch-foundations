import torch

x = torch.tensor(2.0, requires_grad=True) # 表示这个变量后面我要对它求梯度

y = x ** 2 + 3 * x + 2

y.backward()

print("x =", x.item())
print("y =", y.item())
print("gradient =", x.grad.item())