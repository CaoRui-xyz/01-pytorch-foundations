import torch

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])

# 先手动给一个初始参数
w = torch.tensor([[1.0]])
b = torch.tensor([[-3.5]])

# 线性部分
z = x @ w + b # @ 是矩阵乘法运算符

# Sigmoid
y_pred = sigmoid(z)


def binary_cross_entropy(y_pred, y_true):
    # 避免 log(0) 的情况
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()

loss = binary_cross_entropy(y_pred, y)

print("loss:")
print(loss.item())

print("z:")
print(z)

print("probability:")
print(y_pred)

n = x.shape[0]

error = y_pred - y

dw = (x.T @ error) / n

db = torch.sum(error) / n

print("dw:", dw.item())
print("db:", db.item())

lr = 0.1

w = w - lr * dw
b = b - lr * db

print("updated w:", w.item())
print("updated b:", b.item())

epochs = 1000
loss_history = []

for epoch in range(epochs):

    # 1. forward
    z = x @ w + b

    # 2. sigmoid
    y_pred = sigmoid(z)

    # 3. loss
    loss = binary_cross_entropy(y_pred, y)

    # 4. gradient
    error = y_pred - y

    dw = (x.T @ error) / len(x)
    db = torch.sum(error) / len(x)

    # 5. update
    w = w - lr * dw
    b = b - lr * db

    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )
print("Final w:", w.item())
print("Final b:", b.item())

z = x @ w + b
y_prob = sigmoid(z)

y_pred_class = (y_prob >= 0.5).float()

print("Probabilities:")
print(y_prob)

print("Predicted classes:")
print(y_pred_class)

print("True labels:")
print(y)

accuracy = (y_pred_class == y).float().mean()
print("Accuracy:", accuracy.item())


test_x = torch.tensor([
    [2.5],
    [3.2],
    [3.5],
    [3.8],
    [5.0]
])

test_prob = sigmoid(test_x @ w + b)
test_class = (test_prob >= 0.5).float()

print("Test x:")
print(test_x)

print("Test probability:")
print(test_prob)

print("Test class:")
print(test_class)

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Manual Logistic Regression Loss")
plt.savefig("04_logistic_regression/loss_curve.png")
plt.show()