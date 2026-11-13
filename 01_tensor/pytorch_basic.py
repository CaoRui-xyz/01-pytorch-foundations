import torch

x = torch.tensor([1.0, 2.0, 3.0])

print("Tensor:", x)
print("Shape:", x.shape)
print("Dtype:", x.dtype)

print("x * 2:", x * 2)
print("Sum:", torch.sum(x))
print("Mean:", torch.mean(x))

matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("Matrix:")
print(matrix)

print("Transpose:")
print(matrix.T)