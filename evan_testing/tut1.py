import torch
import numpy


"""
torch.empty()
torch.rand()
torch.zeroes()
torch.ones()

"""
x = torch.ones(2, 2, dtype=torch.float16)

print(x)
print(x.dtype)
print(x.size())

x = torch.tensor([2.5, 0.1])
print(x)


x = torch.rand(2, 2)
y = torch.rand(2, 2)

print(x, y)
print(x + y)
print(x - y)
print(x * y)


z = torch.add(x, y)

print(-6 * x)

y.add_(x)
# trailing underscore will do inplace operation
