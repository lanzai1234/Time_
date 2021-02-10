import torch
import torch.nn.init as i

"""
a = torch.randn(16, 4)

print(a)

b = a.view(4, 4, 4)

print(b)
"""
print(dir(i))
a = torch.empty(3, 5)
i.xavier_uniform_(a)