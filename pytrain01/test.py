import torch
import numpy as np

a = torch.tensor([[2, 2, 3, 3, 0.5], [2, 2, 4, 4, 0.4]])
b = torch.tensor([[1, 2, 3, 4, 4 ], [3, 3, 3, 1, 1 ]])
print(torch.cat([a, b], dim=1))
