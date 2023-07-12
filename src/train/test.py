import torch

a = torch.rand((1,3,2,1,3,3,3))
b = torch.rand((1,3,2,1,3,3)).unsqueeze(-1)
print(torch.einsum("...ij,...jk->...ik",a, b) == a @ b)
print()


print(torch.is_vulkan_available())