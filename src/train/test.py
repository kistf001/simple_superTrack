# import torch

# # a = torch.rand((1,3,2,1,3,3,3))
# # b = torch.rand((1,3,2,1,3,3)).unsqueeze(-1)
# # print(torch.einsum("...ij,...jk->...ik",a, b) == a @ b)
# # print()
# # print(torch.is_vulkan_available())


# inps = torch.arange(4096*21, dtype=torch.float32).view(-1, 8, 21)
# tgts = torch.arange(4096*21, dtype=torch.float32).view(-1, 8, 21)
# wewe = torch.arange(4096*21, dtype=torch.float32).view(-1, 8, 21)

# dataset = torch.utils.data.TensorDataset(inps, tgts, wewe)

# loader = torch.utils.data.DataLoader(
#     dataset, batch_size=128, pin_memory=True, shuffle=True)

# SKT = [sample for batch_ndx, sample in enumerate(loader)]
# print(list(enumerate(loader))[0])
# # for batch_ndx, sample in enumerate(loader):
# #     batch_ndx, sample = enumerate(loader)
# #     print(
# #         batch_ndx, 
# #         sample[0].shape,
# #         sample[1].shape,
# #         sample[2].shape,
# #         )

try:
    import torch_directml
except:
    print("err")