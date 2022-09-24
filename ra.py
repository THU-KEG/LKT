import torch

ran = torch.rand(1, 16)

with open('ran.pth', 'wb') as w:
    torch.save(ran, w)