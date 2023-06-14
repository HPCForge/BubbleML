import torch
from neuralop.models import FNO, UNO

input = torch.rand((8, 2, 160, 1600)).cuda().float()
label = torch.rand((8, 1, 160, 1600)).cuda().float()

model = FNO(n_modes=(32, 32),
            hidden_channels=32,
            in_channels=2,
            out_channels=1,
            n_layers=3,
            domain_padding=0.01,
            domain_padding_mode='one-sided'
           ).cuda().float()
pred = model(input)

# > torch.Size([8, 1, 448, 1312]) torch.Size([8, 1, 160, 1600])
print(pred.size(), label.size())
