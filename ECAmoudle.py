import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)


class eca_moudle(nn.Module):
    def __init__(self):
        super(eca_moudle, self).__init__()
        self.eca3 = eca_layer(3)
        self.eca5 = eca_layer(5)

    def forward(self, x):
        y1 = self.eca3(x)
        y2 = self.eca5(x)
        x = x*y1
        return x*y2

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = eca_moudle().to(device) 
    feature_maps = torch.randn((1, 64, 128, 128)).to(device)
    ans = model(feature_maps)
    print(ans.shape)
