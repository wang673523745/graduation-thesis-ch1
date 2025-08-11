import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SEBlock("max", 64, 4).to(device) #54是通道数，ratio是中间向下了多大
    feature_maps = torch.randn((1, 64, 128, 128)).to(device)
    ans = model(feature_maps)
    print(ans.shape)
