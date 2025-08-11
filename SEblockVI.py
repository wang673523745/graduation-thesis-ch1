import torch
import torch.nn as nn
import numpy as np


class SEBlockVIS(nn.Module):
    def __init__(self, channels, ratio):
        super(SEBlockVIS, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # if mode == "max":
        self.global_maxpooling = self.max_pooling
        # elif mode == "avg":
        self.global_avgpooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(
            channels, channels, 7, stride=1, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, _, _ = x.shape

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        m = self.global_maxpooling(x).view(b, c)
        m = self.fc_layers(m).view(b, c, 1, 1)
        m = self.sigmoid(m)

        x = attn * m

        return x 

class SEBlockIR(nn.Module):
    def __init__(self, channels, ratio):
        super(SEBlockIR, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # if mode == "max":
        self.global_maxpooling = self.max_pooling
        # elif mode == "avg":
        self.global_avgpooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(
            channels, channels, 7, stride=1, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, _, _ = x.shape

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        
        m = self.global_maxpooling(x).view(b, c)
        m = self.fc_layers(m).view(b, c, 1, 1)
        m = self.sigmoid(m)
        x = attn * m
        
        return x

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SEBlockVIS(64, 4).to(device) #54是通道数，ratio是中间向下了多大
    feature_maps = torch.randn((1, 64, 128, 128)).to(device)
    ans = model(feature_maps)
    print(ans.shape)
