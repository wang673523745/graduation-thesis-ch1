import torch
import torch.nn as nn

from timm.models.layers import DropPath
# from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        # trunc_normal_init)
from torch.nn.modules.utils import _pair as to_2tuple
# from mmseg.models.builder import BACKBONES

from mmcv.cnn import build_norm_layer
# from mmcv.runner import BaseModule
import math
import warnings


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):#增加一个门控卷积
    def __init__(self, dim):
        super(AttentionModule,self).__init__()
        self.conv0 = nn.Conv2d(dim, dim*2, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(
            dim, dim*2, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv_spatial2 = nn.Conv2d(
        dim, dim*2, 5, stride=1, padding=10, groups=dim, dilation=5)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.ELU()

    def forward(self, x):
        
        u = x.clone()
        attn = self.conv0(x)

        #增加了门控卷积
        x1=attn.split(int(attn.shape[1]/2),dim=1)
        gate=torch.sigmoid(x1[0])
        attn=self.activation(x1[1])*gate
        
        attn = self.conv_spatial1(attn)  

        x1=attn.split(int(attn.shape[1]/2),dim=1)
        gate=torch.sigmoid(x1[0])
        attn=self.activation(x1[1])*gate


        attn = self.conv_spatial2(attn)
        x1=attn.split(int(attn.shape[1]/2),dim=1)
        gate=torch.sigmoid(x1[0])
        attn=self.activation(x1[1])*gate


        attn = self.conv1(attn)
        return u * attn
    
class LKA_moudle(nn.Module):
    def __init__(self, d_model):
        super(LKA_moudle,self).__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LKA_moudle(128).to(device)
    x = torch.randn((1, 64, 128, 128)).to(device)
    y = torch.randn((1, 64, 128, 128)).to(device)
    ans = model(x,y)
    print(ans.shape)
