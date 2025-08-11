import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)

        self.q_dwconv = nn.Conv2d(
            dim , dim , kernel_size=3, stride=1, padding=1, groups=dim , bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)

        self.kv_dwconv = nn.Conv2d(
            dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, q, kv):
        b, c, h, w = q.shape

        kv = self.kv_dwconv(self.kv(kv))
        q = self.q_dwconv(self.q(q))

        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    def __init__(self,downsample=None):
        
        super(Bottleneck, self).__init__()
        inplanes = 64
        planes = 64
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes)

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        #卷积操作
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)

        return out

class Updownblock(nn.Module):
    def __init__(self):
        super(Updownblock, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.attention = Attention(128,8,False)
        self.bottleneck = Bottleneck()

    def forward(self, x):
        x2 = self.down(x)    
        # x2 = self.bottleneck(x2)+x2
        # x2 = self.down(x2)    
        low = F.interpolate(x2, size = x.size()[-2:], mode='bilinear', align_corners=True) #低频
        high = x - low #高频
        low = self.attention(high,low)
        high = self.attention(low,high)
        tol = torch.cat((low,high),dim=1)
        return self.relu(self.conv(tol)) + x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Updownblock().to(device) #54是通道数，ratio是中间向下了多大

    x = torch.randn((1, 128, 128, 128)).to(device)
    ans = model(x)
    print(ans.shape)