# -- coding: utf-8 --
import torch
import torchvision
import torch.nn as nn
from thop import profile
from net import Restormer_Encoder, Restormer_Decoder
from SEblock import SEBlock
from LKAmoudle import LKA_moudle

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.DIDF_Encoder = Restormer_Encoder()
        self.DIDF_Decoder = Restormer_Decoder()
        self.BaseFuseLayer = LKA_moudle(64)
        self.DetailFuseLayer = LKA_moudle(64)
    def forward(self,data_VIS):
        out_lt_vis,out_lt_ir,out_lt_vis_detach,out_lt_ir_detach,predicted_ir,predicted_vis,\
        fea_vis_detail,fea_ir_detail = self.DIDF_Encoder(data_VIS, data_VIS)
        feature_F_B = self.BaseFuseLayer(out_lt_vis+out_lt_ir)
        feature_F_D = self.DetailFuseLayer(fea_vis_detail + fea_ir_detail)
        data_Fuse, feature_F = self.DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)
        return data_Fuse

if __name__ == "__main__":
  model = MyModel()

  # Model
  print('==> Building model..')
  # model = torchvision.models.alexnet(pretrained=False)

  dummy_input = torch.randn(1, 1, 128, 128)
  flops, params = profile(model, (dummy_input,))
  print('flops: ', flops, 'params: ', params)
  print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
