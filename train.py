# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
from test_IVF import test_ivf
import datetime

# 获取当前日期和时间
current_datetime = datetime.datetime.now()

# 格式化日期时间字符串，例如：2023-11-24_10-30-15
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# 构建文件名
file_name = f"Mix_transformer+{formatted_datetime}.txt"

# 创建文件并写入内容
with open(file_name, 'w') as file:
    file.write("Mix_tranformer" + formatted_datetime)

print(f"文件 '{file_name}' 已创建。")

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()

model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 200  # total epoch
epoch_gap = 0  # epoches of Phase I 40

lr = 2e-4
weight_decay = 0
batch_size = 16
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 5
optim_gamma = 0.8

from SEblock import SEBlock
from LKAmoudle import LKA_moudle

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)  # 创建网络
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(LKA_moudle(128)).to(device)
DetailFuseLayer = nn.DataParallel(LKA_moudle(128)).to(device)

# # optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)



scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

def norm0_1(input_tensor):
    min_value = torch.min(input_tensor)
    max_value = torch.max(input_tensor)
    normalized_tensor_0_to_1 = (input_tensor - min_value) / (max_value - min_value)
    return normalized_tensor_0_to_1

for epoch in range(num_epochs):
    ''' train '''
    for iii, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # predicted_vis,predicted_ir,out_lt_vis,out_lt_ir,feature_vis_detail,\
        # feature_ir_detail = DIDF_Encoder(data_VIS, data_IR)
        #分别是要得到的结果  要得到结果的detach  预测的结果
        out_lt_vis,out_lt_ir,out_lt_vis_detach,out_lt_ir_detach,predicted_ir,predicted_vis,\
        fea_vis_detail,fea_ir_detail = DIDF_Encoder(data_VIS, data_IR)

        feature_F_B = BaseFuseLayer(out_lt_vis+out_lt_ir )
        feature_F_D = DetailFuseLayer(fea_vis_detail+ fea_ir_detail)

        data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

        #这里是间接的
        # Contrast_loss = MSELoss(predicted_vis,out_lt_ir_detach) + MSELoss(predicted_ir,out_lt_vis_detach) #越小
       
        #这里是直接的
        Contrast_loss = MSELoss(out_lt_ir,out_lt_vis_detach) + MSELoss(out_lt_vis,out_lt_ir_detach) #越小
       
        # Contrast_loss2 = MSELoss(fea_predicted_ir,fea_vis_detail_detach) + MSELoss(fea_predicted_vis,fea_ir_detail_detach)
        # cc_loss_feature = MSELoss(feature_vis_detail, feature_ir_detail) #越大越好
        # loss_decomp =   Contrast_loss - cc_loss_feature  

        ssim_loss = Loss_ssim(data_IR, data_Fuse) + Loss_ssim(data_VIS, data_Fuse) 

        fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

        loss = fusionloss + Contrast_loss * 2 + ssim_loss

        loss.backward()
        nn.utils.clip_grad_norm_(
            DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # Determine approximate time left   batch里的
        batches_done = epoch * len(loader['train']) + iii
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if iii % 30 == 0:
            with open(file_name, 'a') as file:
                file.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
                           % (
                               epoch,
                               num_epochs,
                               iii,
                               len(loader['train']),
                               loss.item(),
                               time_left,
                           )
                           )

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                iii,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )
    if epoch >= epoch_gap+1 and epoch % 2 == 0:  # 每五次进行保存并进行测试
        checkpoint = {
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        }
        torch.save(checkpoint, os.path.join("models/CDDFuse_new+epoch:" + str(epoch) + '.pth'))
        test_ivf(file_name, epoch)

    # adjust the learning rate

    scheduler1.step()  # epoch里的
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:  # 最小lr不能小于1e-6
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

# 保存的是最后的训练好的，因为没有label
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CDDFuse_" + timestamp + '.pth'))