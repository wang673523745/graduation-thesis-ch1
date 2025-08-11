#绘制中间图片
from net import Restormer_Encoder, Restormer_Decoder
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from SEblock import SEBlock
import cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

def printOneImg(img_Tensor,dataset_Name,img_name,lei,data_VIS_Cr,data_VIS_Cb):
    #对这个tensor每一层进行一个输出
    # print(img_Tensor.shape)
    saveMiddata_Name = "MidImg"+dataset_Name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_Fuse = torch.zeros([480,640]).to(device)

    for i in range(64):
        data_Fuse = img_Tensor[0][i]
    
        data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
        fi = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)

        fi = fi.astype(np.uint8)
        # concatnate
        ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb)) #将其当做YCrCb中的Y
        rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
        img_save(rgb_fi, img_name.split('.')[0]+'_'+lei+'_'+str(i), saveMiddata_Name)  #将融合好的图片保存

def printImg(ckpt_path,dataset_Name,img_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(SEBlock("max", 64, 4)).to(device)
    DetailFuseLayer = nn.DataParallel(SEBlock("max", 64, 4)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    
    with torch.no_grad():
        
        data_IR=image_read_cv2(os.path.join(dataset_Name,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
        data_VIS = cv2.split(image_read_cv2(os.path.join(dataset_Name,"vi",img_name), mode='YCrCb'))[0][np.newaxis,np.newaxis, ...]/255.0
        
        # ycrcb, uint8
        data_VIS_BGR = cv2.imread(os.path.join(dataset_Name,"vi",img_name))
        _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

        data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)

        feature_V_B,feature_I_B,out_lt_vis,out_lt_ir,feature_V_D,feature_I_D = Encoder(data_VIS, data_IR)
        lei = "feature_V_B"
        printOneImg(feature_V_B,dataset_Name,img_name,lei,data_VIS_Cr,data_VIS_Cb)
        lei = "feature_I_B"
        printOneImg(feature_I_B,dataset_Name,img_name,lei,data_VIS_Cr,data_VIS_Cb)
        lei = "feature_V_D"
        printOneImg(feature_V_D,dataset_Name,img_name,lei,data_VIS_Cr,data_VIS_Cb)
        lei = "feature_I_D"
        printOneImg(feature_I_D,dataset_Name,img_name,lei,data_VIS_Cr,data_VIS_Cb)

if __name__ == '__main__':
    ckpt_path="models/CDDFuse_new+epoch:"+ str(80) + '.pth'
    dataset_Name = "test_img/MSRS"
    image_name = "00004N.png"
    printImg(ckpt_path,dataset_Name,image_name)
    print('生成完成')