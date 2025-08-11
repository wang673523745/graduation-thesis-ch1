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

from LKAmoudle import LKA_moudle

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

def make_RGB(ckpt_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for dataset_name in ["MSRS-detection"]:
        model_name="CDDFuse    "
        test_folder=os.path.join('test_img',dataset_name)
        test_out_folder=os.path.join('test_RGB',dataset_name)#找到文件夹

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Encoder = nn.DataParallel(Restormer_Encoder()).to(device)  # 创建网络
        Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
        BaseFuseLayer = nn.DataParallel(LKA_moudle(128)).to(device)
        DetailFuseLayer = nn.DataParallel(LKA_moudle(128)).to(device)

        Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
        Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
        BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
        DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
        Encoder.eval()
        Decoder.eval()
        BaseFuseLayer.eval()
        DetailFuseLayer.eval()

        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder,"ir")):

                data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='YCrCb'))[0][np.newaxis,np.newaxis, ...]/255.0
                
                # ycrcb, uint8
                data_VIS_BGR = cv2.imread(os.path.join(test_folder,"vi",img_name))
                _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
        #                 out_lt_vis,out_lt_ir,out_lt_vis_detach,out_lt_ir_detach,predicted_ir,predicted_vis,\
        #                   fea_vis_detail,fea_ir_detail 
                feature_V_B,feature_I_B,out_lt_vis,out_lt_ir,_,_,feature_V_D,feature_I_D = Encoder(data_VIS, data_IR)

                feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
                feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
                data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))

                fi = np.squeeze((data_Fuse * 255.0).cpu().numpy())
                
                # float32 to uint8
                fi = fi.astype(np.uint8)
                # concatnate
                ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb)) #将其当做YCrCb中的Y
                rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
                img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)


if __name__ == '__main__':
    ckpt_path="models/CDDFuse_new+epoch:"+ str(16) + '.pth'
    make_RGB(ckpt_path)
    print('生成完成')
    