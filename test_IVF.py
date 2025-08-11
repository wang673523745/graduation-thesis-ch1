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
from LKAmoudle import LKA_moudle

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
def test_ivf(file_name,epoch):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ckpt_path="models/CDDFuse_new+epoch:"+ str(epoch) + '.pth'
    for dataset_name in ["MSRS","TNO","RoadScene"]:   #
        print("\n"*2+"="*80)
        model_name="CDDFuse    "
        print("The test result of "+dataset_name+' :')
        test_folder=os.path.join('test_img',dataset_name)
        test_out_folder=os.path.join('test_result',dataset_name)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
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
                data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

                out_lt_vis,out_lt_ir,out_lt_vis_detach,out_lt_ir_detach,predicted_ir,predicted_vis,\
                fea_vis_detail,fea_ir_detail = Encoder(data_VIS, data_IR)
                
                # torch.cat((out_lt_vis, out_lt_ir),dim=1)
                feature_F_B = BaseFuseLayer(out_lt_vis+out_lt_ir)
                feature_F_D = DetailFuseLayer(fea_vis_detail+fea_ir_detail)

                data_Fuse, feature_F = Decoder(data_VIS, feature_F_B, feature_F_D)

                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))

                fi = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
                img_save(fi, img_name.split(sep='.')[0], test_out_folder)  #将融合好的图片保存

        eval_folder=test_out_folder
        ori_img_folder=test_folder

        metric_result = np.zeros((8))
        for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
                ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
                vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
                fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                            , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                            , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                            , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

        metric_result /= len(os.listdir(eval_folder))
        print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
        print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'  #各个评价指标，了解一下各个评价指标表示的什么
                +str(np.round(metric_result[1], 2))+'\t'
                +str(np.round(metric_result[2], 2))+'\t'
                +str(np.round(metric_result[3], 2))+'\t'
                +str(np.round(metric_result[4], 2))+'\t'
                +str(np.round(metric_result[5], 2))+'\t'
                +str(np.round(metric_result[6], 2))+'\t'
                +str(np.round(metric_result[7], 2))
                )
        print("="*80)
        with open(file_name, 'a') as file:
            file.write("\n\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\n")
            file.write(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'  #各个评价指标，了解一下各个评价指标表示的什么
                +str(np.round(metric_result[1], 2))+'\t'
                +str(np.round(metric_result[2], 2))+'\t'
                +str(np.round(metric_result[3], 2))+'\t'
                +str(np.round(metric_result[4], 2))+'\t'
                +str(np.round(metric_result[5], 2))+'\t'
                +str(np.round(metric_result[6], 2))+'\t'
                +str(np.round(metric_result[7], 2))+'\n'
                )
if __name__ == '__main__':
    test_ivf()