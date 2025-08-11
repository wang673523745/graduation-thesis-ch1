# DSFF: Dual-Scale Feature Fusion for Multi-Modality Image Fusion

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DSFF是一个基于深度学习的多模态图像融合框架，专门用于红外-可见光图像融合和医学图像融合任务。该模型采用双尺度特征融合策略，结合Transformer和CNN的优势，实现高质量的图像融合效果。

## 🌟 主要特性

- **双尺度特征提取**: 结合全局和局部特征信息
- **多模态融合**: 支持红外-可见光、医学图像等多种模态
- **高效架构**: 基于Transformer和CNN的混合架构
- **端到端训练**: 完整的训练和推理流程
- **多评价指标**: 提供8种图像融合评价指标

## 📋 目录结构

```
DSFF/
├── net.py                 # 主要网络架构
├── train.py              # 训练脚本
├── test_IVF.py           # 红外-可见光融合测试
├── dataprocessing.py     # 数据预处理
├── utils/                # 工具函数
│   ├── dataset.py        # 数据集类
│   ├── loss.py           # 损失函数
│   ├── Evaluator.py      # 评价指标
│   └── img_read_save.py  # 图像读写
├── models/               # 模型保存目录
├── data/                 # 数据目录
├── test_img/             # 测试图像
├── test_result/          # 测试结果
└── requirements.txt      # 依赖包
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.1+ (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/DSFF.git
cd DSFF
```

2. **创建虚拟环境**
```bash
conda create -n dsff python=3.8.10
conda activate dsff
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 数据准备

1. **下载数据集**
   - MSRS数据集: [下载链接](https://github.com/Linfeng-Tang/MSRS)
   - TNO数据集: [下载链接](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)

2. **数据预处理**
```bash
# 将数据集放在对应目录
mkdir -p RoadScene_train/ir RoadScene_train/vi
mkdir -p MSRS_train/ir MSRS_train/vi

# 运行数据预处理
python dataprocessing.py
```

## 🎯 使用方法

### 训练模型

```bash
python train.py
```

**主要训练参数**:
- `num_epochs`: 训练轮数 (默认: 200)
- `batch_size`: 批次大小 (默认: 16)
- `lr`: 学习率 (默认: 2e-4)
- `coeff_mse_loss_VF`: 可见光损失系数 (默认: 1.0)
- `coeff_mse_loss_IF`: 红外损失系数 (默认: 1.0)
- `coeff_decomp`: 分解损失系数 (默认: 2.0)
- `coeff_tv`: 总变分损失系数 (默认: 5.0)

### 测试模型

```bash
# 红外-可见光融合测试
python test_IVF.py
```

**测试数据集**:
- TNO: 标准红外-可见光融合数据集
- RoadScene: 道路场景数据集
- MSRS: 多光谱遥感数据集

### 模型评估

测试结果包含以下评价指标:
- **EN**: 信息熵 - 衡量图像信息量
- **SD**: 标准差 - 衡量图像对比度
- **SF**: 空间频率 - 衡量图像清晰度
- **MI**: 互信息 - 衡量信息保留程度
- **SCD**: 差异相关性 - 衡量融合质量
- **VIF**: 视觉信息保真度
- **Qabf**: 边缘保持质量
- **SSIM**: 结构相似性

## 🔧 模型架构

### 网络组件

1. **编码器 (Restormer_Encoder)**
   - 双分支特征提取
   - Lite Transformer处理全局特征
   - INN处理局部细节特征

2. **融合层 (LKA_moudle)**
   - 大核注意力机制
   - 门控卷积增强
   - 多尺度特征融合

3. **解码器 (Restormer_Decoder)**
   - 特征重建
   - 残差连接
   - 输出融合图像

### 关键模块

- **SEBlock**: 通道注意力机制
- **ECAmoudle**: 高效通道注意力
- **AttentionModule**: 空间注意力
- **Updownblock**: 上下采样模块

## 📊 性能表现

### 红外-可见光融合结果

| 数据集 | EN | SD | SF | MI | SCD | VIF | Qabf | SSIM |
|--------|----|----|----|----|----|----|----|----|
| TNO | 7.12 | 46.0 | 13.15 | 2.19 | 1.76 | 0.77 | 0.54 | 1.03 |
| RoadScene | 7.44 | 54.67 | 16.36 | 2.3 | 1.81 | 0.69 | 0.52 | 0.98 |

### 医学图像融合结果

| 数据集 | EN | SD | SF | MI | SCD | VIF | Qabf | SSIM |
|--------|----|----|----|----|----|----|----|----|
| MRI_CT | 4.88 | 79.17 | 38.14 | 2.61 | 1.41 | 0.61 | 0.68 | 1.34 |
| MRI_PET | 4.22 | 70.74 | 29.57 | 2.03 | 1.69 | 0.71 | 0.71 | 1.49 |

## ⚙️ 参数调整

### 训练参数优化

```python
# 在train.py中调整以下参数
num_epochs = 200          # 增加训练轮数提高性能
batch_size = 16          # 根据GPU内存调整
lr = 2e-4               # 学习率调整
coeff_decomp = 2.0      # 分解损失权重
coeff_tv = 5.0          # 总变分损失权重
```

### 网络参数调整

```python
# 在net.py中调整网络参数
dim = 128               # 特征维度
num_heads = 8           # 注意力头数
ffn_expansion_factor = 2 # 前馈网络扩展因子
```

### 数据增强

```python
# 在dataprocessing.py中调整
img_size = 128          # 图像块大小
stride = 200            # 滑动步长
```

## 🛠️ 工具脚本

### 模型分析
```bash
# 计算模型FLOPs和参数量
python testFlops.py
```

### 特征可视化
```bash
# 可视化中间特征图
python printMidImage.py
```

### RGB图像生成
```bash
# 生成彩色融合图像
python make_RGBimage.py
```


## 🤝 贡献

欢迎提交Issue和Pull Request来改进DSFF项目！
