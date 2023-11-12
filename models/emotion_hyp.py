import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone



def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model




class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])


        for param in self.face_landback.parameters():
            param.requires_grad = False

        ###########################################################################333


        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        # ir_checkpoint = ir_checkpoint["model"]
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024,512)

        #############################################################3

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=512,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)


        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)


    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat

# 根据您提供的代码，模型的结构可以总结如下：
#
# ### 主要模块和流程
#
# 1. **数据预处理**：使用`transforms`对数据进行预处理，包括图像转换为PIL格式、随机水平翻转、调整大小、转换为Tensor、标准化以及随机擦除。
#
# 2. **数据集加载**：根据参数`args.dataset`，加载不同的数据集（RAF-DB、AffectNet、AffectNet8Class），并应用预处理转换。
#
# 3. **模型构建**：根据提供的参数（如`modeltype`），构建`pyramid_trans_expr`模型。
#
# ### 模型`pyramid_trans_expr`
#
# 1. **面部标志点提取网络（Face Landmark Backbone）**：使用`MobileFaceNet`获取面部标志点，加载预训练权重，并将其设置为不参与梯度更新。
#
# 2. **图像特征提取网络（Image Feature Backbone）**：使用IR50（一种卷积网络）来提取图像特征，同样加载预训练权重。
#
# 3. **金字塔融合变换器（Pyramid Fusion Transformer）**：这是一个自定义的Transformer网络，名为`HyVisionTransformer`，用于融合面部标志点特征和图像特征。
#
# 4. **SE块（Squeeze-and-Excitation Block）**：一个SE块用于进一步精细化特征。
#
# 5. **分类头（Classification Head）**：最后一个线性层用于将融合后的特征映射到类别标签上。
#
# ### 训练流程
#
# 1. **初始化**：设置GPU环境，种子，初始化模型和数据加载器。
#
# 2. **优化器**：根据参数选择AdamW、Adam或SGD作为优化器，并使用SAM（Sharpness-Aware Minimization）优化算法。
#
# 3. **损失函数**：使用交叉熵损失和标签平滑交叉熵损失。
#
# 4. **训练循环**：在每个epoch中，对训练集进行遍历，计算损失，进行反向传播和优化器的两步更新。同时，在验证集上评估模型性能。
#
# 5. **保存模型**：如果验证准确率超过设定阈值，保存模型的状态。
#
# ### 特点
#
# - **双流架构**：模型采用了面部标志点和图像特征的双流架构，利用面部标志点提供的精细局部信息和IR50提供的全局图像信息。
# - **自定义Transformer**：金字塔融合变换器是模型的核心，负责融合两种类型的特征，并通过自注意力机制捕捉复杂的特征关系。
# - **SE块**：SE块通过重新校准通道特征来增强模型的表示能力。
# - **优化策略**：使用SAM优化算法来改善泛化能力。
#
# 这个模型是一个复杂的深度学习架构，结合了卷积神经网络和Transformer网络的优点，用于解决面部表情识别的问题。
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embed size needs to be divisible by num heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention calculation
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
