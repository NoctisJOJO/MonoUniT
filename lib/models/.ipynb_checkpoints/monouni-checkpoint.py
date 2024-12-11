import torch
import torch.nn as nn
import numpy as np

from lib.backbones.resnet import resnet50
from lib.backbones.dla import dla34
from lib.backbones.dlaup import DLAUp
from lib.backbones.dlaup import DLAUpv2

import torchvision.ops.roi_align as roi_align
from lib.losses.loss_function import extract_input_from_tensor
from lib.helpers.decode_helper import _topk,_nms

def weights_init_xavier(m):
    """使用Xavier 权重初始化
    全连接层：将每个输入神经元与每个输出神经元完全连接，常用于特征的线性组合。
    卷积层：通过卷积操作提取输入数据的局部特征，常用于图像处理等空间相关的数据。
    批归一化层：用于归一化每一层的输入，保持数据分布的稳定，防止梯度问题，主要用于加速训练和提高稳定性。"""
    classname = m.__class__.__name__  # 获取该模型的类名
    if classname.find('Linear') != -1:  # 类名包含Linear 表示全连接层 
        # Xavier 初始化的目的，是使每层网络的输入和输出的方差保持一致，防止信号在网络中逐层传播时逐渐消失(初始化为全0)或放大(初始化为全1)。
        # 这对于确保梯度的良好传播，进而保证神经网络的有效训练非常重要。
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # 如果该层有偏置项，将其初始化为0，目的是在网络训练的初始阶段减少偏置对输出的影响，确保网络能更好地通过学习更新偏置值。
            nn.init.constant_(m.bias, 0.0)  # y = Weight * x + bias
    elif classname.find('Conv') != -1:  # 类名包含Conv 表示卷积层
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:  # 类名包含BatchNorm 表示批归一化层
        if m.affine:  # 如果该层有可学习的参数(affine=True)
            # affine=True表示该批归一化层中有两个可学习的参数：weight 和 bias。这两个参数会乘以标准化后的值，允许模型学习到更灵活的变换。
            # weight: 对标准化后的输出进行缩放调整
            # bias：对标准化后的输出进行平移调整
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
 
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        try:
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
        except:
            nn.init.constant_(m.bias, 0.0)

class MonoUNI(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, mean_size=None,cfg=None):
        assert downsample in [4, 8, 16, 32]  # 确保下采样是有效的（4，8，16，32）
        super().__init__()

        # 加载主干网络 DLA34，支持预训练模型，并返回多层特征
        self.backbone = globals()[backbone](pretrained=True, return_levels=True)
        self.head_conv = 256  # 头部卷积层的通道数
        # mean_size 用于表示每个类别的平均尺寸，并作为一个不可训练的参数
        self.mean_size = nn.Parameter(torch.tensor(mean_size,dtype=torch.float32),requires_grad=False)
        self.cls_num = mean_size.shape[0]  # 类别数
        channels = self.backbone.channels  # 骨干网生成的特征图的通道列表
        # 在模型运行中，随着层数加深，输入特征图分辨率逐层降低，语义信息则逐层升高
        # 因此，first_level 指定上采样开始的图层
        # 比如 downsample=4，表示模型需要较高分辨率的特征图，那么first_level就较小，first_level=2，选择更靠近输入层的特征图
        # 比如 downsample=32,表示模型选择低分辨率、语义更高层次的特征图，first=5，即从backbone的第6个特征图进行上采样
        self.first_level = int(np.log2(downsample))   # 计算下采样倍数的级别(2^first_level = downsample)
        # 根据下采样级别生成对应的缩放比例列表  eg. downsample=16, first_level=4, scales=[1,2,4,8]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # 通过neck(DLAUp)实现特征图的上采样
        self.feat_up = globals()[neck](channels[self.first_level:], scales_list=scales)
        self.cfg = cfg
        self.bin_size = 1  # 默认的深度分段大小为1
        if self.cfg['multi_bin']:  # 如果配置文件中启用了 多段深度 ，则初始化相关的深度区间
            min_f = 2250  # 最小焦距
            max_f = 2800  # 最大焦距
            interval_min = torch.tensor(self.cfg['interval_min']) - 4.5  # 深度区间的最小值调整
            self.interval_min = interval_min / max_f  # 归一化
            interval_max = torch.tensor(self.cfg['interval_max']) + 4.5 
            self.interval_max = interval_max / min_f
            # 将连续的深度值划分为多个区间（分箱）是一种离散化策略。
            # 对于目标检测和深度估计任务，有时直接回归连续的深度值较为困难，因此可以通过将深度划分成若干区间（例如 5 个）来简化问题。
            # 每个区间就对应一个深度预测的通道。
            # eg. 如果我们需要预测的深度范围是 1 - 10 米，可以将其分为 5 个区间，每个区间表示 2 米的范围。
            # 在 self.vis_depth 中，网络最终输出的通道数是 bin_size，也就是说最终预测特征图的每个位置都有 bin_size 个通道，每个通道对应一个深度分箱。
            # 对于每个像素位置，网络会对这些通道的值进行预测：
            # 假设 bin_size=5，那么输出特征图的每个像素位置会有 5 个通道的输出值，这 5 个值表示网络在该位置上对深度的估计。
            # 每个通道的值代表了模型对该像素深度位于该区间的“置信度”或“可能性”。
            # 通过这种方式，模型可以更清晰地表示目标处于不同深度区间的概率，而不需要直接预测具体的深度数值。
            # 简化了深度预测任务，提高了预测的鲁棒性。
            self.bin_size = 5  # 深度分段数量5

        # 初始化各个头部，分别用于不同的任务
        
        # 热力图(heatmap)  根据主干网络的特征图生成一个热力图，其每个通道代表一个类别的评分
        # 在3*3卷积操作后使用Relu激活函数增强模型的非线性表达能力，随后通过1*1卷积将通道数降低为类别数cls_num
        # 最终输出的热力图用于估计物体的分类信息或目标概率，帮助模型定位目标
        self.heatmap = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, self.cls_num, kernel_size=1, stride=1, padding=0, bias=True))
        # 2D偏移量预测
        self.offset_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        # 2D尺寸预测
        self.size_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        # 3D偏移量预测  *2 是残差网络，同时输入低层局部信息和高层全局信息；加入类别信息使得模型可以参考类别生成更精准的预测；+2 增加额外的2D偏移信息(x和y)
        self.offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +self.cls_num+2, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),  # 批归一化 使用批归一化（Batch Normalization）将输出的特征图进行标准化，有助于模型稳定和加快收敛。
                                                                                # 批归一化通常用于多维特征的网络层，以减少内部协变量偏移，适合偏移量和尺寸预测等回归任务。
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d(1),  # 自适应池化 将特征图池化到1*1大小，将通道上的特征进行压缩，使得输出信息与输入的空间大小无关
                                                                        # 有助于获得全局特征，适合定位物体的3D位置
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        # 3D尺寸预测
        self.size_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]  *2 +self.cls_num+2, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        # 朝向预测（heading），输出24个方向
        self.heading = nn.Sequential(nn.Conv2d(channels[self.first_level]  *2 +self.cls_num+2, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))
        # 通过特征融合和分箱（深度区间）策略实现了对深度的细化估计，有助于在多物体检测中生成不同距离范围的深度预测。
        self.vis_depth = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.LeakyReLU(inplace=True),  # f(x) = max{ αx, x } , α << 1
                                       nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        self.att_depth = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        # 深度估计不确定性
        self.vis_depth_uncer = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        self.att_depth_uncer = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        # 如果启用了多端深度预测，初始化深度区间预测头
        if self.cfg['multi_bin']:
            self.depth_bin = nn.Sequential(nn.Conv2d(channels[self.first_level] *2  +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d(1),
                                                nn.Conv2d(self.head_conv, 10, kernel_size=1, stride=1, padding=0, bias=True))
            self.depth_bin.apply(weights_init_xavier)  # 深度区间预测头使用Xavier权重初始化
        # 初始化层
        self.heatmap[-1].bias.data.fill_(-2.19)
        # 通过使用 fill_fc_weights 函数对 offset_2d 和 size_2d 的卷积层权重进行合理的初始化，
        # 模型可以在训练初期获得稳定的输出，有助于加速网络的学习过程并提高定位和尺寸预测的精度。
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)
        # 使用 xavier 初始化其他模块的权重
        self.vis_depth.apply(weights_init_xavier)
        self.att_depth.apply(weights_init_xavier)
        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)
        self.vis_depth_uncer.apply(weights_init_xavier)
        self.att_depth_uncer.apply(weights_init_xavier)

    def forward(self, input, coord_ranges,calibs, targets=None, K=100, mode='train', calib_pitch_sin=None, calib_pitch_cos=None):
        """1.特征提取：通过 backbone 提取图像特征，并通过上采样层提升分辨率。
        2.基本输出层：生成 heatmap、offset_2d 和 size_2d，用于二维平面预测。
        3.模式判断：根据训练或测试模式，处理不同的目标索引和掩码。
        4.三维 RoI 特征提取：进一步获取特定目标区域的特征，以提供精细的三维推断信息。"""
        # input：输入的图像张量[Batch,Channel,H,W]  
        # target:训练时使用的标签信息，如目标的索引和类别
        # K：在测试阶段使用的前K个最高概率的检测框数量
        device_id = input.device
        BATCH_SIZE = input.size(0)
        # feat = feature
        feat = self.backbone(input)  # 提取输入（图像）的深层特征
        feat = self.feat_up(feat[self.first_level:])  # 从第first_level层开始，使用feat_up对提取的特征进行上采样，以得到更高分辨率的特征图
        '''
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        '''
        ret = {}
        ret['heatmap']=self.heatmap(feat)
        ret['offset_2d']=self.offset_2d(feat)
        ret['size_2d']=self.size_2d(feat)
        #two stage
        assert(mode in ['train','val','test'])
        if mode=='train':   #extract train structure in the train (only) and the val mode
            # inds：目标物体的索引，用于在特征图中定位具体位置
            inds,cls_ids = targets['indices'],targets['cls_ids']  # 从target中获得索引和类别
            # mask：掩码，用于筛选有效的目标区域
            masks = targets['mask_2d']  # 二值掩码，表示目标所在的位置
        else:    #extract test structure in the test (only) and the val mode
            # _nms：非极大值抑制，去除重叠的检测框  #_topk：选择前K个概率最高的检测框  #masks：为所有选中的索引创建一个掩码
            inds,cls_ids = _topk(_nms(torch.clamp(ret['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K)[1:3]
            masks = torch.ones(inds.size()).type(torch.bool).to(device_id)
        # 调用get_roi_feat 函数，基于选定的特征区域生成Roi特征
        ret.update(self.get_roi_feat(feat,inds,masks,ret,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin, calib_pitch_cos))
        return ret

    def get_roi_feat_by_mask(self,feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin=None, calib_pitch_cos=None):       
    # 假设我们有一个特征图 feat，其维度是 [batch_size, channels, height, width]，并且在这个特征图中，我们希望提取若干感兴趣的区域（ROI，Region of Interest）。
    # 对于这个特征图上的目标检测任务，inds 和 mask 用于精确地标记和筛选这些感兴趣区域。
    # 假设场景：
    # 输入特征图尺寸：feat 维度为 [2, 64, 100, 100]，代表一个 batch 中有两个图像，每个图像的特征图大小是 100x100，有 64 个通道。
    # box2d_maps：包含每个区域的 2D 边界框信息，大小为 [batch_size, 5, height, width]，其中 5 是边界框信息的数量（例如 [batch_id, x_min, y_min, x_max, y_max]）。
    # inds：表示感兴趣区域的索引。例如 inds = [0, 1, 3] 表示从 box2d_maps 中选择第 0、1 和 3 个区域。
    # mask：布尔掩码，指定哪些区域是有效的。例如，mask = [True, False, True] 表示只需要处理 inds 中第一个和第三个区域。
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()  # num_masked_bin 代表有效的ROI区域数量
        res = {}
        if num_masked_bin!=0:
            #get box2d of each roi region
            scale_box2d_masked = extract_input_from_tensor(box2d_maps,inds,mask)
            print("scale_box2d_mask:\n",scale_box2d_masked)
            #get roi feature
            # print(torch.max(box2d_masked[:,0]))
            # print(torch.max(box2d_masked[:,1]))
            # print(torch.max(box2d_masked[:,2]))
            # print(torch.max(box2d_masked[:,3]))
            # print(torch.max(box2d_masked[:,4]))
            # 如果有有效的目标区域，则从 box2d_maps 中提取对应的区域，并使用 roi_align 从特征图 feat 中获得 7x7 大小的 ROI 特征。
            roi_feature_masked = roi_align(feat,scale_box2d_masked,[7,7])
            print("roi_feature_mask:\n",roi_feature_masked)

            # 为每个目标生成全局特征图 roi_feature_global，并与提取的 ROI 特征 roi_feature_masked 拼接在一起。
            # box2d_masked_copy 是用于指定全局区域的一个边界框张量，后续会用它来从特征图 feat 中提取全局的 ROI 特征。
            box2d_masked_copy = torch.zeros_like(scale_box2d_masked)
            print("box2d_masked_copy:\n",box2d_masked_copy)
            
            # scale_box2d_masked[:,0] 存放的是每个对象的 batch 索引值。将 scale_box2d_masked 的第 0 列复制到 box2d_masked_copy 的第 0 列，以保持 batch 信息一致。
            box2d_masked_copy[:,0] = scale_box2d_masked[:,0]
            # box2d_masked_copy[:,1] = 0
            # box2d_masked_copy[:,2] = 0 
            # 表示将所有对象的边界框右下角的 x 坐标设为 239，y设为127。此操作将全局边界框的大小固定下来，从而选取特征图中一个较大区域，作为“全局”参考区域。
            box2d_masked_copy[:,3] = 239
            box2d_masked_copy[:,4] = 127
            # roi_align 是一个用于从特征图中对齐提取感兴趣区域（ROI）的操作，将 feat 中由 box2d_masked_copy 指定的区域缩放到大小 [7, 7]。
            # roi_feature_global 得到的特征表示一个全局上下文特征（即指定的全局区域的特征），用于为目标检测提供辅助的背景信息。[num_masked_bin, channels,7,7]
            roi_feature_global = roi_align(feat,box2d_masked_copy,[7,7])
            print("roi_feature_global:\n",roi_feature_global)
            # 得到的 roi_feature_masked_ 包含了局部特征和全局上下文特征，这样可以使模型在分析目标区域时，同时参考全局信息，有助于提高目标检测精度。
            roi_feature_masked_ = torch.cat((roi_feature_masked,roi_feature_global),1)
            print("roi_feature_masked_:\n:",roi_feature_masked_)
            
            # #get coord range of each roi 坐标映射到原图像大小
            coord_ranges_mask2d = coord_ranges[scale_box2d_masked[:,0].long()]

            #map box2d coordinate from feature map size domain to original image size domain
            # 这一段代码的作用是将 scale_box2d_masked 中的二维边界框坐标从特征图尺度映射回原图尺度。
            # 具体来说，每个边界框的四个坐标（左上角和右下角的 (x, y) 坐标）会按比例缩放，以适应图像的实际尺寸。
            box2d_masked = torch.cat([scale_box2d_masked[:,0:1],
                    # scale_box2d_masked[:, 1:2] 是特征图上的边界框左上角 x 坐标。
                    # /WIDE将特征图的 x 坐标按特征图宽度进行归一化，使其范围在 [0, 1] 之间。
                    # *(coord_ranges_mask2d[:, 1, 0:1] - coord_ranges_mask2d[:, 0, 0:1]) 将归一化后的 x 坐标转换为原图的宽度比例。
                    # + coord_ranges_mask2d[:, 0, 0:1] 用于调整至原图的起始 x 坐标，从而得到原图坐标中的 x 坐标。
                       scale_box2d_masked[:,1:2]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,2:3]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2],
                       scale_box2d_masked[:,3:4]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,4:5]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2]],1)
            print("box2d_masked:\n",box2d_masked)
            roi_calibs = calibs[box2d_masked[:,0].long()]
            roi_sin = calib_pitch_sin[box2d_masked[:,0].long()]
            roi_cos = calib_pitch_cos[box2d_masked[:,0].long()]
            
            # #project the coordinate in the normal image to the camera coord by calibs
            # 使用校准信息将图像坐标转换为相机坐标系，以便后续使用深度信息估算目标的三维位置。
            # 2D边界框坐标： box2d_masked 是一个包含每个 ROI 2D 边界框信息的张量，维度为 [num_masked_bin, 5]，每行表示一个边界框，内容如下：
            # box2d_masked[:,1:3]：表示左上角的坐标 [x1, y1]。 box2d_masked[:,3:5]：表示右下角的坐标 [x2, y2]。
            # torch.cat([box2d_masked[:,1:3], torch.ones([num_masked_bin,1]).to(device_id)], -1)：将左上角坐标 x1, y1 与深度 1 组合成 [x1, y1, 1] 的三维坐标。
            # self.project2rect(roi_calibs, ...)：调用 project2rect 函数，将每个 ROI 的左上角和右下角坐标投影到三维相机坐标。
            # [:,:2]：提取投影后的 x 和 y 坐标。
            # 将左上角 [x1, y1] 和右下角 [x2, y2] 的三维相机坐标拼接，最终形成的 coords_in_camera_coord 大小为 [num_masked_bin, 4]，表示每个 ROI 的边界框在相机坐标中的坐标范围。
            coords_in_camera_coord = torch.cat([self.project2rect(roi_calibs,torch.cat([box2d_masked[:,1:3],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2],
                                          self.project2rect(roi_calibs,torch.cat([box2d_masked[:,3:5],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2]],-1)
            print("coords_in_camera_coord:\n",coords_in_camera_coord)
            
            box2d_v1 = box2d_masked[:,2:3] # 左上角的y坐标(y_min)
            box2d_v2 = box2d_masked[:,4:5] # 右下角的y坐标(y_max)
            
            # 计算每行网格的 y 坐标。box2d_v1 是顶部 y 坐标，box2d_v2 是底部 y 坐标，i 则表示网格行的步长。这里除6因为有7个行坐标
            # 最后 v_maps 为 (batch_size, bin_size, 7, 7) 的矩阵，表示每个采样框中有 bin_size 个通道
            v_maps = torch.cat([box2d_v1+i*(box2d_v2-box2d_v1)/(7-1) for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1).repeat([1,self.bin_size,1,1])
            
            # box2d_masked[:,0,1]即 batch_idx。添加至coords_in_camera_coord的第一个位置，使其包含框的批次索引信息
            coords_in_camera_coord = torch.cat([box2d_masked[:,0:1],coords_in_camera_coord],-1)
            print("coords_in_camera_coord:\n",coords_in_camera_coord)

            # 生成坐标映射图 与前面的v_maps相同，最后生成(batch_size,7,7)的垂直方向网格
            coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/6 for i in range(7)],-1).unsqueeze(1).repeat([1,7,1]).unsqueeze(1),
                                torch.cat([coords_in_camera_coord[:,2:3]+i*(coords_in_camera_coord[:,4:5]-coords_in_camera_coord[:,2:3])/6 for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1)],1)
            print("coord_maps:\n",coord_maps)
            
            # #concatenate coord maps with feature maps in the channel dim
            # 初始化一个全零矩阵，大小为(num_masked_bin, cls_num)，表示每个被mask的bin对应的类别 one-hot 编码
            cls_hots = torch.zeros(num_masked_bin,self.cls_num).to(device_id)
            
            # 生成一个序列 [0, 1, ..., num_masked_bin-1]，用于指定位于 cls_hots 中的行索引。
            # cls_ids[mask].long()：提取对应的类别索引，将 cls_hots 的相应位置设为 1.0，形成类别的 one-hot 表示。
            cls_hots[torch.arange(num_masked_bin).to(device_id),cls_ids[mask].long()] = 1.0
            
            # 拼接所有特征
            roi_feature_masked = torch.cat([roi_feature_masked_,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)
            print("roi_feature_masked:\n",roi_feature_masked)

            # 深度预测和不确定性估计
            # scale_box2d_masked[:,4]-....[:,2]，计算scale_box2d_masked中每个边界框的高度
            # 再乘以系数4*2.109375，可能是经验参数，以便后续深度缩放的计算
            # 将 scale_box2d_masked 和 box2d_masked 的高度比值作为 scale_depth，用于缩放深度值。
            scale_depth = torch.clamp((scale_box2d_masked[:,4]-scale_box2d_masked[:,2])*4*2.109375, min=1.0) / \
                          torch.clamp(box2d_masked[:,4]-box2d_masked[:,2], min=1.0)
            scale_depth = scale_depth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            print("scale_depth:\n",scale_depth)
            
            # compute 3d dimension offset 输入神经网络得到目标信息
            size3d_offset = self.size_3d(roi_feature_masked)[:,:,0,0]
            vis_depth = self.vis_depth(roi_feature_masked)
            att_depth = self.att_depth(roi_feature_masked)
            vis_depth_uncer = self.vis_depth_uncer(roi_feature_masked)
            att_depth_uncer = self.att_depth_uncer(roi_feature_masked)


            # 不同模式下的深度估计
            # 在训练和测试模式下分别进行深度预测，并根据不同分支的权重和相机校准参数计算最终深度。
            if self.cfg['multi_bin']:
                depth_bin = self.depth_bin(roi_feature_masked)[:,:,0,0]  # 输入神经网络得到深度区间信息
                res['depth_bin']= depth_bin
                vis_depth = torch.sigmoid(vis_depth)  # sigmoid 归一化进行激活

                fx = roi_calibs[:,0,0]
                fy = roi_calibs[:,1,1]
                cy = roi_calibs[:,1,2]
                cy = cy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 转化为四维张量
                # fp 是通过计算相机焦距 fx 和 fy 的平方根倒数来得到的，表示 x 和 y 轴焦距的组合量值，用于后续的深度推断。
                fp = torch.sqrt(1.0/(fx * fx) + 1.0/(fy * fy)) / 1.41421356
                fp = fp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                fy = fy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # 获得每个像素相对于相机中心的视角切片，用于描述每个像素再视角上的相对位置，应用于后续的角度变换中
                tan_maps = (v_maps - cy) * 1.0 / fy
                pitch_sin = roi_sin.view(roi_calibs.shape[0], 1, 1, 1)  # 俯仰角的正弦、余弦值
                pitch_cos = roi_cos.view(roi_calibs.shape[0], 1, 1, 1)
                
                # 这个公式的物理意义是，将物体法线的水平分量减去竖直方向上因倾角（tan_maps）带来的影响。
                # 这样可以表示每个像素位置在不同倾角下的物体法线方向，使得 norm_theta 描述了物体在相机坐标系中的真实空间方向。
                norm_theta = (pitch_cos - pitch_sin * tan_maps).float()  # 得到物体的法线方向，帮助确定目标物体的方向和姿态
                
                # 深度估计的最小和最大区间，用于限制深度的取值范围    
                interval_min = self.interval_min.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(vis_depth.shape[0],1,1,1).to(vis_depth.device)
                interval_max = self.interval_max.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(vis_depth.shape[0],1,1,1).to(vis_depth.device)
                if mode=='train':  # 训练模式下，不使用深度区间，而是直接通过公式计算深度
                    # vis_depth 是一种“可见深度”的初始估计值，取值在 [0, 1] 的范围。
                    # 通过以下公式我们将其映射到 interval_min 和 interval_max 之间的实际深度范围。
                    vis_depth_ =   vis_depth * (interval_max-interval_min) + interval_min
                    vis_depth_ = vis_depth_ * norm_theta / fp * scale_depth # 对深度进行缩放
                    ins_depth = vis_depth_ + att_depth  # 最终的实例深度，可见深度+额外深度
                    # 使用 torch.logsumexp 函数融合了 vis_depth_uncer 和 att_depth_uncer。
                    # torch.logsumexp 可用于将多个不确定性度量合并成单个度量。它使用指数空间中的和来组合不确定性信息，以得到整体的不确定性。
                    ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)
                else:   # 测试模式下，通过对深度分档进行筛选和选择来计算深度
                    depth_bin_1 = torch.softmax(depth_bin[:,:2],-1)  # depth_bin 是一个深度分类网络的输出，包含不同深度范围的概率
                    depth_bin_2 = torch.softmax(depth_bin[:,2:4],-1) # 每组分档通过softmax 转为概率分布，然后保留每组的最大概率
                    depth_bin_3 = torch.softmax(depth_bin[:,4:6],-1)
                    depth_bin_4 = torch.softmax(depth_bin[:,6:8],-1)
                    depth_bin_5 = torch.softmax(depth_bin[:,8:10],-1)
                    depth_bin = torch.cat((depth_bin_1[:,1:2],depth_bin_2[:,1:2],depth_bin_3[:,1:2],depth_bin_4[:,1:2],depth_bin_5[:,1:2]),-1)
                    _,depth_bin_max_index = torch.max(depth_bin,-1) # 最大概率的分档索引，用于选择最可能的深度范围
                    print("depth_bin_max_index:\n",depth_bin_max_index)
                    # 与训练模式相同，先将vis_depth映射到实际深度范围，再通过相机参数和俯仰角进行调整
                    vis_depth_ =   vis_depth * (interval_max-interval_min) + interval_min
                    vis_depth_ = vis_depth_ * norm_theta / fp * scale_depth 
                    vis_depth = vis_depth_[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    att_depth = att_depth[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    vis_depth_uncer = vis_depth_uncer[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    att_depth_uncer = att_depth_uncer[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    ins_depth = vis_depth + att_depth # 得到最终的深度和深度不确定性
                    ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)
            else:
                vis_depth = (-vis_depth).exp().squeeze(1)
                att_depth = att_depth.squeeze(1)
                vis_depth = vis_depth * scale_depth.squeeze(-1)
                vis_depth_uncer = vis_depth_uncer[:, 0, :, :]
                att_depth_uncer = att_depth_uncer[:, 0, :, :]
                ins_depth = vis_depth + att_depth
                ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)


            

            res['train_tag'] = torch.ones(num_masked_bin).type(torch.bool).to(device_id)
            res['heading'] = self.heading(roi_feature_masked)[:,:,0,0]
            res['vis_depth'] = vis_depth
            res['att_depth'] = att_depth
            res['ins_depth'] = ins_depth
            res['vis_depth_uncer'] = vis_depth_uncer
            res['att_depth_uncer'] = att_depth_uncer
            res['ins_depth_uncer'] = ins_depth_uncer
            res['offset_3d'] = self.offset_3d(roi_feature_masked)[:,:,0,0]
            res['size_3d']= size3d_offset
            
        else:
            res['offset_3d'] = torch.zeros([1,2]).to(device_id)
            res['size_3d'] = torch.zeros([1,3]).to(device_id)
            res['train_tag'] = torch.zeros(1).type(torch.bool).to(device_id)
            res['heading'] = torch.zeros([1,24]).to(device_id)
            res['vis_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['att_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['ins_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['vis_depth_uncer'] = torch.zeros([1,5,7,7]).to(device_id)
            res['att_depth_uncer'] = torch.zeros([1,5,7,7]).to(device_id)
            res['ins_depth_uncer'] = torch.zeros([1,5,7,7]).to(device_id)
            if self.cfg['multi_bin']:
                res['depth_bin'] = torch.zeros([1,10]).to(device_id)

        return res

    def get_roi_feat(self,feat,inds,mask,ret,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin=None, calib_pitch_cos=None):
        """get_roi_feat 生成了目标在特征图中的 2D 边框（box2d_maps），并调用 get_roi_feat_by_mask 从特征图中提取 ROI 区域特征。
        整个过程的关键在于结合 offset_2d 和 size_2d，得到目标区域的精确位置，为后续的特征分析提供支持。"""
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        
        # coord_map是每个像素在特征图中的坐标映射图
        # torch.arrange(WIDE)创建宽度范围的坐标向量，torch.arrange(HEIGHT)创建高度范围的坐标向量
        # 通过repeat和cat操作将两个方向的坐标组合成(BATCH_SIZE,2,HEIGHT,WIDE)的坐标图
        coord_map = torch.cat([torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT,1]).unsqueeze(0),\
                        torch.arange(HEIGHT).unsqueeze(-1).repeat([1,WIDE]).unsqueeze(0)],0).unsqueeze(0).repeat([BATCH_SIZE,1,1,1]).type(torch.float).to(device_id)
        
        box2d_centre = coord_map + ret['offset_2d']  # 将坐标图coord_map和offset_2d相加，得到边框中心点box2d_centre
        
        print("box2d_center:\n",box2d_centre)
        
        # 每个物体的2D边界框，以中心点box2d_centre为基准，结合size_2d的宽高生成边框范围
        box2d_maps = torch.cat([box2d_centre-ret['size_2d']/2,box2d_centre+ret['size_2d']/2],1) 
        # 得到的box2d_maps形状为(BATCH_SIZE,4,HEIGHT,WIDE)，分别对应[xmin,ymin,xmax,ymax]的边框
        
        # 将批次索引信息添加到 box2d_maps 中，使模型能够在后续处理时区分不同样本的边框信息。
        box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,1,HEIGHT,WIDE]).type(torch.float).to(device_id),box2d_maps],1)
        # 新的 box2d_maps 形状为 (BATCH_SIZE, 5, HEIGHT, WIDE)，第一维度的索引值代表了每个样本的 ID
        print("box2d_maps\n",box2d_maps)
        #box2d_maps is box2d in each bin
        # 最后，get_roi_feat 调用 get_roi_feat_by_mask 函数，将生成的 box2d_maps 作为输入，配合 inds 和 mask 信息，提取目标区域的具体特征。
        print("inds:\n",inds,"\nmask:\n",mask)
        res = self.get_roi_feat_by_mask(feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin, calib_pitch_cos)
        return res


    def project2rect(self,calib,point_img):
        """2维图像坐标系->3维相机坐标系"""
        c_u = calib[:,0,2]
        c_v = calib[:,1,2]
        f_u = calib[:,0,0]
        f_v = calib[:,1,1]
        b_x = calib[:,0,3]/(-f_u) # relative
        b_y = calib[:,1,3]/(-f_v)
        
        x = (point_img[:,0]-c_u)*point_img[:,2]/f_u + b_x
        y = (point_img[:,1]-c_v)*point_img[:,2]/f_v + b_y
        z = point_img[:,2]
        centre_by_obj = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],-1)  # [batch_size, 3]
        return centre_by_obj

    def fill_fc_weights(self, layers):
        """正态分布初始化"""
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    import torch
    net = CenterNet3D()
    print(net)
    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
