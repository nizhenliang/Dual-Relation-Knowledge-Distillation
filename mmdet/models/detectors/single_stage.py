import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import torch
from torchvision.ops import roi_align

import math
import torch.nn.functional as F
from .GloRe import GloRe_Unit_2D

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''
        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class ObjectRelation(nn.Module):
    def __init__(self,in_channels):
        super(ObjectRelation, self).__init__()
        self.softmax=nn.Softmax(-1)
        self.in_channels=in_channels
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # x = x.permute(1, 0, 2, 3) #   256 , 50,  300 , 300
        bbox_num = x.size(0)
        theta_x = self.theta(x).view(bbox_num, -1)   #   2 , 128 , 300 x 300
        phi_x = self.phi(x).view(bbox_num, -1)       #   2 , 128 , 150 x 150
        phi_x = phi_x.permute(1,0) 
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        f = self.softmax(f)

        return f

class SEBlock(nn.Module):
    def __init__(self,ch_in,reduction):
        super(SEBlock, self).__init__()
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
          
    def forward(self, x):
        b=x.size(0)
        feature_in=x.view(1, b,-1)#1*b*-1
        w=self.gap(feature_in) #1*b*1  
        w=self.fc(w)
        x_w=feature_in*w
        x_f=x_w.view(x.size())+x
        return x_f

# class ObjectRelation_1(nn.Module):
#     def __init__(self, in_channels):
#         super(ObjectRelation_1, self).__init__()
#         self.in_channels = in_channels
#         self.inter_channels = in_channels
#         self.softmax = nn.Softmax(-1)
#         conv_nd = nn.Conv2d
#         bn = nn.BatchNorm2d

#         self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                          kernel_size=1, stride=1, padding=0)

#         self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
#                     kernel_size=1, stride=1, padding=0)
        
#         self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                              kernel_size=1, stride=1, padding=0)

#         self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
#                            kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # x = x.permute(1, 0, 2, 3) #   256 , 50,  300 , 300
#         bbox_num = x.size(0)
#         channel_feat = x.size(1)
#         g_x = self.g(x).view(bbox_num, -1)   #   2 , 128 , 150 x 150

#         theta_x = self.theta(x).view(bbox_num, -1)   #   2 , 128 , 300 x 300
#         phi_x = self.phi(x).view(bbox_num, -1)       #   2 , 128 , 150 x 150
#         phi_x = phi_x.permute(1, 0) 
#         f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
#         f_div_C = self.softmax(f)
#         # N = f.size(-1)  #   150 x 150
#         # f_div_C = f / N #   2 , 300x300, 150x150

#         y = torch.matmul(f_div_C, g_x)  #   2, 128, 300x300
#         y = y.view(bbox_num, channel_feat, *x.size()[2:])
#         W_y = self.W(y)
#         z = W_y + x

#         return z

class ObjectRelation_1(nn.Module):
    def __init__(self, in_channels):
        super(ObjectRelation_1, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels//2
        self.softmax = nn.Softmax(-1)
        self.theta = nn.Sequential( 
            nn.Linear(self.in_channels, self.inter_channels))
        self.phi = nn.Sequential( 
            nn.Linear(self.in_channels, self.inter_channels))
        self.g = nn.Sequential( 
            nn.Linear(self.in_channels, self.inter_channels))
        self.W = nn.Linear(self.inter_channels, self.in_channels)
    
    # def resize(self, x):
    #     b=x.size(0)
    #     fc=nn.Linear(self.in_channels, b).cuda()
    #     mask=fc(x)
    #     mask=self.softmax(mask)
    #     return mask
    
    def forward(self, x):
        bbox_num = x.size(0)
        x_r=x.view(bbox_num, -1) 
        g_x = self.g(x_r)   #   2 , 128 , 150 x 150
        theta_x = self.theta(x_r)   #   2 , 128 , 300 x 300
        phi_x = self.phi(x_r)      #   2 , 128 , 150 x 150
        phi_x = phi_x.permute(1, 0) 
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        # mask=self.resize(x_r)
        # f = f*mask
        f_div_C = self.softmax(f)

        y = torch.matmul(f_div_C, g_x)  #   2, 128, 300x300
        W_y = self.W(y)
        z = W_y.view(x.size()) + x

        return z

# class ObjectRelation_2(nn.Module):
#     def __init__(self, in_channels):
#         super(ObjectRelation_2, self).__init__()
#         self.in_channels = in_channels
#         self.inter_channels = in_channels//2
#         self.softmax = nn.Softmax(-1)
#         self.theta = nn.Sequential( 
#             nn.Linear(self.in_channels, self.inter_channels))
#         self.phi = nn.Sequential( 
#             nn.Linear(self.in_channels, self.inter_channels))
#         self.g = nn.Sequential( 
#             nn.Linear(self.in_channels, self.inter_channels))
#         self.W = nn.Linear(self.inter_channels, self.in_channels)
    
#     def forward(self, x):
#         bbox_num = x.size(0)
#         x_r=x.view(bbox_num, -1) 
#         g_x = self.g(x_r)   #   2 , 128 , 150 x 150
#         theta_x = self.theta(x_r)   #   2 , 128 , 300 x 300
#         phi_x = self.phi(x_r)      #   2 , 128 , 150 x 150
#         phi_x = phi_x.permute(1, 0) 
#         f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
#         f_div_C = self.softmax(f)

#         y = torch.matmul(f_div_C, g_x)  #   2, 128, 300x300
#         W_y = self.W(y)
#         z = W_y.view(x.size()) + x

#         return z
    
@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.adaptation_type = '1x1conv'
        # self.l1=nn.L1Loss()
        self.mse=nn.MSELoss()
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
        # self.student_non_local = nn.ModuleList(
        #     [   
        #         GloRe_Unit_2D(256,64),
        #         GloRe_Unit_2D(256,64),
        #         GloRe_Unit_2D(256,256),
        #         GloRe_Unit_2D(256,256),
        #         GloRe_Unit_2D(256,256),
        #         # NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
        #         # NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
        #         # NonLocalBlockND(in_channels=256),
        #         # NonLocalBlockND(in_channels=256),
        #         # NonLocalBlockND(in_channels=256)
        #     ]
        # )
        # self.teacher_non_local = nn.ModuleList(
        #     [
        #         GloRe_Unit_2D(256,64),
        #         GloRe_Unit_2D(256,64),
        #         GloRe_Unit_2D(256,256),
        #         GloRe_Unit_2D(256,256),
        #         GloRe_Unit_2D(256,256),
        #         # NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
        #         # NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
        #         # NonLocalBlockND(in_channels=256),
        #         # NonLocalBlockND(in_channels=256),
        #         # NonLocalBlockND(in_channels=256)
        #     ]
        # )
        # self.non_local_adaptation = nn.ModuleList([
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # ])
        
        self.student_relation = nn.ModuleList(
            [
                ObjectRelation_1(in_channels=256*1*1),
                ObjectRelation_1(in_channels=256*1*1),
                # ObjectRelation_1(in_channels=256*1*1),
                # SEBlock(256*9,2),
                # SEBlock(256*9,2)
            ]
        )
        self.teacher_relation = nn.ModuleList(
            [
                ObjectRelation_1(in_channels=256*1*1),
                ObjectRelation_1(in_channels=256*1*1),
                # ObjectRelation_1(in_channels=256*1*1),
                # SEBlock(256*9,2),
                # SEBlock(256*9,2)
            ]
        )

        self.relation_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])

        # self.for_adaptation = nn.ModuleList([
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # ])
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def get_teacher_info(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        t_info = {'feat':x}
        return t_info

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img,gt_bboxes)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        # kd_nonlocal_loss = 0
        # if t_info is not None:
        #     t_feats = t_info['feat']
        #     for _i in range(len(t_feats)):
        #         s_relation = self.student_non_local[_i](x[_i])
        #         t_relation = self.teacher_non_local[_i](t_feats[_i])
        #         kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
        # losses.update(kd_glore_loss=kd_nonlocal_loss * 4e-3)
        
        kd_relation_loss = 0
        # kd_foreground_loss=0
        if t_info is not None:
            t_feats = t_info['feat']
            for _i in range(2):
                spatial_scale=x[_i].size(-1)/(x[0].size(-1)*8)
                for batch_index in range(len(gt_bboxes)):
                    s_region=roi_align(x[_i], boxes=[gt_bboxes[batch_index]], output_size=3, spatial_scale=spatial_scale)
                    t_region=roi_align(t_feats[_i], boxes=[gt_bboxes[batch_index]], output_size=3, spatial_scale=spatial_scale)
                    s_object_relation = self.student_relation[_i](s_region)
                    t_object_relation = self.teacher_relation[_i](t_region)
                    kd_relation_loss += torch.dist(self.relation_adaptation[_i](s_object_relation), t_object_relation, p=2)
                    # kd_foreground_loss += torch.dist(s_region, t_region, p=2)
                    # kd_foreground_loss += torch.dist(self.for_adaptation[_i](s_region), t_region, p=2)
        losses.update({'kd_relation_loss': kd_relation_loss*0.005}) 
        # losses.update({'kd_foreground_loss': kd_foreground_loss*0.006})
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError
