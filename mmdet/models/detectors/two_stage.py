import torch
import torch.nn as nn
import math
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math
from .GloRe import GloRe_Unit_2D
from torchvision.ops import roi_align
import torch.nn.functional as F
from tools.visualization import draw_feature_map

import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
def draw_relation(x,name):
    sc=plt.matshow(x)
    plt.colorbar(sc, shrink=0.8)
    plt.savefig('relation_matrix/'+name.replace('jpg','pdf'))

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

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


def plot_attention_mask(mask):
    mask = torch.squeeze(mask, dim=0)
    mask = mask.cpu().detach().numpy()
    plt.imshow(mask)
    plt.plot(mask)
    plt.savefig('1.png')
    print('saved')
    input()

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
    
    def resize(self, x):
        b=x.size(0)
        fc=nn.Linear(self.in_channels, b).cuda()
        mask=fc(x)
        mask=self.softmax(mask)
        return mask
    
    def forward(self, x):
        bbox_num = x.size(0)
        x_r=x.view(bbox_num, -1) 
        g_x = self.g(x_r)   #   2 , 128 , 150 x 150
        theta_x = self.theta(x_r)   #   2 , 128 , 300 x 300
        phi_x = self.phi(x_r)      #   2 , 128 , 150 x 150
        phi_x = phi_x.permute(1, 0) 
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        mask=self.resize(x_r)
        # print(mask)
        f = f*mask
        f_div_C = self.softmax(f)
        # print(f_div_C)

        y = torch.matmul(f_div_C, g_x)  #   2, 128, 300x300
        W_y = self.W(y)
        z = W_y.view(x.size()) + x

        return z, f_div_C

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        self.adaptation_type = '1x1conv'

        self.student_non_local = nn.ModuleList(
            [   
                GloRe_Unit_2D(256,64),
                GloRe_Unit_2D(256,64),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(256,256),
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                GloRe_Unit_2D(256,64),
                GloRe_Unit_2D(256,64),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(256,256),
                GloRe_Unit_2D(256,256),
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])
        
        self.student_relation = nn.ModuleList(
            [
                ObjectRelation_1(in_channels=256*3*3),
                ObjectRelation_1(in_channels=256*3*3),
            ]
        )
        self.teacher_relation = nn.ModuleList(
            [
                ObjectRelation_1(in_channels=256*3*3),
                ObjectRelation_1(in_channels=256*3*3),
            ]
        )

        self.relation_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])

        
        self.for_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            # spatial_scale=1/4
            # for batch_index in range(len(gt_bboxes)):
            #     s_region=roi_align(x[0], boxes=[gt_bboxes[batch_index]], output_size=3, spatial_scale=spatial_scale)
            #     s_object_relation = self.student_relation[0](s_region)
            # draw_feature_map(img, x[0], save_dir = 'feature_map',name ='285_base')
            # relation =self.student_non_local[0](x[0])
            # draw_feature_map(img, relation, save_dir = 'feature_map',name ='285_glore')
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def get_teacher_info(self,
                         img,
                         img_metas,
                         gt_bboxes,
                         gt_labels,
                         gt_bboxes_ignore=None,
                         gt_masks=None,
                         proposals=None,
                         t_feats=None,
                         **kwargs):
        teacher_info = {}
        x = self.extract_feat(img)
        teacher_info.update({'feat': x})
        # RPN forward and loss
        '''
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list, rpn_outs = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            teacher_info.update({'proposal_list': proposal_list})
            #   teacher_info.update({'rpn_out': rpn_outs})
        else:
            proposal_list = proposals
        '''
        '''
        roi_losses, roi_out = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                          gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks, get_out=True,
                                                          **kwargs)
        teacher_info.update(
            cls_score=roi_out['cls_score'],
            pos_index=roi_out['pos_index'],
            bbox_pred=roi_out['bbox_pred'],
            labels=roi_out['labels'],
            bbox_feats=roi_out['bbox_feats'],
            x_cls=roi_out['x_cls'],
            x_reg=roi_out['x_reg']
        )
        '''
        return teacher_info

    def with_student_proposal(self,
                              img,
                              img_metas,
                              gt_bboxes,
                              gt_labels,
                              gt_bboxes_ignore=None,
                              gt_masks=None,
                              proposals=None,
                              s_info=None,
                              t_info=None,
                              **kwargs):

        with torch.no_grad():
            _, t_roi_out = self.roi_head.forward_train(t_info['feat'], img_metas, s_info['proposal_list'],
                                                       gt_bboxes, gt_labels,
                                                       gt_bboxes_ignore, gt_masks, get_out=True,
                                                       **kwargs)

        t_cls, s_cls, pos_index, labels = t_roi_out['cls_score'], s_info['cls_score'], t_roi_out[
            'pos_index'], t_roi_out['labels']
        t_cls_pos, s_cls_pos, labels_pos = t_cls[pos_index.type(torch.bool)], s_cls[pos_index.type(torch.bool)], labels[
            pos_index.type(torch.bool)]
        teacher_prediction = torch.max(t_cls_pos, dim=1)[1]
        correct_index = (teacher_prediction == labels_pos).detach()
        t_cls_pos_correct, s_cls_pos_correct = t_cls_pos[correct_index], s_cls_pos[correct_index]
        kd_pos_cls_loss = CrossEntropy(s_cls_pos_correct, t_cls_pos_correct) * 0.005
        kd_loss = dict(kd_pos_cls_loss=kd_pos_cls_loss)
        return kd_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #   for spatial attention
        t = 0.5
        s_ratio = 1.0
        x = self.extract_feat(img)
        losses = dict()
       
        kd_nonlocal_loss = 0
        if t_info is not None:
            t_feats = t_info['feat']
            for _i in range(len(t_feats)):
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
                if _i==0:
                    # print(x[0][0,:,:,:].size(),s_relation[0,:,:,:].size(),len(img_metas))
                    draw_feature_map(img, x[0][0].unsqueeze(0), save_dir = 'feature_map/heatmap1/',name =img_metas[0]['ori_filename'].split('.')[0]+'_base')
                    draw_feature_map(img, s_relation[0].unsqueeze(0), save_dir = 'feature_map/heatmap1/',name =img_metas[0]['ori_filename'].split('.')[0]+'_glore')
        losses.update(kd_glore_loss=kd_nonlocal_loss * 4e-3*0.25)

        # import functools
        # def compare(a,b):
        #     if (a[2]-a[0])*(a[1]-a[3])<(b[2]-b[0])*(b[1]-b[3]):
        #         return -1
        #     elif (a[2]-a[0])*(a[1]-a[3])>(b[2]-b[0])*(b[1]-b[3]):
        #         return 1
        #     else:
        #         return 0
        
        kd_relation_loss = 0
        kd_foreground_loss=0
        if t_info is not None:
            t_feats = t_info['feat']
            for _i in range(2):
                spatial_scale=x[_i].size(-1)/(x[0].size(-1)*4)
                for batch_index in range(len(gt_bboxes)):
                    # print(img_metas[batch_index])
                    # gt_sort=gt_bboxes[batch_index].tolist()
                    # gt_sort.sort(key=functools.cmp_to_key(compare))
                    # gt_sort=torch.Tensor(gt_sort).cuda()
                    # s_region=roi_align(x[_i], boxes=[gt_sort], output_size=3, spatial_scale=spatial_scale)
                    # t_region=roi_align(t_feats[_i], boxes=[gt_sort], output_size=3, spatial_scale=spatial_scale)
                    s_region=roi_align(x[_i], boxes=[gt_bboxes[batch_index]], output_size=3, spatial_scale=spatial_scale)
                    t_region=roi_align(t_feats[_i], boxes=[gt_bboxes[batch_index]], output_size=3, spatial_scale=spatial_scale)
                    s_object_relation, _ = self.student_relation[_i](s_region)
                    t_object_relation, _ = self.teacher_relation[_i](t_region)
                    # if _i==0:
                    #     draw_relation(relation_matrix.cpu().detach().numpy(),img_metas[batch_index]['ori_filename'])
                    kd_relation_loss += torch.dist(self.relation_adaptation[_i](s_object_relation), t_object_relation, p=2)
                    kd_foreground_loss += torch.dist(self.for_adaptation[_i](s_region), t_region, p=2)
        losses.update({'kd_relation_loss': kd_relation_loss*0.004}) 
        losses.update({'kd_fore_loss': kd_foreground_loss*0.008*0.3})

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                          gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks,
                                                          **kwargs)
        
        losses.update(roi_losses)
        return losses

    '''
    s_info.update(proposal_list=proposal_list, cls_score=roi_out['cls_score'])
    _, student_with_teacher_proposal_outs = self.roi_head.forward_train(x, img_metas, t_info['proposal_list'],
                                                                        gt_bboxes, gt_labels,
                                                                        gt_bboxes_ignore, gt_masks, get_out=True,
                                                                        **kwargs)
    pos_index, s_cls, s_reg, t_cls, t_reg = t_info['pos_index'], student_with_teacher_proposal_outs['x_cls'], student_with_teacher_proposal_outs['x_reg'], t_info['x_cls'], t_info['x_reg']
    kd_feat_reg_loss = torch.dist(self.reg_adaptation(s_reg[pos_index]), t_reg[pos_index]) * 1e-4
    kd_feat_cls_loss = torch.dist(self.cls_adaptation(s_cls), t_cls) * 1e-4
    losses.update(kd_feat_reg_loss=kd_feat_reg_loss, kd_feat_cls_loss=kd_feat_cls_loss)
    '''

    '''
    #   distill positive objects
    t_feat, s_feat, pos_index = t_info['bbox_feats'], student_with_teacher_proposal_outs['bbox_feats'], t_info['pos_index']
    t_feat_pos, s_feat_pos = t_feat[pos_index], s_feat[pos_index]
    kd_bbox_feat_loss = torch.dist(t_feat_pos, self.bbox_feat_adaptation(s_feat_pos), p=2) * 1e-4
    t_feat_pos_flat, s_feat_pos_flat = torch.flatten(t_feat_pos, start_dim=1), torch.flatten(s_feat_pos, start_dim=1)
    t_feat_pos_relation = F.normalize(torch.mm(t_feat_pos_flat, t_feat_pos_flat.t()), p=2)
    s_feat_pos_relation = F.normalize(torch.mm(s_feat_pos_flat, s_feat_pos_flat.t()), p=2)
    kd_bbox_feat_relation_loss = torch.dist(s_feat_pos_relation, t_feat_pos_relation, p=2) * 0.01
    losses.update(kd_bbox_feat_relation_loss=kd_bbox_feat_relation_loss)
    losses.update(kd_bbox_feat_loss=kd_bbox_feat_loss)
    '''

    '''
    t_cls, s_cls, pos_index, labels = t_info['cls_score'], student_with_teacher_proposal_outs['cls_score'], t_info[
        'pos_index'], student_with_teacher_proposal_outs['labels']
    t_cls_pos, s_cls_pos, labels_pos = t_cls[pos_index.type(torch.bool)], s_cls[pos_index.type(torch.bool)], labels[
        pos_index.type(torch.bool)]
    t_prediction = torch.max(t_cls_pos, dim=1)[1]
    correct_index = t_prediction == labels_pos
    t_cls_pos_correct, s_cls_pos_correct = t_cls_pos[correct_index], s_cls_pos[correct_index]
    kd_pos_correct_cls_loss = CrossEntropy(s_cls_pos_correct, t_cls_pos_correct) * 0.05
    losses.update(kd_cls_teacher_loss=kd_pos_correct_cls_loss)
    '''
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
