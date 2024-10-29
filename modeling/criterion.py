import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=255, already_softmax=False):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.already_softmax = already_softmax

    def forward(self, logits, label, pixel_weight=None):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        if not self.already_softmax:
            logs = self.log_softmax(logits)
        else:
            logs = torch.log(logits)

        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if pixel_weight is not None:
            assert loss.shape == pixel_weight.shape 
            loss = loss * pixel_weight
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class CmdiseCriterion(nn.Module):
    def __init__(self, num_classes=19, pseudo_threshold=0.968, reduction='mean', class_weight=None, loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.pseudo_threshold = pseudo_threshold
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        if self.class_weight is not None:
            assert isinstance(self.class_weight, list) and len(self.class_weight) == self.num_classes

    def cross_entropy(self, pred, label, pixel_weight=None, class_weight=None, reduction='mean', avg_factor=None,
                      ignore_index=255):

        loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)

        # apply weights and do the reduction
        if pixel_weight is not None:
            pixel_weight = pixel_weight.float()
        loss = weight_reduce_loss(loss, weight=pixel_weight, reduction=reduction, avg_factor=avg_factor)
        return loss

    def forward_source(self, source_rgb_pred, source_gt):
        losses = {}
        class_weight = source_rgb_pred.new_tensor(self.class_weight) if self.class_weight is not None else None
        # #################################
        # ##### calculate source loss #####
        # #################################
        source_rgb_pred = F.interpolate(source_rgb_pred, size=source_gt.shape[-2:], mode='bilinear',
                                        align_corners=False)
        source_loss = self.loss_weight * self.cross_entropy(source_rgb_pred, source_gt[:, 0], pixel_weight=None,
                                                            class_weight=class_weight, reduction=self.reduction)
        losses['source_loss'] = source_loss
        return losses

    def cal_feature_distance_loss(self, source_feats, ori_source_feats, loss_weight=1.0):
        assert len(source_feats) == len(ori_source_feats)
        loss_list = []
        for i in range(len(source_feats)):
            loss_list.append(
                F.mse_loss(input=source_feats[i], target=ori_source_feats[i])
            )
        loss = sum(loss_list) * (1 / len(source_feats)) * loss_weight
        return loss


    def forward(self, outputs, targets, **kwargs):
        """
        :param outputs: dict of tensors, {
            'source_rgb_pred': [N, num_class, H // 8, W // 8],
            'target_sec_modal_pred': [N, num_class, H // 8, W // 8]
        }
        :param targets: dict of tensors  {
            'source_gt': [N, 1, H, W],
            'target_pl': [N, 1, H, W],
            'target_pw': [N, H, W],
        }
        :return: dict of loss.   {'source_loss': , 'target_loss': }
        """
        losses = {}
        source_rgb_pred = outputs['source_rgb_pred']
        target_sec_modal_pred = outputs['target_sec_modal_pred']
        source_gt = targets['source_gt']
        target_pl = targets['target_pl']
        target_pw = targets['target_pw']
        if 'pha_loss_weight' in outputs.keys():
            tar_sec_unet_feats = outputs['tar_sec_unet_feats']
            pha_unet_feats = targets['pha_unet_feats']

        class_weight = source_rgb_pred.new_tensor(self.class_weight) if self.class_weight is not None else None

        # #################################
        # ##### calculate source loss #####
        # #################################
        source_rgb_pred = F.interpolate(source_rgb_pred, size=source_gt.shape[-2:], mode='bilinear',
                                        align_corners=False)
        source_loss = self.loss_weight * self.cross_entropy(source_rgb_pred, source_gt[:, 0], pixel_weight=None,
                                                            class_weight=class_weight, reduction=self.reduction)
        losses['source_loss'] = source_loss

        # ###################################
        # ##### calculate target loss #####
        # ###################################
        target_sec_modal_pred = F.interpolate(target_sec_modal_pred, size=target_pl.shape[-2:], mode='bilinear',
                                              align_corners=False)
        target_loss = self.loss_weight * self.cross_entropy(target_sec_modal_pred, target_pl[:, 0], pixel_weight=target_pw,
                                                            class_weight=class_weight, reduction=self.reduction)
        losses['target_loss'] = target_loss

        # ###################################
        # ##### calculate target_pha loss #####
        # ###################################
        if 'pha_loss_weight' in outputs.keys():
            assert len(tar_sec_unet_feats) == len(pha_unet_feats)
            for i in range(len(pha_unet_feats)):
                loss = F.mse_loss(tar_sec_unet_feats[i], pha_unet_feats[i]) * (1 / len(pha_unet_feats))
                if i == 0:
                    losses['target_loss_pha'] = loss
                else:
                    losses['target_loss_pha'] += loss
            losses['target_loss_pha'] *= outputs['pha_loss_weight']
        
        if 'masked_prompt_consistency' in targets.keys():
            masked_pred = targets['masked_prompt_consistency']['pred']
            label = targets['masked_prompt_consistency']['label']
            pixel_weight = targets['masked_prompt_consistency']['pixel_weight']

            masked_pred = F.interpolate(masked_pred, size=label.shape[-2:], mode='bilinear', align_corners=False)
            masked_loss = self.loss_weight * self.cross_entropy(masked_pred, label[:, 0], pixel_weight=pixel_weight, class_weight=class_weight, reduction=self.reduction)
            losses['masked_prompt_consistency_loss'] = masked_loss
            
        if 'feature_distance' in targets.keys():
            losses['feature_distance_loss'] = self.cal_feature_distance_loss(**targets['feature_distance'])
        
        if 'denoise_consistency' in targets.keys():
            '''denoise_pred = targets['denoise_consistency']['pred']
            label = targets['denoise_consistency']['label']

            denoise_pred = F.interpolate(denoise_pred, size=label.shape[-2:], mode='bilinear', align_corners=False)
            denoise_loss = self.loss_weight * self.cross_entropy(denoise_pred, label[:, 0], class_weight=class_weight, reduction=self.reduction)
            losses['denoise_consistency_loss'] = denoise_loss'''
            pred, gt, pixel_weight = targets['denoise_consistency']['pred'], targets['denoise_consistency']['gt'], targets['denoise_consistency']['pixel_weight']
            if targets['denoise_consistency']['loss_type'] == 'L1':
                denoise_loss = F.l1_loss(input=pred, target=gt) * pixel_weight
            else:
                denoise_loss = F.mse_loss(input=pred, target=gt) * pixel_weight
            losses['denoise_consistency_loss'] = denoise_loss * targets['denoise_consistency']['loss_weight']

        if 'vae_decoder_loss' in targets.keys():
            for key in targets['vae_decoder_loss'].keys():
                pred, gt, mask = targets['vae_decoder_loss'][key]['pred'], targets['vae_decoder_loss'][key]['gt'], targets['vae_decoder_loss'][key]['mask']
                if targets['vae_decoder_loss'][key]['loss_type'] == 'L1':
                    decoder_loss = F.l1_loss(input=pred, target=gt, reduction='none')
                else:
                    decoder_loss = F.mse_loss(input=pred, target=gt, reduction='none')  
                mask = F.interpolate(mask, size=gt.shape[-2:], mode='nearest').repeat(1, gt.shape[1], 1, 1)
                losses['vae_decoder_{}_loss'.format(key)] = torch.sum((decoder_loss * mask)) / decoder_loss.numel() * targets['vae_decoder_loss'][key]['loss_weight']
        
        if 'mic_decoder_loss' in targets.keys():
            pred, gt, pixel_weight = targets['mic_decoder_loss']['pred'], targets['mic_decoder_loss']['gt'], targets['mic_decoder_loss']['pixel_weight']
            if targets['mic_decoder_loss']['loss_type'] == 'L1':
                mic_decoder_loss = F.l1_loss(input=pred, target=gt) * pixel_weight
            else:
                mic_decoder_loss = F.mse_loss(input=pred, target=gt) * pixel_weight
            losses['mic_vae_decoder_loss'] = mic_decoder_loss * targets['mic_decoder_loss']['loss_weight']
        return losses
