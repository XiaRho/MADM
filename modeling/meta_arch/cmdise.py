import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
import diffdist.functional as diff_dist
import math
import torch
import torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm

# Additional Package
import os
import random
from copy import deepcopy
from torch.nn import functional as F
from matplotlib import pyplot as plt
from utils.visualization import subplotimg, show_image_attention_maps
from detectron2.utils.comm import get_local_rank
from utils.dacs_transforms import get_class_masks, strong_transform, BlockMaskGenerator

logger = logging.getLogger(__name__)


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: int, device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    Use diff_dist to get gradient
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


@META_ARCH_REGISTRY.register()
class CMDISE(nn.Module):

    psweight_ignore_top = 15
    psweight_ignore_bottom = 120
    vis_max_cols = 5

    def __init__(
        self, 
        backbone, 
        sem_seg_head, 
        criterion, 
        category_head=None, 
        clip_head=None,
        sem_seg_head_sec_modal=False, 
        ema_alpha=0.999, 
        pseudo_threshold=0.968, 
        blur=True,
        color_jitter_strength=0.2, 
        color_jitter_probability=0.2, 
        train_palette=None, 
        enable_mixup=True,
        remove_amp=None, 
        pl_crop=False, 
        remove_texture=None, 
        mic=False, 
        mask_ratio=None,
        fd=False, 
        fd_attention=False,
        prompt_confidence=None, 
        rand_prompt_scale=None, 
        merge_with_pl_data=None, 
        color_aug_flag=True, 
        ####### denoise_supervise                
        denoise_supervise = False,
        denoise_timestep_range = None,
        denoise_interval = None,
        ema_w_unet=False,
        **kwargs
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if sem_seg_head_sec_modal:
            self.sem_seg_head_sec_modal = deepcopy(self.sem_seg_head)
        else:
            self.sem_seg_head_sec_modal = self.sem_seg_head
        self.criterion = criterion
        self.category_head = category_head
        self.clip_head = clip_head
        self.other_parms = kwargs
        self.train_iter_index = 0
        self.ema_alpha = ema_alpha
        self.pseudo_threshold = pseudo_threshold
        self.blur = blur
        self.color_jitter_strength = color_jitter_strength
        self.color_jitter_probability = color_jitter_probability
        self.train_palette = train_palette
        self.enable_mixup = enable_mixup
        self.remove_amp = remove_amp
        self.color_aug_flag = color_aug_flag
        if self.remove_amp is not None:
            self.random_val = torch.rand(kwargs['max_iter'] + 1,
                                         generator=torch.Generator().manual_seed(kwargs['seed'])).tolist()
        self.pl_crop = pl_crop
        self.remove_texture = remove_texture
        assert not (self.remove_amp is not None and self.remove_texture is not None)
        self.target_attention_loss = hasattr(self.backbone, 'target_attention_loss') and self.backbone.target_attention_loss
        self.masked_prompt_loss = hasattr(self.backbone.feature_extractor, 'mask_prompt_ratio') and self.backbone.feature_extractor.mask_prompt_ratio
        self.prompt_perturbation_loss = hasattr(self.backbone.feature_extractor, 'prompt_perturbation') and self.backbone.feature_extractor.prompt_perturbation
        self.mic = mic
        self.mask_ratio = mask_ratio if mask_ratio is not None else 0.7
        self.mask_gen = BlockMaskGenerator(mask_ratio=self.mask_ratio, mask_block_size=32) if self.mic else None
        assert self.masked_prompt_loss + self.prompt_perturbation_loss + self.mic <= 1
        self.fd = fd
        self.fd_attention = fd_attention
        self.prompt_confidence = prompt_confidence
        self.rand_prompt_scale = rand_prompt_scale
        if self.prompt_confidence is not None:
            assert self.rand_prompt_scale is not None
            self.backbone.feature_extractor.rand_prompt_scale = 0.5

        ###################################
        #### denoise_supervise setting ####
        ###################################
        self.denoise_supervise = denoise_supervise
        self.denoise_timestep_range = denoise_timestep_range
        self.denoise_interval = denoise_interval

        self.ema_w_unet = ema_w_unet

        self.merge_with_pl_data = merge_with_pl_data
        if self.merge_with_pl_data is not None:
            if '-' in self.merge_with_pl_data:
                self.merge_with_pl_data, self.pl_merge_val = self.merge_with_pl_data.split('-')
                self.pl_merge_val = float(self.pl_merge_val)
                assert 0 <= self.pl_merge_val <= 1
        assert self.merge_with_pl_data in {'only_pl_data', 'linear_mix', 'gradual_linear_mix', 'anti_gradual_linear_mix', 'random_choice', None}
        if self.merge_with_pl_data in {'gradual_linear_mix', 'anti_gradual_linear_mix'}:
            self.train_max_iter = kwargs['max_iter']

        if 'vis_period' in kwargs.keys():
            self.vis_period = kwargs['vis_period']
        if 'output_dir' in kwargs.keys():
            self.output_dir = kwargs['output_dir']
        self.logger = logging.getLogger(__name__)

        # assert self.other_parms['pixel_mean'] in {[0.0, 0.0, 0.0], [127.5, 127.5, 127.5]}
        # self.norm_range = '01' if self.other_parms['pixel_mean'] == [0.0, 0.0, 0.0] else '-1+1'
        if self.other_parms['pixel_mean'] == [0.0, 0.0, 0.0]:
            self.aug_mean = None
            self.aug_std = None
        else:
            assert self.other_parms['pixel_mean'] == [127.5, 127.5, 127.5]
            self.register_buffer("aug_mean", torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), False)
            self.register_buffer("aug_std", torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), False)

        self.register_buffer("pixel_mean", torch.Tensor(self.other_parms['pixel_mean']).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.other_parms['pixel_std']).view(-1, 1, 1), False)
        self.size_divisibility = self.other_parms['size_divisibility']
        self.vis_save_path = 'vis_results'
        self._inti_ema_weights()

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def vis_results(self, save_path, iter_index, vis_data=None):
        if vis_data is None:
            vis_data = self.vis_data
        vis_num = len(vis_data)
        batch_size = vis_data[0]['data'].shape[0]
        if 'attention_maps' in {i['data_type'] for i in vis_data}:
            attention_maps_exist = True
            cols = vis_num - 1
            rows = batch_size * (1 + math.ceil(len(self.other_parms['class_names']) / cols))
        else:
            attention_maps_exist = False
            rows, cols = batch_size * math.ceil(vis_num / self.vis_max_cols), min(self.vis_max_cols, vis_num)
        _, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False,
                              gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0})
        for i in range(vis_num):
            if 'resize' in vis_data[i].keys() and vis_data[i]['resize']:
                vis_data[i]['data'] = F.interpolate(vis_data[i]['data'],
                                                    size=vis_data[0]['data'].shape[2:],
                                                    mode='bilinear', align_corners=False)
            for j in range(batch_size): 
                _row = j if not attention_maps_exist else j * (1 + math.ceil(len(self.other_parms['class_names']) / cols))
                assert not attention_maps_exist
                vis_x, vis_y = i % self.vis_max_cols, j * math.ceil(vis_num / self.vis_max_cols) + i // self.vis_max_cols
                if vis_data[i]['data_type'] == 'image':
                    subplotimg(axs[vis_y][vis_x], vis_data[i]['data'][j].to(torch.float32),
                               vis_data[i]['info'], norm_mean=self.aug_mean, norm_std=self.aug_std)
                elif vis_data[i]['data_type'] == 'logits':
                    pred_softmax = torch.softmax(vis_data[i]['data'][j], dim=0)
                    _, pred_seg = torch.max(pred_softmax, dim=0)
                    subplotimg(axs[vis_y][vis_x], pred_seg, vis_data[i]['info'], palette=self.train_palette)
                elif vis_data[i]['data_type'] == 'heatmap':
                    subplotimg(axs[vis_y][vis_x], vis_data[i]['data'][j].to(torch.float32),
                               vis_data[i]['info'], heat_map=True)
                elif vis_data[i]['data_type'] == 'attention_maps':
                    now_index = (_row + 1) * cols
                    
                    if vis_data[i]['data'] is None:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                _ = self.backbone(vis_data[i]['sec_modal_images'], input_modal='others')
                                # vis_data[i]['data'] = self.backbone.feature_extractor.process_attention_map(batch_size=batch_size)['attention_maps']
                                vis_data[i]['data'] = self.backbone.get_attention_loss_params()['attention_maps']

                    attention_maps = vis_data[i]['data']
                    if isinstance(attention_maps, dict):
                        res = random.choice(list(attention_maps.keys()))
                        attention_maps = attention_maps[res]
                    else:
                        res = attention_maps.shape[-2]

                    assert attention_maps.shape[1] == len(vis_data[i]['info'])
                    for k in range(attention_maps.shape[1]):
                        atte_show = show_image_attention_maps(attention_maps[j, k, :, :], vis_data[i]['sec_modal_images'][j], 
                                                              relevnace_res=attention_maps.shape[-2], attention_norm=False)
                        subplotimg(axs[now_index // cols][now_index % cols], atte_show,
                                   vis_data[i]['info'][k] + '_{}'.format(res), norm_mean=self.aug_mean, norm_std=self.aug_std)
                        now_index += 1
                else:
                    assert vis_data[i]['data_type'] == 'label'
                    subplotimg(axs[vis_y][vis_x], vis_data[i]['data'][j], vis_data[i]['info'], palette=self.train_palette)

        for ax in axs.flat:
            ax.axis('off')
        dst_path = os.path.join(save_path, self.vis_save_path)
        os.makedirs(dst_path, exist_ok=True)
        plt.savefig(os.path.join(dst_path, '{:06d}_rank{}.png'.format(iter_index, get_local_rank())))
        plt.close()

    def _inti_ema_weights(self):
        self.backbone.ema_feature_projections = deepcopy(self.backbone.feature_projections)
        self.ema_sem_seg_head = deepcopy(self.sem_seg_head)
        self.ema_parms = [
            self.backbone.ema_feature_projections,
            self.ema_sem_seg_head
        ]
        self.updated_parms = [
            self.backbone.feature_projections,
            self.sem_seg_head
        ]
        if self.ema_w_unet:
            self.backbone.feature_extractor.ldm_extractor.ema_unet = deepcopy(self.backbone.feature_extractor.ldm_extractor.unet)
            self.ema_parms.append(self.backbone.feature_extractor.ldm_extractor.ema_unet)
            self.updated_parms.append(self.backbone.feature_extractor.ldm_extractor.unet)

        self.backbone.feature_extractor.ema_clip_project_others = deepcopy(self.backbone.feature_extractor.clip_project_others)
        self.ema_parms.append(self.backbone.feature_extractor.ema_clip_project_others)
        self.updated_parms.append(self.backbone.feature_extractor.clip_project_others)

        assert len(self.ema_parms) == len(self.updated_parms)
        for i in range(len(self.ema_parms)):
            for param in self.ema_parms[i].parameters():
                param.detach_()

        if self.fd or self.fd_attention:
            self.ori_unet = deepcopy(self.backbone.feature_extractor) if self.fd else deepcopy(self.backbone)
            for param in self.ori_unet.parameters():
                param.detach_()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.ema_alpha)
        for i in range(len(self.ema_parms)):
            for ema_param, param in zip(self.ema_parms[i].parameters(),
                                        self.updated_parms[i].parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else:
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]
    
    def get_high_res_texture(self, low_res_feats, target_sec_modal_tensor):
        high_res_texture = self.backbone.feature_extractor.ldm_extractor.high_res_texture
        return high_res_texture


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """

        if self.training:
            # ##########################
            # ##### Update teacher #####
            # ##########################
            if self.train_iter_index > 0:
                self._update_ema(self.train_iter_index)

            # ##############################
            # ##### Data preprocessing #####
            # ##############################
            source_images = [x['source_rgb'].to(self.device) for x in batched_inputs]
            source_images = [(x - self.pixel_mean) / self.pixel_std for x in source_images]
            source_images = ImageList.from_tensors(source_images, self.size_divisibility)
            # source_images.tensor: torch.Size([2, 3, 512, 512])  0~1
            if self.merge_with_pl_data is not None:
                source_pl_data = [x['source_pl_data'].to(self.device) for x in batched_inputs]
                source_pl_data = [(x - self.pixel_mean) / self.pixel_std for x in source_pl_data]
                source_pl_data = ImageList.from_tensors(source_pl_data, self.size_divisibility)
                if self.merge_with_pl_data == 'only_pl_data':
                    source_images = source_pl_data
                elif self.merge_with_pl_data == 'linear_mix':
                    source_images.tensor = (1 - self.pl_merge_val) * source_images.tensor + self.pl_merge_val * source_pl_data.tensor
                elif self.merge_with_pl_data == 'gradual_linear_mix':
                    pl_merge_val = self.train_iter_index / self.train_max_iter
                    source_images.tensor = (1 - pl_merge_val) * source_images.tensor + pl_merge_val * source_pl_data.tensor
                elif self.merge_with_pl_data == 'anti_gradual_linear_mix':
                    pl_merge_val = max(0, (1 - self.train_iter_index / (self.train_max_iter * 0.5)))
                    source_images.tensor = (1 - pl_merge_val) * source_images.tensor + pl_merge_val * source_pl_data.tensor
                elif self.merge_with_pl_data == 'random_choice':
                    if random.uniform(0, 1) > 1 - self.pl_merge_val:
                        source_images = source_pl_data

            target_sec_modal = [x['target_second_modality'].to(self.device) for x in batched_inputs]
            target_sec_modal = [(x - self.pixel_mean) / self.pixel_std for x in target_sec_modal]
            target_sec_modal = ImageList.from_tensors(target_sec_modal, self.size_divisibility)
            # target_sec_modal.tensor: torch.Size([2, 3, 512, 512])  0~1

            gt_sem_seg = [x['source_label'].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(gt_sem_seg, self.size_divisibility)
            # gt_sem_seg.tensor: torch.Size([2, 1, 512, 512])  0~num_classes, 255

            if self.remove_amp is not None or self.remove_texture is not None:
                target_sec_modal_pha = [x['target_second_modality_pha'].to(self.device) for x in batched_inputs]
                target_sec_modal_pha = [(x - self.pixel_mean) / self.pixel_std for x in target_sec_modal_pha]
                target_sec_modal_pha = ImageList.from_tensors(target_sec_modal_pha, self.size_divisibility)

            # KL divergence consistency loss 50% probability
            # if self.remove_amp is not None and self.random_val[self.train_iter_index] > 2.0:
            #     # #######################
            #     # ##### source pred #####
            #     # #######################
            #     source_pred = self.sem_seg_head(self.backbone(source_images.tensor, input_modal='rgb'))
            #     losses = self.criterion.forward_source(source_pred, gt_sem_seg.tensor)

            #     if self.random_val[self.train_iter_index] > 0.75:  # FDA_sec_modal and FDA_sec_modal
            #         kl_divergence_object = remove_batch_tensor_amp(target_sec_modal.tensor, self.remove_amp)
            #     else:  # FDA_sec_modal and Ori_sec_modal
            #         kl_divergence_object = target_sec_modal.tensor

            #     kl_divergence_1 = self.sem_seg_head_sec_modal(self.backbone(target_sec_modal_pha.tensor, input_modal='others'))
            #     kl_divergence_2 = self.sem_seg_head_sec_modal(self.backbone(kl_divergence_object, input_modal='others'))

            #     kl_loss = F.mse_loss(kl_divergence_1, kl_divergence_2)

            #     losses['kl_consist'] = kl_loss

            #     self.vis_data = [
            #         {'data_type': 'image', 'info': 'source_rgb', 'data': source_images.tensor},
            #         {'data_type': 'logits', 'info': 'source_pred', 'data': source_pred, 'resize': True},
            #         {'data_type': 'label', 'info': 'source_label', 'data': gt_sem_seg.tensor},
            #         {'data_type': 'image', 'info': 'kl_divergence_1', 'data': target_sec_modal_pha.tensor},
            #         {'data_type': 'image', 'info': 'kl_divergence_2', 'data': kl_divergence_object},
            #         {'data_type': 'logits', 'info': 'kl_divergence_1_pred', 'data': kl_divergence_1},
            #         {'data_type': 'logits', 'info': 'kl_divergence_2_pred', 'data': kl_divergence_2},
            #     ]
            #     self.train_iter_index += 1
            #     return losses

            batch_size = source_images.tensor.shape[0]
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_strength,
                'color_jitter_p': self.color_jitter_probability,
                'blur': random.uniform(0, 1) if self.blur else 0,
                'mean': self.aug_mean,
                'std': self.aug_std
            }

            # #######################
            # ##### source pred #####
            # #######################
            if self.fd:  # get intermediate features
                source_feats = self.backbone.feature_extractor(
                    dict(img=self.backbone.preprocess_image(source_images.tensor)), 
                    input_modal='rgb', 
                )[1:]
                source_pred = self.sem_seg_head(
                    self.backbone.checkpoint_forward_features(
                        features=source_feats, 
                        input_image_size=source_images.tensor.shape[-2:]
                    )
                )
            else:
                source_pred = self.sem_seg_head(self.backbone(source_images.tensor, input_modal='rgb'))

            if self.fd_attention:
                source_feats = self.backbone.atte_controller.get_average_attention()['up_cross']

            # #########################################
            # ##### create pseudo label and MixUp #####
            # #########################################
            with torch.no_grad():
 
                low_res_feats = self.backbone(target_sec_modal.tensor, input_modal='others', ema_forward=True)
                ema_logits = self.ema_sem_seg_head(low_res_feats)

                _ema_logits = F.interpolate(ema_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                ema_softmax = torch.softmax(_ema_logits.detach(), dim=1)

                pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_val = torch.sum(ps_large_p).item() / ps_size
                pseudo_weight = pseudo_val * torch.ones(pseudo_prob.shape, device=ema_softmax.device)

                # modify pseudo_weight based on prompt_confidence
                if self.prompt_confidence is not None:
                    rand_prompt_logits = self.ema_sem_seg_head(self.backbone(target_sec_modal.tensor, input_modal='rand_prompt',
                                                               ema_forward=True))
                    _rand_prompt_logits = F.interpolate(rand_prompt_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                    rand_prompt_softmax = torch.softmax(_rand_prompt_logits.detach(), dim=1)
                    rand_prompt_prob, rand_prompt_label = torch.max(rand_prompt_softmax, dim=1)
                    # lower the weight of inconsistent areas
                    consistent_ratio = torch.sum(pseudo_label == rand_prompt_label).item() / ps_size
                    pseudo_weight *= consistent_ratio
                    # pseudo_weight[pseudo_label != rand_prompt_label] *= self.prompt_confidence

                if self.pl_crop:
                    pseudo_weight[:, :self.psweight_ignore_top, :] = 0
                    if self.replace_pl_down_region_with_attention is None:
                        pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0

                if self.enable_mixup:
                    gt_pixel_weight = torch.ones((pseudo_weight.shape), device=ema_softmax.device)

                    # Apply mixing
                    mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
                    mix_masks = get_class_masks(gt_sem_seg.tensor)
                    mixed_seg_weight = pseudo_weight.clone()

                    # if self.remove_amp is not None and self.random_val[self.train_iter_index] > 0.5:
                    #     target_mixup = target_sec_modal_pha
                    # else:
                    target_mixup = target_sec_modal

                    for i in range(batch_size):
                        strong_parameters['mix'] = mix_masks[i]
                        mixed_img[i], mixed_lbl[i] = strong_transform(
                            strong_parameters,
                            data=torch.stack((source_images.tensor[i], target_mixup.tensor[i])),
                            target=torch.stack((gt_sem_seg.tensor[i][0], pseudo_label[i])),
                            color_aug_flag=self.color_aug_flag
                        )
                        _, mixed_seg_weight[i] = strong_transform(
                            strong_parameters,
                            target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                    mixed_img = torch.cat(mixed_img)
                    mixed_lbl = torch.cat(mixed_lbl)
                else:
                    mixed_img, _ = strong_transform(
                        strong_parameters,
                        data=target_sec_modal.tensor,
                        mixup=False
                    )
                    mixed_lbl = pseudo_label[None]

            # ###############################
            # ##### target pred (MixUp) #####
            # ###############################
            target_pred = self.sem_seg_head_sec_modal(self.backbone(mixed_img, input_modal='mixed'))

            # ###################################
            # ##### mask prompt consistency #####
            # ###################################
            if self.masked_prompt_loss:
                masked_target_pred = self.sem_seg_head_sec_modal(self.backbone(target_sec_modal.tensor, input_modal='masked_prompt'))
            elif self.prompt_perturbation_loss:
                masked_target_pred = self.sem_seg_head_sec_modal(self.backbone(target_sec_modal.tensor, input_modal='prompt_perturbation'))
            elif self.mic:
                strong_parameters['mix'] = None
                masked_img, _ = strong_transform(strong_parameters, data=target_sec_modal.tensor.clone())
                # Apply masking to image
                masked_img = self.mask_gen.mask_image(masked_img)
                masked_target_pred = self.sem_seg_head_sec_modal(self.backbone(masked_img, input_modal='others'))
            elif self.remove_texture is not None:
                strong_parameters['mix'] = None
                masked_img, _ = strong_transform(strong_parameters, data=target_sec_modal_pha.tensor.clone())
                masked_target_pred = self.sem_seg_head_sec_modal(self.backbone(masked_img, input_modal='others'))
            
            # #############################
            # ##### denoise_supervise #####
            # #############################
            if self.denoise_supervise:
                t_timestep = random.randint(self.denoise_timestep_range[0], self.denoise_timestep_range[1])
                t_timestep = (t_timestep, t_timestep + 1)
                s_timestep = (t_timestep[0] + self.denoise_interval, t_timestep[1] + self.denoise_interval)
                with torch.no_grad():
                    denoise_t_logits = self.ema_sem_seg_head(self.backbone(target_sec_modal.tensor, input_modal='others', ema_forward=True, timestep=t_timestep))
                    _denoise_t_logits = F.interpolate(denoise_t_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                    denoise_t_softmax = torch.softmax(_denoise_t_logits.detach(), dim=1)
                    _, denoise_t_pl = torch.max(denoise_t_softmax, dim=1)
                denoise_s_pred = self.sem_seg_head_sec_modal(self.backbone(mixed_img, input_modal='others', timestep=s_timestep))
        
            # ##########################
            # ##### calculate loss #####
            # ##########################
            loss_input = {'source_rgb_pred': source_pred, 'target_sec_modal_pred': target_pred}
            loss_target = {'source_gt': gt_sem_seg.tensor, 'target_pl': mixed_lbl, 'target_pw': mixed_seg_weight}

            if self.masked_prompt_loss or self.prompt_perturbation_loss or self.mic or self.remove_texture is not None:
                loss_target['masked_prompt_consistency'] = {
                    'pred': masked_target_pred,
                    'label': pseudo_label[None],
                    'pixel_weight': pseudo_weight,
                }
            if self.denoise_supervise:
                loss_target['denoise_consistency'] = {
                    'pred': denoise_s_pred,
                    'label': denoise_t_pl[None],
                    'pixel_weight': pseudo_weight,
                }
            
            # ############################
            # ##### feature distance #####
            # ############################
            if self.fd or self.fd_attention:
                with torch.no_grad():
                    if self.fd:
                        ori_source_feats = self.ori_unet(
                            dict(img=self.backbone.preprocess_image(source_images.tensor)), 
                            input_modal='rgb', 
                        )[1:]
                    else:
                        _ = self.ori_unet(source_images.tensor, input_modal='rgb')
                        ori_source_feats = self.ori_unet.atte_controller.get_average_attention()['up_cross']
                loss_target['feature_distance'] = {
                    'source_feats': source_feats,
                    'ori_source_feats': ori_source_feats,
                    'loss_weight': self.fd if self.fd else self.fd_attention
                }

            losses = self.criterion(loss_input, loss_target)
            self.vis_data = [
                {'data_type': 'image', 'info': 'source_rgb', 'data': source_images.tensor},
                {'data_type': 'logits', 'info': 'source_pred', 'data': source_pred, 'resize': True},
                {'data_type': 'label', 'info': 'source_label', 'data': gt_sem_seg.tensor},
                # {'data_type': 'image', 'info': 'target_rgb', 'data': target_images.tensor},
                {'data_type': 'image', 'info': 'target_sec_modal', 'data': target_sec_modal.tensor},
                {'data_type': 'logits', 'info': 'target_sec_modal_pl', 'data': _ema_logits},
                {'data_type': 'image', 'info': 'mixup_modal', 'data': mixed_img},
                {'data_type': 'logits', 'info': 'mixup_pred', 'data': target_pred, 'resize': True},
                {'data_type': 'label', 'info': 'mixup_label', 'data': mixed_lbl},
            ]
            if self.mic or self.remove_texture is not None:
                self.vis_data.extend([
                    {'data_type': 'image', 'info': 'masked_image', 'data': masked_img},
                    {'data_type': 'logits', 'info': 'masked_image_pred', 'data': masked_target_pred, 'resize': True},
                ])
            if self.prompt_confidence is not None:
                self.vis_data.extend([
                    {'data_type': 'heatmap', 'info': 'pl_prob', 'data': pseudo_prob},
                    {'data_type': 'logits', 'info': 'rand_prompt_pred', 'data': rand_prompt_softmax, 'resize': True},
                    {'data_type': 'heatmap', 'info': 'rand_prompt_prob_{:.2f}'.format(consistent_ratio), 'data': rand_prompt_prob},
                    {'data_type': 'heatmap', 'info': 'pl_weight_{:.2f}'.format(pseudo_val), 'data': pseudo_weight}
                ])
            if self.denoise_supervise:
                self.vis_data.extend([
                    {'data_type': 'logits', 'info': 'denoise_s_pred', 'data': denoise_s_pred, 'resize': True},
                    {'data_type': 'logits', 'info': 'denoise_t_pred', 'data': denoise_t_logits, 'resize': True},
                ])
            self.train_iter_index += 1
            return losses
        else:
            target_sec_modal = [x['target_second_modality'].to(self.device) for x in batched_inputs]
            target_sec_modal = [(x - self.pixel_mean) / self.pixel_std for x in target_sec_modal]
            ori_size = target_sec_modal[0].shape[1:]
            target_sec_modal = ImageList.from_tensors(target_sec_modal, self.size_divisibility)

            backbone_feats = self.backbone(target_sec_modal.tensor, input_modal='others')
            outputs = self.sem_seg_head_sec_modal(backbone_feats)

            outputs = F.interpolate(outputs, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
            outputs = outputs[:, :, :ori_size[0], :ori_size[1]]
            # print('test_input:', target_sec_modal.tensor.shape)  # torch.Size([1, 3, 448, 640])
            # print('test_output:', outputs.shape)  # torch.Size([1, 19, 112, 160])
            return [{'sem_seg': outputs}]

