import logging
import numpy as np
import torch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList

import random
from torch.nn import functional as F
from utils.dacs_transforms import get_class_masks, strong_transform

from modeling.meta_arch.cmdise import CMDISE
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image
from modeling.meta_arch.ldm_diffusers import vae_encoder
from utils.dacs_transforms import BlockMaskGenerator

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MTMADISE(CMDISE):

    def __init__(
        self, 
        lora_configs,
        target_modality,
        wo_lora=False,
        vae_decoder_loss='',
        vae_decoder_loss_type=None,
        vae_decoder_loss_weight=[1.0, 1.0],
        mask_diff=None,
        reg_uncertain=False,
        reg_target_palette=None,
        add_zero_grad=False,
        mic_reg=False,
        rev_noise_sup=False,
        rev_noise_end_iter=None,
        rev_noise_gradually=False,
        noise_reg=None,
        MIC_reg_wo_pl_val=False,
        w_rgb_lora=False,
        eval_with_noise=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.lora_configs = dict()
        for lora_config in lora_configs:
            name, rank, alpha = lora_config.split('_')
            assert name in {'default', 'Infrared', 'Depth', 'Event'}
            self.lora_configs[name] = dict()
            self.lora_configs[name]['rank'] = int(rank[1:])
            self.lora_configs[name]['alpha'] = int(alpha[1:])
        self.target_modality = target_modality

        self.add_zero_grad = add_zero_grad
        if len(self.lora_configs.keys()) != 0:
            self.set_multi_lora()
        self.vae_decoder_loss = vae_decoder_loss
        self.vae_decoder_loss_type = vae_decoder_loss_type
        self.vae_decoder_loss_weight = vae_decoder_loss_weight
        # if self.vae_decoder_loss:
        #     self.backbone.feature_extractor.ldm_extractor.save_unet_final_output = True

        self.mask_diff = mask_diff
        if self.mask_diff is not None:
            self.mask_val = dict()
            if self.mask_diff == 'circle':
                pass
            else:
                # Event=-1_Depth=1_Base=0
                for modality_val in self.mask_diff.split('_'):
                    modality_name, mask_val = modality_val.split('=')
                    self.mask_val[modality_name] = float(mask_val)

        self.reg_uncertain = reg_uncertain
        self.mic_reg = mic_reg
        if self.mic_reg:
            assert not self.mic
            self.mask_gen = BlockMaskGenerator(mask_ratio=self.mask_ratio, mask_block_size=32)

        if reg_target_palette is None:
            self.reg_target_palette = self.train_palette
        else:
            assert reg_target_palette == 'discrete'
            self.reg_target_palette = [
                255, 0, 255, 0, 255, 0, 127, 255, 127, 255, 127, 127, 0, 255, 255, 255,
	            255, 0, 0, 0, 255, 255, 0, 0, 127, 0, 127, 255, 255, 255, 0, 0, 0
            ]
        # cal the distance between all points
        self.reg_target_tensor = torch.randn((3, len(self.train_palette) // 3, 1, 1)).cuda()
        for i in range(len(self.train_palette)):
            self.reg_target_tensor[i % 3, i // 3, 0, 0] = self.train_palette[i] / 255.0

        # pad zero for 255 label
        zero_pad = 256 * 3 - len(self.train_palette)
        for _ in range(zero_pad):
            self.train_palette.append(0)

        zero_pad = 256 * 3 - len(self.reg_target_palette)
        for _ in range(zero_pad):
            self.reg_target_palette.append(0)

        self.rev_noise_sup = rev_noise_sup
        self.rev_noise_end_iter = rev_noise_end_iter
        self.rev_noise_gradually = rev_noise_gradually
        if self.rev_noise_gradually:
            self.train_max_iter = kwargs['max_iter']
        self.noise_reg = noise_reg
        self.MIC_reg_wo_pl_val = MIC_reg_wo_pl_val
        self.eval_with_noise = eval_with_noise
        self._inti_ema_weights()

    def set_multi_lora(self,):

        for lora_name in self.lora_configs.keys():
            lora_config = LoraConfig(
                r=self.lora_configs[lora_name]['rank'],
                lora_alpha=self.lora_configs[lora_name]['alpha'],
                
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.backbone.feature_extractor.ldm_extractor.unet.add_adapter(adapter_config=lora_config, adapter_name=lora_name)
        self.backbone.feature_extractor.ldm_extractor.unet.set_adapter(list(self.lora_configs.keys())) 
        self.backbone.feature_extractor.ldm_extractor._freeze()

    def set_lora_adapter(self, state):
        # control self.disable_adapters (True or False) & self.active_adapters (List: [str])
        if len(self.lora_configs.keys()) == 0:
            return
        if isinstance(state, str):
            state = [state]
        # if state == 'base':
        #     for _, module in unet.named_modules():
        #         if isinstance(module, BaseTunerLayer):
        #             module._disable_adapters = True
        # else:
        #     for _, module in unet.named_modules():
        #         if isinstance(module, BaseTunerLayer):
        #             module._disable_adapters = False
        #             module._active_adapter = [state]
        unet = self.backbone.feature_extractor.ldm_extractor.unet
        for _, module in unet.named_modules():
            if isinstance(module, BaseTunerLayer):
                module._active_adapter = state

    def add_zero_gead_on_unused_lora(self, used_modal):
        loss = []
        unet = self.backbone.feature_extractor.ldm_extractor.unet
        for name, p in unet.named_parameters():
            if 'lora' in name and used_modal not in name:
                # p.grad = torch.zeros_like(p)
                loss.append(torch.sum(p))
        loss = sum(loss) * 0.
        return loss

    @staticmethod
    def convert_label_to_rgb(label, palette):
        B, C, H, W = label.shape
        color_label = []
        valid_mask = (label != 255).float()
        device = label.device
        label = label.cpu().numpy()
        for i in range(B):
            _label = label[i, 0]  # [B, H, W]
            _label = Image.fromarray(_label.astype(np.uint8)).convert('P')
            _label.putpalette(palette)
            _label = _label.convert('RGB')
            _label = torch.from_numpy(np.array(_label).transpose(2, 0, 1)).to(device)
            _label = (_label / 255 - 0.5) / 0.5
            color_label.append(_label)
        color_label = torch.stack(color_label, dim=0)
        return color_label, valid_mask

    def forward(self, batched_inputs):

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

            # assert len(batched_inputs) == 1
            # close multi-target adaption
            #if len(self.lora_names) == 1:

            assert not isinstance(batched_inputs[0]['target_second_modality'], dict)
            target_modal_type = self.target_modality
            target_sec_modal = [x['target_second_modality'].to(self.device) for x in batched_inputs]

            # else:
            #     target_modal_type = random.choice(list(batched_inputs[0]['target_second_modality'].keys()))
            #     target_sec_modal = [x['target_second_modality'][target_modal_type].to(self.device) for x in batched_inputs]

            target_sec_modal = [(x - self.pixel_mean) / self.pixel_std for x in target_sec_modal]
            target_sec_modal = ImageList.from_tensors(target_sec_modal, self.size_divisibility)
            # target_sec_modal.tensor: torch.Size([2, 3, 512, 512])  0~1

            gt_sem_seg = [x['source_label'].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(gt_sem_seg, self.size_divisibility)
            # gt_sem_seg.tensor: torch.Size([2, 1, 512, 512])  0~num_classes, 255

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
                self.set_lora_adapter(state='default')

                source_input_dict = {
                    'input_modal':'rgb',
                }
                if self.mask_diff is not None:
                    modality_mask = torch.zeros_like(source_images.tensor)[:, 0:1] + self.mask_val['rgb']
                    source_input_dict['modality_mask'] = F.interpolate(modality_mask, scale_factor=1/8., mode='nearest')

                if 's' in self.vae_decoder_loss:
                    feats, source_vae_decoder_output = self.backbone(source_images.tensor, return_unet_final_output=True, **source_input_dict)
                    source_pred = self.sem_seg_head(feats)
                    # source_vae_decoder_output = self.backbone.feature_extractor.ldm_extractor.unet_final_output  # ['before_vae.decoder', 'after_vae.decoder']
                    source_color_gt, source_color_gt_mask = self.convert_label_to_rgb(label=gt_sem_seg.tensor, palette=self.reg_target_palette)
                    source_color_gt_latent = vae_encoder(vae=self.backbone.feature_extractor.ldm_extractor.vae, images=source_color_gt, encoder_block_indices=[])[0]
                else:
                    source_pred = self.sem_seg_head(self.backbone(source_images.tensor, **source_input_dict))

            if self.fd_attention:
                source_feats = self.backbone.atte_controller.get_average_attention()['up_cross']


            # get mixed_img first for target forward
            with torch.no_grad():
                if self.enable_mixup:
                    mixed_img = [None] * batch_size
                    mix_masks = get_class_masks(gt_sem_seg.tensor)
                    target_mixup = target_sec_modal
                    for i in range(batch_size):
                        strong_parameters['mix'] = mix_masks[i]
                        mixed_img[i], _ = strong_transform(
                            strong_parameters,
                            data=torch.stack((source_images.tensor[i], target_mixup.tensor[i])),
                            color_aug_flag=self.color_aug_flag
                        )
                    mixed_img = torch.cat(mixed_img)
                else:
                    mixed_img, _ = strong_transform(
                        strong_parameters,
                        data=target_sec_modal.tensor,
                        mixup=False
                    )

            # ###############################
            # ##### target pred (MixUp) #####
            # ###############################
            self.set_lora_adapter(state=target_modal_type)

            target_input_dict = {
                'input_modal':'mixed',
            }
            if self.mask_diff is not None:
                mix_masks_tensor = torch.concatenate(mix_masks, dim=0)
                s_modality_mask = torch.zeros_like(mixed_img)[:, 0:1] + self.mask_val['rgb']  # [1, 1, 512, 512]
                t_modality_mask = torch.zeros_like(mixed_img)[:, 0:1] + self.mask_val[target_modal_type]  # [1, 1, 512, 512]
                modality_mask = s_modality_mask * mix_masks_tensor + t_modality_mask * (1 - mix_masks_tensor)
                target_input_dict['modality_mask'] = F.interpolate(modality_mask, scale_factor=1/8., mode='nearest')

            if 't' in self.vae_decoder_loss:
                feats, target_vae_decoder_output = self.backbone(mixed_img, return_unet_final_output=True, **target_input_dict)
                target_pred = self.sem_seg_head(feats)
            else:
                target_pred = self.sem_seg_head_sec_modal(self.backbone(mixed_img, **target_input_dict))


            # #########################################
            # ##### create pseudo label and MixUp #####
            # #########################################
            with torch.no_grad():

                self.set_lora_adapter(state=target_modal_type)

                target_pl_input_dict = {
                    'input_modal':'others',
                    'ema_forward': True,
                }
                if self.mask_diff is not None:
                    modality_mask = torch.zeros_like(target_sec_modal.tensor)[:, 0:1] + self.mask_val[target_modal_type]
                    target_pl_input_dict['modality_mask'] = F.interpolate(modality_mask, scale_factor=1/8., mode='nearest')
                if self.rev_noise_sup and self.train_iter_index <= self.rev_noise_end_iter:
                    rev_noise_timestep = random.randint(self.denoise_timestep_range[0], self.denoise_timestep_range[1])
                    if self.rev_noise_gradually:
                        # rev_noise_timestep = int(rev_noise_timestep * (1 - self.train_iter_index / self.train_max_iter))
                        rev_noise_timestep = int(rev_noise_timestep * (1 - self.train_iter_index / self.rev_noise_end_iter))
                    rev_noise_timestep = (rev_noise_timestep, rev_noise_timestep + 1)
                    target_pl_input_dict['timestep'] = rev_noise_timestep

                if self.reg_uncertain:
                    low_res_feats, pl_vae_decoder_output = self.backbone(target_sec_modal.tensor, return_unet_final_output=True, **target_pl_input_dict)
                    pl_after_vae_decoder_output = (pl_vae_decoder_output['after_vae.decoder'] + 1) / 2  # [B, 3, H, W] 0~1
                    distance_reg = torch.norm(pl_after_vae_decoder_output[:, :, None] - self.reg_target_tensor[None], dim=1)
                    prob_reg = 1 / (distance_reg + 1e-3)  # [B, 11, H, W]
                    ema_softmax_reg = torch.softmax(prob_reg.detach(), dim=1)
                    pseudo_prob_reg, pseudo_label_reg = torch.max(ema_softmax_reg, dim=1)
                else:
                    low_res_feats = self.backbone(target_sec_modal.tensor, **target_pl_input_dict)  # 'base'  target_modal_type

                ema_logits = self.ema_sem_seg_head(low_res_feats)

                _ema_logits = F.interpolate(ema_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                ema_softmax = torch.softmax(_ema_logits.detach(), dim=1)

                pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
                if self.mic_reg:
                    pl_color, _ = self.convert_label_to_rgb(label=pseudo_label[:, None], palette=self.reg_target_palette)
                    pl_color_latent = vae_encoder(vae=self.backbone.feature_extractor.ldm_extractor.vae, images=pl_color, encoder_block_indices=[])[0]
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_val = torch.sum(ps_large_p).item() / ps_size
                pseudo_weight = pseudo_val * torch.ones(pseudo_prob.shape, device=ema_softmax.device)

                if self.pl_crop:
                    pseudo_weight[:, :self.psweight_ignore_top, :] = 0

                if self.enable_mixup:
                    '''gt_pixel_weight = torch.ones((pseudo_weight.shape), device=ema_softmax.device)

                    # Apply mixing
                    mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
                    mix_masks = get_class_masks(gt_sem_seg.tensor)
                    mixed_seg_weight = pseudo_weight.clone()

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
                    mixed_lbl = torch.cat(mixed_lbl)'''
                    gt_pixel_weight = torch.ones((pseudo_weight.shape), device=ema_softmax.device)

                    # Apply mixing
                    mixed_lbl = [None] * batch_size
                    mixed_seg_weight = pseudo_weight.clone()

                    for i in range(batch_size):
                        strong_parameters['mix'] = mix_masks[i]
                        _, mixed_lbl[i] = strong_transform(
                            strong_parameters,
                            target=torch.stack((gt_sem_seg.tensor[i][0], pseudo_label[i])),
                        )
                        _, mixed_seg_weight[i] = strong_transform(
                            strong_parameters,
                            target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                    mixed_lbl = torch.cat(mixed_lbl)
                else:
                    mixed_lbl = pseudo_label[None]

            if 't' in self.vae_decoder_loss:
                target_color_gt, target_color_gt_mask = self.convert_label_to_rgb(label=mixed_lbl, palette=self.reg_target_palette)
                target_color_gt_latent = vae_encoder(vae=self.backbone.feature_extractor.ldm_extractor.vae, images=target_color_gt, encoder_block_indices=[])[0]
                target_color_gt_mask *= pseudo_weight[:, None]
    
            # ###################################
            # ##### mask prompt consistency #####
            # ###################################
            if self.mic or self.mic_reg:
                strong_parameters['mix'] = None
                masked_img, _ = strong_transform(strong_parameters, data=target_sec_modal.tensor.clone())
                # Apply masking to image
                masked_img = self.mask_gen.mask_image(masked_img)
                self.set_lora_adapter(state=target_modal_type)

                mic_input_dict = {
                    'input_modal':'others',
                }
                if self.mask_diff is not None:
                    modality_mask = torch.zeros_like(masked_img)[:, 0:1] + self.mask_val[target_modal_type]
                    mic_input_dict['modality_mask'] = F.interpolate(modality_mask, scale_factor=1/8., mode='nearest')
                if self.mic:
                    masked_target_pred = self.sem_seg_head_sec_modal(self.backbone(masked_img, **mic_input_dict))
                else:
                    _, masked_vae_decoder_output = self.backbone(masked_img, return_unet_final_output=True, **mic_input_dict)
            
            # #############################
            # ##### denoise_supervise #####
            # #############################
            if self.denoise_supervise:
                s_timestep = random.randint(self.denoise_timestep_range[0], self.denoise_timestep_range[1])
                s_timestep = (s_timestep, s_timestep + 1)
                self.set_lora_adapter(state=target_modal_type)
                if self.denoise_supervise > 0:
                    denoise_feats, denoise_s_vae_decoder_output = self.backbone(
                        target_sec_modal.tensor, input_modal='others', timestep=s_timestep, return_unet_final_output=True
                    )
                else:
                    with torch.no_grad():
                        denoise_feats, denoise_s_vae_decoder_output = self.backbone(
                            target_sec_modal.tensor, input_modal='others', timestep=s_timestep, return_unet_final_output=True
                        )
                with torch.no_grad():
                    denoise_ema_logits = self.ema_sem_seg_head(denoise_feats)
                    denoise_ema_logits = F.interpolate(denoise_ema_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)


            # ################################
            # ##### noise_regularization #####
            # ################################
            if self.noise_reg is not None:
                self.set_lora_adapter(state=target_modal_type)

                strong_parameters['mix'] = None
                aug_target_img, _ = strong_transform(strong_parameters, data=target_sec_modal.tensor.clone())

                noise_reg_t = random.randint(self.denoise_timestep_range[0], self.denoise_timestep_range[1])
                noise_reg_t = (noise_reg_t, noise_reg_t + 1)

                _, noise_reg_vae_decoder_output = self.backbone(aug_target_img, input_modal='others', return_unet_final_output=True)

                # get noise_reg_pl_color_latent to supervise the 
                with torch.no_grad():
                    noise_reg_logits = self.ema_sem_seg_head(self.backbone(target_sec_modal.tensor, timestep=noise_reg_t, input_modal='others', ema_forward=True))
                    noise_reg_logits = F.interpolate(noise_reg_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                    _, noise_reg_pl = torch.max(torch.softmax(noise_reg_logits, dim=1), dim=1)
                    noise_reg_pl_color, _ = self.convert_label_to_rgb(label=noise_reg_pl[:, None], palette=self.reg_target_palette)
                    noise_reg_pl_color_latent = vae_encoder(vae=self.backbone.feature_extractor.ldm_extractor.vae, images=noise_reg_pl_color, encoder_block_indices=[])[0]

            # ##########################
            # ##### calculate loss #####
            # ##########################
            loss_input = {'source_rgb_pred': source_pred, 'target_sec_modal_pred': target_pred}
            loss_target = {'source_gt': gt_sem_seg.tensor, 'target_pl': mixed_lbl, 'target_pw': mixed_seg_weight}

            if self.masked_prompt_loss or self.prompt_perturbation_loss or self.mic:
                loss_target['masked_prompt_consistency'] = {
                    'pred': masked_target_pred,
                    'label': pseudo_label[None],
                    'pixel_weight': pseudo_weight,
                }
            if self.mic_reg:
                if self.MIC_reg_wo_pl_val:
                    mic_reg_pixel_weight = 1.0
                else:
                    mic_reg_pixel_weight = pseudo_val
                loss_target['mic_decoder_loss'] = {
                    'pred': masked_vae_decoder_output['before_vae.decoder'],
                    'gt': pl_color_latent,
                    'pixel_weight': mic_reg_pixel_weight,
                    'loss_weight': self.mic_reg,
                    'loss_type': self.vae_decoder_loss_type,
                }
            if self.denoise_supervise and self.denoise_supervise > 0:
                loss_target['denoise_consistency'] = {
                    # 'pred': denoise_s_pred,
                    # 'label': denoise_t_pl[None],
                    # 'pixel_weight': pseudo_weight,
                    'pred': denoise_s_vae_decoder_output['before_vae.decoder'],
                    'gt': pl_color_latent,
                    'pixel_weight': pseudo_val,
                    'loss_weight': self.denoise_supervise,
                    'loss_type': self.vae_decoder_loss_type,
                }
            if self.vae_decoder_loss is not None:
                loss_target['vae_decoder_loss'] = dict()
                if 's' in self.vae_decoder_loss:
                    loss_target['vae_decoder_loss']['source'] = {
                        'pred': source_vae_decoder_output['before_vae.decoder'],
                        'gt': source_color_gt_latent,
                        'mask': source_color_gt_mask,
                        'loss_weight': self.vae_decoder_loss_weight[0],
                        'loss_type': self.vae_decoder_loss_type,
                    }
                if 't' in self.vae_decoder_loss:
                    loss_target['vae_decoder_loss']['target'] = {
                        'pred': target_vae_decoder_output['before_vae.decoder'],
                        'gt': target_color_gt_latent,
                        'mask': target_color_gt_mask,
                        'loss_weight': self.vae_decoder_loss_weight[1],
                        'loss_type': self.vae_decoder_loss_type,
                    }

            # ################################
            # ##### noise_regularization #####
            # ################################
            if self.noise_reg is not None:
                loss_target['noise_reg_loss'] = {
                    'pred': noise_reg_vae_decoder_output['before_vae.decoder'],
                    'gt': noise_reg_pl_color_latent,
                    'loss_weight': self.noise_reg,
                    'loss_type': self.vae_decoder_loss_type,
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
            
            self.train_iter_index += 1
            if self.train_iter_index % self.vis_period == 0:

                target_sec_modal_pl_info = 'target_sec_modal_pl'
                # modify the time info for rev_noise_sup
                if self.rev_noise_sup and self.train_iter_index <= self.rev_noise_end_iter:
                    target_sec_modal_pl_info += '_{}_t'.format(rev_noise_timestep[0])

                vis_data = [
                    {'data_type': 'image', 'info': 'source_rgb', 'data': source_images.tensor},
                    {'data_type': 'logits', 'info': 'source_pred', 'data': source_pred, 'resize': True},
                    {'data_type': 'label', 'info': 'source_label', 'data': gt_sem_seg.tensor},
                    # {'data_type': 'image', 'info': 'target_rgb', 'data': target_images.tensor},
                    {'data_type': 'image', 'info': 'target_sec_modal', 'data': target_sec_modal.tensor},
                    {'data_type': 'logits', 'info': target_sec_modal_pl_info, 'data': _ema_logits},
                    {'data_type': 'image', 'info': 'mixup_modal', 'data': mixed_img},
                    {'data_type': 'logits', 'info': 'mixup_pred', 'data': target_pred, 'resize': True},
                    {'data_type': 'label', 'info': 'mixup_label', 'data': mixed_lbl},
                ]

                if self.mic or self.mic_reg:
                    vis_data.extend([{'data_type': 'image', 'info': 'masked_image', 'data': masked_img}])
                    if self.mic:
                        vis_data.extend([{'data_type': 'logits', 'info': 'masked_image_pred', 'data': masked_target_pred, 'resize': True}])
                    else:
                        vis_data.extend([{'data_type': 'image', 'info': 'masked_vae_decoder_out', 'data': (masked_vae_decoder_output['after_vae.decoder'] + 1) / 2}])
                if self.denoise_supervise:
                    '''vis_data.extend([
                        {'data_type': 'logits', 'info': 'denoise_s_pred', 'data': denoise_s_pred, 'resize': True},
                        {'data_type': 'logits', 'info': 'denoise_t_pred', 'data': denoise_t_logits, 'resize': True},
                    ])'''
                    vis_data.extend([{
                        'data_type': 'image', 
                        'info': 'denoise_s_{}_decoder_out'.format(s_timestep[0]), 
                        'data': (denoise_s_vae_decoder_output['after_vae.decoder'] + 1) / 2
                    }, 
                    {'data_type': 'logits', 'info': 'denoise_seg_decoder_pred', 'data': denoise_ema_logits},])
                if 's' in self.vae_decoder_loss:
                    vis_data.extend([
                        {'data_type': 'image', 'info': 'source_vae_decoder_out', 'data': (source_vae_decoder_output['after_vae.decoder'] + 1) / 2},
                    ])
                if 't' in self.vae_decoder_loss:
                    vis_data.extend([
                        {'data_type': 'image', 'info': 'target_vae_decoder_out', 'data': (target_vae_decoder_output['after_vae.decoder'] + 1) / 2},
                    ])
                if self.reg_uncertain:
                    vis_data.extend([
                        {'data_type': 'logits', 'info': 'pl_reg', 'data': ema_softmax_reg},
                        {'data_type': 'heatmap', 'info': 'pl_prob_reg', 'data': pseudo_prob_reg},
                        {'data_type': 'heatmap', 'info': 'pl_prob_{:.3f}'.format(pseudo_val), 'data': pseudo_prob},
                    ])

                with torch.no_grad():
                    # vis "no_noise" teacher results if self.train_iter_index <= self.rev_noise_end_iter
                    if self.rev_noise_sup and self.train_iter_index <= self.rev_noise_end_iter:
                        if 's' in self.vae_decoder_loss:
                            no_noise_t_feats, no_noise_t_vae_decoder_output = self.backbone(
                                target_sec_modal.tensor, input_modal='others', return_unet_final_output=True
                            )
                            vis_data.extend([
                                {'data_type': 'image', 'info': 'no_noise_t_reg', 'data': (no_noise_t_vae_decoder_output['after_vae.decoder'] + 1) / 2}, 
                            ])
                        else:
                            no_noise_t_feats = self.backbone(
                                target_sec_modal.tensor, input_modal='others'
                            )
                        no_noise_t_ema_logits = self.ema_sem_seg_head(no_noise_t_feats)
                        no_noise_t_ema_logits = F.interpolate(no_noise_t_ema_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                        vis_data.extend([
                            {'data_type': 'logits', 'info': 'no_noise_t_pred', 'data': no_noise_t_ema_logits}
                        ])
                    # vis "add_noise" teacher results if self.train_iter_index > self.rev_noise_end_iter
                    elif self.rev_noise_sup and self.train_iter_index > self.rev_noise_end_iter:
                        noise_t = random.randint(self.denoise_timestep_range[0], self.denoise_timestep_range[1])
                        if self.rev_noise_gradually:
                            # noise_t = int(noise_t * (1 - self.train_iter_index / self.train_max_iter))
                            pass
                        noise_t = (noise_t, noise_t + 1)
                        if 's' in self.vae_decoder_loss:
                            noise_t_feats, noise_t_vae_decoder_output = self.backbone(
                                target_sec_modal.tensor, input_modal='others', return_unet_final_output=True, timestep=noise_t
                            )
                            vis_data.extend([
                                {'data_type': 'image', 'info': 'noise_{}_t_reg'.format(noise_t[0]), 'data': (noise_t_vae_decoder_output['after_vae.decoder'] + 1) / 2}, 
                            ])
                        else:
                            noise_t_feats = self.backbone(
                                target_sec_modal.tensor, input_modal='others', timestep=noise_t
                            )
                        noise_t_ema_logits = self.ema_sem_seg_head(noise_t_feats)
                        noise_t_ema_logits = F.interpolate(noise_t_ema_logits, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
                        vis_data.extend([
                            {'data_type': 'logits', 'info': 'noise_{}_t_pred'.format(noise_t[0]), 'data': noise_t_ema_logits}
                        ])

                if self.noise_reg is not None:
                    vis_data.extend([
                        {'data_type': 'image', 'info': 'noise_reg_vae_decoder_out', 'data': (noise_reg_vae_decoder_output['after_vae.decoder'] + 1) / 2},
                        {'data_type': 'logits', 'info': 'noise_reg_{}_pl'.format(noise_reg_t[0]), 'data': noise_reg_logits}
                    ])

                self.vis_results(vis_data=vis_data, save_path=self.output_dir, iter_index=self.train_iter_index)
                self.logger.info("Visualization the results for [{}] iteration:".format(self.train_iter_index))
            if self.add_zero_grad:
                losses['zero_grad'] = self.add_zero_gead_on_unused_lora(target_modal_type)
            return losses
        else:
            assert len(batched_inputs) == 1

            # close multi-target adaption
            # if len(self.lora_names) == 1:
            assert not 'modality_type' in batched_inputs[0].keys()
            target_modal_type = self.target_modality
            # else:
            #     target_modal_type = batched_inputs[0]['modality_type']

            target_sec_modal = [x['target_second_modality'].to(self.device) for x in batched_inputs]
            target_sec_modal = [(x - self.pixel_mean) / self.pixel_std for x in target_sec_modal]
            ori_size = target_sec_modal[0].shape[1:]
            target_sec_modal = ImageList.from_tensors(target_sec_modal, self.size_divisibility)
            
            self.set_lora_adapter(state=target_modal_type)

            test_input_dict = {
                'input_modal':'others',
            }
            if self.mask_diff is not None:
                modality_mask = torch.zeros_like(target_sec_modal.tensor)[:, 0:1] + self.mask_val[target_modal_type]
                test_input_dict['modality_mask'] = F.interpolate(modality_mask, scale_factor=1/8., mode='nearest')

            if self.eval_with_noise is not None:
                test_input_dict['timestep'] = (self.eval_with_noise, self.eval_with_noise + 1)

            backbone_feats = self.backbone(target_sec_modal.tensor, **test_input_dict)
            outputs = self.sem_seg_head_sec_modal(backbone_feats)

            outputs = F.interpolate(outputs, size=target_sec_modal.tensor.shape[2:], mode='bilinear', align_corners=False)
            outputs = outputs[:, :, :ori_size[0], :ori_size[1]]
            # print('test_input:', target_sec_modal.tensor.shape)  # torch.Size([1, 3, 448, 640])
            # print('test_output:', outputs.shape)  # torch.Size([1, 19, 112, 160])
            return [{'sem_seg': outputs}]

