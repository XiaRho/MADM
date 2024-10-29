import os
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from modeling.neti import NeTICLIPTextModel

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from transformers import CLIPTokenizer


class LdmDiffusers(nn.Module):
    latent_image_size = (64, 64)
    text_embed_shape = torch.Size([77, 768])
    unet_time_embed_out_features = 1280
    uncond_inputs_size = torch.Size([1, 77, 768])
    feature_size = (512, 512)
    feature_dims = [512, 512, 2560, 1920, 960, 640, 512, 512]
    feature_strides = [4, 8, 64, 32, 16, 8, 8, 4]
    num_groups = 8
    grouped_indices = [[0], [1], [2], [3], [4], [5], [6], [7]]
    timesteps = 0
    unet_layers = ['IN01', 'IN02', 'IN04', 'IN05', 'IN07', 'IN08', 'MID',
                   'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'OUT09', 'OUT10', 'OUT11']
    dtype=torch.float16
    input_mean = 0.5
    input_std = 0.5

    def __init__(self, stable_diffusion_name_or_path, encoder_block_indices, unet_block_indices,
                 decoder_block_indices, input_range='01', unet_block_indices_type='in', finetune_unet='no',
                 concat_pixel_shuffle=False, add_latent_noise=-1, norm_latent_noise=False, vae_decoder_loss=False, input_channel_plus=0, 
                 final_fuse_vae_decoder_feat=False):
        super().__init__()

        self.stable_diffusion_name_or_path = os.path.expanduser(stable_diffusion_name_or_path)
        self.encoder_block_indices = encoder_block_indices
        self.unet_block_indices = unet_block_indices
        self.decoder_block_indices = decoder_block_indices
        self.input_range = input_range
        assert self.input_range in {'01', '-1+1'}
        self.unet_block_indices_type = unet_block_indices_type
        assert self.unet_block_indices_type in {'in', 'after'}
        self.finetune_unet = finetune_unet
        assert self.finetune_unet in {'no', 'all', 'attention', 'without cross-attention'}
        self.concat_pixel_shuffle = concat_pixel_shuffle
        self.add_latent_noise = add_latent_noise
        self.norm_latent_noise = norm_latent_noise

        self.vae = _init_vae(self.stable_diffusion_name_or_path).cuda()
        self.unet = _init_unet(self.stable_diffusion_name_or_path, finetune_unet=self.finetune_unet).cuda()
        self.noise_scheduler = _init_noise_scheduler(self.stable_diffusion_name_or_path)
        self.tokenizer = _init_tokenizer(self.stable_diffusion_name_or_path)
        self.text_encoder = _init_text_encoder(self.stable_diffusion_name_or_path).cuda()

        self.input_channel_plus = input_channel_plus
        if self.input_channel_plus != 0:
            weights = self.unet.conv_in.weight.clone() * (4. / 4 + self.input_channel_plus)  # 4 channel --> 5 channel
            bias = self.unet.conv_in.bias.clone()
            self.unet.conv_in = torch.nn.Conv2d(
                self.unet.conv_in.in_channels + self.input_channel_plus, self.unet.conv_in.out_channels, kernel_size=self.unet.conv_in.kernel_size, padding=self.unet.conv_in.padding
            ).to(self.unet.device)
            with torch.no_grad():
                self.unet.conv_in.weight[:, : 4] = weights
                # self.unet.conv_in.weight[:, 4:] = torch.mean(weights, dim=1, keepdim=True).
                self.unet.conv_in.weight[:, 4:] = weights[:, -self.input_channel_plus:]
                self.unet.conv_in.bias.data = bias.data

        rng = torch.Generator().manual_seed(42)
        self.register_buffer("shared_noise", torch.randn(1, self.vae.latent_channels, *self.latent_image_size,
                                                         generator=rng).detach().cuda())
        self.register_buffer("uncond_inputs", self._get_uncond_inputs('').detach().cuda())
        self._freeze()
        
        # save unet final output for 
        self.vae_decoder_loss = vae_decoder_loss
        self.final_fuse_vae_decoder_feat = final_fuse_vae_decoder_feat

        if self.concat_pixel_shuffle:
            self.pixel_unshuffle_layer = nn.Sequential(
                nn.PixelUnshuffle(downscale_factor=8),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
            )
            weights = self.unet.conv_in.weight.clone()
            self.unet.conv_in = torch.nn.Conv2d(
                self.unet.conv_in.in_channels + 64, self.unet.conv_in.out_channels, 
                kernel_size=self.unet.conv_in.kernel_size, padding=self.unet.conv_in.padding
            )
            with torch.no_grad():
                for i in range(0, 68, 4):
                    self.unet.conv_in.weight[:, i: i + 4] = weights / 17.0

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False
        if self.finetune_unet != 'no':
            if self.finetune_unet == 'all':
                for p in self.unet.parameters():
                    p.requires_grad = True
            elif self.finetune_unet == 'without cross-attention':
                for name, p in self.unet.named_parameters():
                    if 'attentions' in name and 'attn2' in name:
                        if 'to_v' in name:
                            assert self.unet.state_dict()[name].shape[1] == 768
                    else:
                        p.requires_grad = True
            else:
                assert self.finetune_unet == 'attention'
                for name, p in self.unet.named_parameters():
                    if 'attentions' in name:
                        p.requires_grad = True
            self.exclude_unused_params()

    def exclude_unused_params(self):
        # img = torch.rand((1, 3, 512, 512)).cuda().half()
        with torch.cuda.amp.autocast():
            # self.forward(batched_inputs={'img': img, 'cond_inputs': self.uncond_inputs, 'cond_emb': None})
            noisy_latents = torch.rand((1, self.unet.conv_in.in_channels, 64, 64)).cuda().half()
            timesteps = torch.randint(low=0, high=1, size=(1,), device=noisy_latents.device).long()
            _hs = torch.rand((1, 77, 768)).cuda().half()  # torch.Size([2, 77, 768])  77 can be changed to other max_length
            # noisy_latents: [2, 4, 64, 64] tensor(-6.4637, device='cuda:0')~tensor(4.7898, device='cuda:0')
            _, unet_features = diffusion_unet(unet=self.unet, sample=noisy_latents, timestep=timesteps,
                                              encoder_hidden_states=_hs, res_time_embedding=None,
                                              unet_block_indices=self.unet_block_indices, 
                                              unet_block_indices_type=self.unet_block_indices_type)
        loss = F.mse_loss(input=unet_features[-1], target=torch.ones_like(unet_features[-1]))
        loss.backward()

        for p in self.unet.parameters():
            if p.grad is None:
                p.requires_grad = False
        self.unet.zero_grad()

    def forward(self, batched_inputs, input_modal, **kwargs):
        images = batched_inputs['img']  # [1, 3, 512, 512]
        if self.input_range == '-1+1':
            images = (images - self.input_mean) / self.input_std
            assert -1 <= torch.min(images) and torch.max(images) <= 1
        text_prompt = batched_inputs['cond_inputs']  # [1, 77, 768]
        res_time_embedding = batched_inputs['cond_emb']  # [1, 1, 1280]

        latents, encoder_features = vae_encoder(vae=self.vae, images=images,
                                                encoder_block_indices=self.encoder_block_indices)
        
        # Sample noise that we'll add to the latents
        bsz = latents.shape[0]
        if 'timestep' in batched_inputs.keys():
            low_timestep, high_timestep = batched_inputs['timestep'][0], batched_inputs['timestep'][1]
        else:
            low_timestep, high_timestep = 0, 1
        timesteps = torch.randint(low=low_timestep, high=high_timestep, size=(bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = add_noise(noise_scheduler=self.noise_scheduler, latents=latents,
                                  timesteps=timesteps, shared_noise=self.shared_noise)  # [1, 4, 64, 64]

        if self.add_latent_noise != -1 and input_modal == 'mixed':
            noisy_latents += torch.randn_like(noisy_latents) * self.add_latent_noise
        if self.norm_latent_noise:
            noisy_latents = (noisy_latents - torch.mean(noisy_latents)) / torch.std(noisy_latents)

        if self.concat_pixel_shuffle:
            down_sample_image = torch.mean(images, dim=1, keepdim=True)  # [1, 1, 512, 512]
            down_sample_image = self.pixel_unshuffle_layer(down_sample_image)  # [1, 64, 64, 64]
            noisy_latents = torch.concat((noisy_latents, down_sample_image), dim=1)  # [1, 68, 64, 64]

        # add modality mask to noisy_latents
        if self.input_channel_plus != 0:
            assert 'modality_mask' in kwargs.keys()
            noisy_latents = torch.concatenate((noisy_latents, kwargs['modality_mask']), dim=1)

        _hs = text_prompt  # torch.Size([2, 77, 768])  77 can be changed to other max_length
        # noisy_latents: [2, 4, 64, 64] tensor(-6.4637, device='cuda:0')~tensor(4.7898, device='cuda:0')
        if 'ema_forward' in kwargs.keys() and kwargs['ema_forward'] and hasattr(self, 'ema_unet'):
            forward_unet = self.ema_unet
        else:
            forward_unet = self.unet
        unet_final_output, unet_features = diffusion_unet(unet=forward_unet, sample=noisy_latents, timestep=timesteps,
                                                          encoder_hidden_states=_hs, res_time_embedding=res_time_embedding,
                                                          unet_block_indices=self.unet_block_indices, 
                                                          unet_block_indices_type=self.unet_block_indices_type)
        
        # cal loss on vae.decoder
        if self.vae_decoder_loss:
            # self.unet_final_output['before_vae.decoder'] = unet_final_output.sample
            decoder_output, _ = vae_decoder(vae=self.vae, latents=unet_final_output.sample, decoder_block_indices=[], output_final=True)
            # self.unet_final_output['after_vae.decoder'] = torch.clip(decoder_output, min=-1.0, max=1.0)  # [1, 3, 512, 512]
            if self.final_fuse_vae_decoder_feat:
                decoder_features = [decoder_output.detach()]
            else:
                assert len(self.encoder_block_indices) == 0
                encoder_features = [decoder_output.detach()]
                decoder_features = []
        else:
            if len(self.decoder_block_indices) != 0:
                _, decoder_features = vae_decoder(vae=self.vae, latents=latents,
                                                decoder_block_indices=self.decoder_block_indices)
            else:
                decoder_features = []

        if "return_unet_feats" in batched_inputs.keys() and batched_inputs["return_unet_feats"]:
            return [*encoder_features, *unet_features, *decoder_features], unet_features
        elif "return_unet_final_output" in kwargs.keys() and kwargs["return_unet_final_output"]:
            return [*encoder_features, *unet_features, *decoder_features], {
                'before_vae.decoder': unet_final_output.sample, 
                'after_vae.decoder': torch.clip(decoder_output, min=-1.0, max=1.0)
            }
        else:
            return [*encoder_features, *unet_features, *decoder_features]

    def _get_uncond_inputs(self, text):
        empty_token = self.tokenizer(text, padding="max_length", truncation=True,
                                     max_length=self.tokenizer.model_max_length,
                                     return_tensors="pt", ).input_ids.cuda()  # torch.Size([1, 77])

        embed_token = self.text_encoder.text_model.embeddings.token_embedding(empty_token)  # torch.Size([1, 77, 768])
        embed_token_pos = self.text_encoder.text_model.embeddings.position_embedding(
            self.text_encoder.text_model.embeddings.position_ids[:, :embed_token.shape[1]]
        )  # torch.Size([1, 77, 768])
        embed_token = embed_token + embed_token_pos  # torch.Size([1, 77, 768])

        bsz, seq_len = empty_token.shape
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, embed_token.dtype).to(embed_token.device)

        uncond_inputs = self.text_encoder.text_model.encoder(
            inputs_embeds=embed_token,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=self.text_encoder.text_model.config.output_attentions,
            output_hidden_states=self.text_encoder.text_model.config.output_hidden_states,
            return_dict=self.text_encoder.text_model.config.use_return_dict
        ).last_hidden_state

        uncond_inputs = self.text_encoder.text_model.final_layer_norm(uncond_inputs)
        return uncond_inputs

    
def _init_vae(pretrained_model_name_or_path) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, torch_dtype=torch.float16)
    return vae


def _init_unet(pretrained_model_name_or_path, finetune_unet=False) -> UNet2DConditionModel:
    torch_type = torch.float32 if finetune_unet != 'no' else torch.float16
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=None, torch_dtype=torch_type
    )#.cuda()
    # unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    return unet
# from ldm.modules.diffusionmodules.openaimodel import UNetModel


def _init_noise_scheduler(pretrained_model_name_or_path) -> DDPMScheduler:
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=torch.float16
    )
    return noise_scheduler


def _init_tokenizer(pretrained_model_name_or_path) -> CLIPTokenizer:
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=torch.float16
    )
    return tokenizer


def _init_text_encoder(pretrained_model_name_or_path) -> NeTICLIPTextModel:
    text_encoder = NeTICLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None,
    )
    return text_encoder


@torch.no_grad()
def vae_encoder(vae, images, encoder_block_indices):
    index = 0
    features = []
    x = vae.encoder.conv_in(images)  # [1, 3, 512, 512] -> [1, 128, 512, 512]
    for down_block in vae.encoder.down_blocks:
        for resnet in down_block.resnets:
            x = resnet(x, temb=None)
            index += 1
            if index in encoder_block_indices:
                features.append(x)
        if down_block.downsamplers is not None:
            for downsampler in down_block.downsamplers:
                x = downsampler(x)
    x = vae.encoder.mid_block(x)
    # post-process
    x = vae.encoder.conv_norm_out(x)
    x = vae.encoder.conv_act(x)
    x = vae.encoder.conv_out(x)

    moments = vae.quant_conv(x)
    posterior = DiagonalGaussianDistribution(moments)

    # NOTE: make encode process deterministic, we use mean instead of sample from posterior
    # latents = posterior.sample().detach() * vae.config.scaling_factor
    latents = posterior.mean * vae.config.scaling_factor

    assert len(encoder_block_indices) == len(features)
    return latents, features


@torch.no_grad()
def vae_decoder(vae, latents, decoder_block_indices, output_final=False):
    index = 0
    features = []

    latents = 1.0 / vae.config.scaling_factor * latents
    sample = vae.post_quant_conv(latents)

    sample = vae.decoder.conv_in(sample)

    # middle
    sample = vae.decoder.mid_block(sample)

    # up
    for up_block in vae.decoder.up_blocks:
        for resnet in up_block.resnets:
            if index in decoder_block_indices:
                features.append(sample)
            index += 1
            sample = resnet(sample, temb=None)
        if up_block.upsamplers is not None:
            for upsampler in up_block.upsamplers:
                sample = upsampler(sample)

    # post-process
    if output_final:
        sample = vae.decoder.conv_norm_out(sample)
        sample = vae.decoder.conv_act(sample)
        sample = vae.decoder.conv_out(sample)
    else:
        sample = None

    return sample, features


def add_noise(noise_scheduler, latents, timesteps, shared_noise=None):
    if shared_noise is not None:
        if shared_noise.shape[2:] != latents.shape[2:]:
            shared_noise = F.interpolate(shared_noise, size=latents.shape[2:],
                                         mode="bicubic", align_corners=False)
        else:
            shared_noise = shared_noise
        noise = shared_noise.expand_as(latents)
    else:
        noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents


def diffusion_upblock2d(upblock, hidden_states, res_hidden_states_tuple, temb,
                        upsample_size, start_index, unet_block_indices, unet_block_indices_type):
    features = []
    for resnet in upblock.resnets:
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if unet_block_indices_type == 'in':
            if start_index in unet_block_indices:
                features.append(hidden_states)
            start_index += 1

        if upblock.training and upblock.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
        else:
            hidden_states = resnet(hidden_states, temb)

        if unet_block_indices_type == 'after':
            if start_index in unet_block_indices:
                features.append(hidden_states)
            start_index += 1

    if upblock.upsamplers is not None:
        for upsampler in upblock.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states, features, start_index


def diffusion_cross_attn_upblock2d(upblock, hidden_states, temb, res_hidden_states_tuple, encoder_hidden_states, cross_attention_kwargs, 
                                   upsample_size, attention_mask, start_index, unet_block_indices, unet_block_indices_type):
    features = []
    # TODO(Patrick, William) - attention mask is not used
    for resnet, attn in zip(upblock.resnets, upblock.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if unet_block_indices_type == 'in':
            if start_index in unet_block_indices:
                features.append(hidden_states)
            start_index += 1

        if upblock.training and upblock.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(attn, return_dict=False),
                hidden_states,
                encoder_hidden_states,
                cross_attention_kwargs,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

        if unet_block_indices_type == 'after':
            if start_index in unet_block_indices:
                features.append(hidden_states)
            start_index += 1

    if upblock.upsamplers is not None:
        for upsampler in upblock.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states, features, start_index


def diffusion_unet(unet, sample, timestep, encoder_hidden_states, res_time_embedding, unet_block_indices, unet_block_indices_type):
    class_labels = None
    timestep_cond = None
    attention_mask = None
    cross_attention_kwargs = None
    down_block_additional_residuals = None
    mid_block_additional_residual = None
    return_dict = True

    default_overall_up_factor = 2**unet.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        forward_upsample_size = True

    # prepare attention_mask
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if unet.config.center_input_sample:  # False
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = unet.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=unet.dtype)

    emb = unet.time_embedding(t_emb, timestep_cond)
    if res_time_embedding is not None:
        if res_time_embedding.shape[1] == 1:
            res_time_embedding = res_time_embedding[:, 0]
        emb += res_time_embedding

    if unet.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if unet.config.class_embed_type == "timestep":
            class_labels = unet.time_proj(class_labels)

        class_emb = unet.class_embedding(class_labels).to(dtype=unet.dtype)
        emb = emb + class_emb

    # 2. pre-process
    sample = unet.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample += down_block_additional_residual
            new_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if unet.mid_block is not None:
        sample = unet.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

    if mid_block_additional_residual is not None:
        sample += mid_block_additional_residual

    # 5. up
    up_block_index = 0
    unet_features = []
    for i, upsample_block in enumerate(unet.up_blocks):
        is_final_block = i == len(unet.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample, features, up_block_index = diffusion_cross_attn_upblock2d(
                upsample_block, 
                hidden_states=sample, 
                temb=emb, 
                res_hidden_states_tuple=res_samples, 
                encoder_hidden_states=encoder_hidden_states, 
                cross_attention_kwargs=cross_attention_kwargs, 
                upsample_size=upsample_size, 
                attention_mask=attention_mask, 
                start_index=up_block_index, 
                unet_block_indices=unet_block_indices,
                unet_block_indices_type=unet_block_indices_type,
            )
        else:
            sample, features, up_block_index = diffusion_upblock2d(
                upsample_block, 
                hidden_states=sample, 
                temb=emb, 
                res_hidden_states_tuple=res_samples, 
                upsample_size=upsample_size, 
                start_index=up_block_index, 
                unet_block_indices=unet_block_indices,
                unet_block_indices_type=unet_block_indices_type,
            )
        
        unet_features.extend(features)

    assert len(unet_features) == len(unet_block_indices)
    # 6. post-process
    if unet.conv_norm_out:
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample), unet_features
