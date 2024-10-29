import logging
from timm.models.layers import trunc_normal_
from copy import deepcopy
import torch
import torch.nn as nn
from modeling.meta_arch.ldm import LdmExtractor, TextAdapter, PositionalLinear


class TextPromptExtractor(nn.Module):
	def __init__(
			self,
			source_text='',
			target_text='',
			mixup_text='',
			prompt_classes=None,
			gamma_init_value=1e-4,
			learnable_time_embed=True,
			num_timesteps=1,
			**kwargs,
	):
		super().__init__()

		self.ldm_extractor = LdmExtractor(**kwargs)
		self.text_embed_shape = self.ldm_extractor.ldm.embed_text([""]).shape[1:]
		time_embed_project_out_feats = self.ldm_extractor.ldm.unet.time_embed[-1].out_features
		self.uncond_inputs = self.ldm_extractor.ldm.uncond_inputs.cuda()
		self.gamma_init_value = gamma_init_value
		self.prompt_classes = prompt_classes
		self.learnable_time_embed = learnable_time_embed

		if prompt_classes is not None:
			classes_str = ''
			for i, class_name in enumerate(prompt_classes):
				if i == len(prompt_classes) - 1:
					classes_str += ', and {}'.format(class_name)
				else:
					classes_str += ', {}'.format(class_name)
			classes_str = classes_str[2:].lower()
			self.source_text = source_text.format(classes_str)
			self.target_text = target_text.format(classes_str)
			self.mixup_text = mixup_text.format(classes_str)

		with torch.no_grad():
			# self.source_prompt = self.ldm_extractor.ldm.embed_text([self.source_text])
			# self.target_prompt = self.ldm_extractor.ldm.embed_text([self.target_text])
			# self.mixup_prompt = self.ldm_extractor.ldm.embed_text([self.mixup_text])
			self.register_buffer("source_prompt", self.ldm_extractor.ldm.embed_text([self.source_text]))
			self.register_buffer("target_prompt", self.ldm_extractor.ldm.embed_text([self.target_text]))
			self.register_buffer("mixup_prompt", self.ldm_extractor.ldm.embed_text([self.mixup_text]))

		# ################################################
		# ###### learnable parms for SD model (RGB) ######
		# ################################################
		self.clip_project_rgb = TextAdapter(gamma_init_value=self.gamma_init_value)
		if self.learnable_time_embed:
			self.time_embed_project_rgb = nn.Parameter(torch.zeros(1, num_timesteps, time_embed_project_out_feats))
			trunc_normal_(self.time_embed_project_rgb, std=0.02)

		# ############################################################
		# ###### learnable parms for SD model (second modality) ######
		# ############################################################
		self.clip_project_others = TextAdapter(gamma_init_value=self.gamma_init_value)
		if self.learnable_time_embed:
			self.time_embed_project_others = nn.Parameter(torch.zeros(1, num_timesteps, time_embed_project_out_feats))
			trunc_normal_(self.time_embed_project_others, std=0.02)

	@property
	def feature_size(self):
		return self.ldm_extractor.feature_size

	@property
	def feature_dims(self):
		return self.ldm_extractor.feature_dims

	@property
	def feature_strides(self):
		return self.ldm_extractor.feature_strides

	@property
	def num_groups(self) -> int:

		return self.ldm_extractor.num_groups

	@property
	def grouped_indices(self):

		return self.ldm_extractor.grouped_indices

	def extra_repr(self):
		return f"learnable_time_embed={self.learnable_time_embed}"

	def forward(self, batched_inputs, input_modal, ema_forward=False):
		"""
		Args:
			:param batched_inputs: (dict) expected keys: "img", Optional["caption"]
			:param input_modal: (str) 'rgb' or 'others',
			:param ema_forward: (bool),
		"""
		assert input_modal in {'rgb', 'others'}

		# image = batched_inputs["img"]  # [batch_size, 3, 512, 512]  0~1

		if input_modal == 'rgb':
			assert ema_forward is False
			batched_inputs["cond_inputs"] = self.clip_project_rgb(self.source_prompt)
			if self.learnable_time_embed:
				batched_inputs["cond_emb"] = self.time_embed_project_rgb
		else:
			if ema_forward:
				batched_inputs["cond_inputs"] = self.ema_clip_project_others(self.target_prompt)
			else:
				batched_inputs["cond_inputs"] = self.clip_project_others(self.mixup_prompt)
			if self.learnable_time_embed:
				batched_inputs["cond_emb"] = self.time_embed_project_others

		self.set_requires_grad(self.training)
		return self.ldm_extractor(batched_inputs)

	def set_requires_grad(self, requires_grad):
		parameters = self.ldm_extractor.ldm.ldm.model if hasattr(self.ldm_extractor, 'ldm') else self.ldm_extractor.unet
		for p in parameters.parameters():
			p.requires_grad = requires_grad
