from typing import Optional, Tuple

import torch
from torch import nn
from dataclasses import dataclass
from transformers import CLIPTextConfig

from .neti_mapper import NeTIMapper


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float


@dataclass
class NeTIBatch:
    input_ids: torch.Tensor
    placeholder_token_id: int
    timesteps: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None


class NeTICLIPTextEmbeddings(nn.Module):
    """ Modification of CLIPTextEmbedding to allow for the use of a NeTIMapper to overwrite the concept token. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def set_mapper(self, mapper: NeTIMapper):
        self.mapper = mapper

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                batch: Optional[NeTIBatch] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if batch is not None:
            input_ids = batch.input_ids

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]  # torch.Size([1, 77])

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)  # [2, 77] --> [2, 77, 768]

        ####################################################################
        # NeTI logic - Use mapper to overwrite the learnable token embedding
        ####################################################################
        bypass_outputs = None
        if batch is not None:
            mapper_outputs = self.mapper(timestep=batch.timesteps.float(),
                                         unet_layer=batch.unet_layers.float(),
                                         truncation_idx=batch.truncation_idx)
            mapper_outputs = mapper_outputs.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)  # torch.Size([2, 1536])
            if self.mapper.output_bypass:
                bypass_outputs = mapper_outputs[:, mapper_outputs.shape[1] // 2:]  # torch.Size([2, 768])
                mapper_outputs = mapper_outputs[:, :mapper_outputs.shape[1] // 2]  # torch.Size([2, 768])

            # Overwrite the index of the placeholder token with the mapper output for each entry in the batch
            learnable_idxs = (input_ids == batch.placeholder_token_id).nonzero(as_tuple=True)[1]  # tensor([7, 8])
            inputs_embeds[torch.arange(input_ids.shape[0]), learnable_idxs] = mapper_outputs  # torch.Size([2, 77, 768])
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings  # torch.Size([2, 77, 768])
        return embeddings, bypass_outputs

    def forward_wo_neti(self, input_ids, position_ids=None, inputs_embeds=None):
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]  # torch.Size([1, 77])
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)  # [2, 77] --> [2, 77, 768]
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings  # torch.Size([2, 77, 768])
        return embeddings