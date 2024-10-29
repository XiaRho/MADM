import logging
import math
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Union
import torch
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as T
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.structures import ImageList
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from ..meta_arch.helper import FeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractorBackbone(Backbone):
    """Backbone implement following for FeatureExtractor

    1. Project same group features into the one single feature map
    2. Sort the features by area, from large to small
    3. Get the stride of each feature map
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        out_features: List[str],
        backbone_in_size: Union[int, Tuple[int]] = (512, 512),
        min_stride: int = 4,
        max_stride: int = 32,
        projection_dim: int = 512,
        num_res_blocks: int = 1,
        use_checkpoint: bool = False,
        slide_training: bool = False,
        slide_inference: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.use_checkpoint = use_checkpoint

        if isinstance(projection_dim, int):
            self.feature_projections = nn.ModuleList()
            for feature_dim in self.feature_extractor.feature_dims:  # [512, 512, 2560, 1920, 960, 640, 512, 512]
                self.feature_projections.append(
                    nn.Sequential(
                        *ResNet.make_stage(
                            BottleneckBlock,
                            num_blocks=num_res_blocks,
                            in_channels=feature_dim,
                            bottleneck_channels=projection_dim // 4,
                            out_channels=projection_dim,
                            norm="GN",
                        )
                    )
                )
        self._slide_inference = slide_inference
        # if isinstance(backbone_in_size, int):
        #     self.image_preprocess = T.Resize(
        #         size=backbone_in_size, max_size=1280, interpolation=T.InterpolationMode.BICUBIC
        #     )
        #     self.backbone_in_size = (backbone_in_size, backbone_in_size)
        #     self._slide_inference = False
        # else:
        #     self.image_preprocess = T.Resize(
        #         size=tuple(backbone_in_size), interpolation=T.InterpolationMode.BICUBIC
        #     )
        #     self.backbone_in_size = tuple(backbone_in_size)
        #     self._slide_inference = True
        if self._slide_inference:
            self.image_preprocess = None
            self.y1_y2_x1_x2 = [(0, 512, 0, 512), (0, 512, 256, 768), (0, 512, 512, 1024)]
        else:
            self.image_preprocess = T.Resize(
                size=tuple(backbone_in_size), interpolation=T.InterpolationMode.BILINEAR # T.InterpolationMode.BICUBIC  
            )
        self.backbone_in_size = tuple(backbone_in_size)

        self._slide_training = slide_training
        if self._slide_training:
            assert self._slide_inference, "slide training must be used with slide inference"

        self.min_stride = min_stride
        self.max_stride = max_stride

        idx_to_stride = {}
        stride_to_indices = defaultdict(list)
        for indices in self.feature_extractor.grouped_indices:
            for idx in indices:
                stride = self.feature_extractor.feature_strides[idx]
                stride = min(max(stride, self.min_stride), self.max_stride)
                idx_to_stride[idx] = stride
                stride_to_indices[stride].append(idx)

        self._sorted_grouped_indices = []
        for stride in sorted(stride_to_indices.keys()):
            self._sorted_grouped_indices.append(stride_to_indices[stride])

        self._out_feature_channels = {}
        self._out_feature_strides = {}

        for indices in self._sorted_grouped_indices:
            stride = idx_to_stride[indices[0]]
            name = f"s{int(math.log2(stride))}"
            if name not in out_features:
                continue
            assert name not in self._out_feature_strides, f"Duplicate feature name {name}"
            self._out_feature_strides[name] = stride
            self._out_feature_channels[name] = projection_dim
        self._out_features = list(self._out_feature_strides.keys())

        logger.info(
            f"backbone_in_size: {backbone_in_size}, "
            f"slide_training: {self._slide_training}, \n"
            f"slide_inference: {self._slide_inference}, \n"
            f"min_stride: {min_stride}, "
            f"max_stride: {max_stride}, \n"
            f"projection_dim: {projection_dim}, \n"
            f"out_feature_channels: {self._out_feature_channels}\n"
            f"out_feature_strides: {self._out_feature_strides}\n"
            f"use_checkpoint: {use_checkpoint}\n"
        )

    @property
    def size_divisibility(self) -> int:
        return 64

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination

    def preprocess_image(self, img):
        if self.image_preprocess is not None:
            img = self.image_preprocess(img)
        # print("processed_image_size:", img.shape)
        img = ImageList.from_tensors(list(img), self.size_divisibility).tensor
        # print("padded size:", img.shape)
        return img
    
    def checkpoint_forward_features(self, features, input_image_size, ema_forward=False):
        if self.use_checkpoint:
            return checkpoint.checkpoint(
                self.forward_features, features, input_image_size, ema_forward, use_reentrant=False
            )
        else:
            return self.forward_features(features, input_image_size, ema_forward)

    def single_forward(self, img, input_modal='rgb', ema_forward=False, timestep=None, **kwargs):

        # save memory
        input_image_size = img.shape[-2:]
        # print("input_image_size:", img.shape)
        img = self.preprocess_image(img)
        
        features = self.feature_extractor(dict(img=img), input_modal, ema_forward, timestep, **kwargs)
        if 'return_unet_final_output' in kwargs.keys():
            forward_features = self.checkpoint_forward_features(features[0], input_image_size, ema_forward)
            return forward_features, features[1]
        else:
            forward_features = self.checkpoint_forward_features(features, input_image_size, ema_forward)
            return forward_features
        # return self.checkpoint_forward_features(features, input_image_size, ema_forward)


    def forward_features(self, features, input_image_size, ema_forward=False):
        output_features = {}
        for name, indices in zip(self._out_features, self._sorted_grouped_indices):
            output_feature = None
            stride = self._out_feature_strides[name]
            for idx in indices:
                # print("before restore", name, idx, features[idx].shape, stride)
                # restore aspect ratio
                restored_feature = F.interpolate(
                    features[idx],
                    size=(input_image_size[-2] // stride, input_image_size[-1] // stride),
                )
                if ema_forward:
                    projected_feature = self.ema_feature_projections[idx](restored_feature)
                else:
                    projected_feature = self.feature_projections[idx](restored_feature)
                if output_feature is None:
                    output_feature = projected_feature
                else:
                    output_feature = output_feature + projected_feature
            output_features[name] = output_feature

        # for k in output_features:
        #     print(k, output_features[k].shape)
        return output_features

    def slide_forward(self, img, input_modal='rgb', ema_forward=False, timestep=None, **kwargs):

        batch_size, _, h_img, w_img = img.shape
        # output_features = {k: torch.zeros_like(v) for k, v in self.single_forward(img).items()}

        # [('s2', torch.Size([1, 512, 128, 256])), ('s3', torch.Size([1, 512, 64, 128])), 
        #  ('s4', torch.Size([1, 512, 32, 64])), ('s5', torch.Size([1, 512, 16, 32]))]
        output_features = {}
        for k in self._out_features:
            stride = self._out_feature_strides[k]
            channel = self._out_feature_channels[k]
            output_features[k] = torch.zeros(
                (batch_size, channel, h_img // stride, w_img // stride),
                dtype=img.dtype,
                device=img.device,
            )

        count_mats = {k: torch.zeros_like(v) for k, v in output_features.items()}

        if self._slide_training:
            short_side = min(min(self.backbone_in_size), min(img.shape[-2:]))
        else:
            # if not slide training then use the shorter side to crop
            short_side = min(img.shape[-2:])  # 512

        # h_img, w_img = img.shape[-2:]

        h_crop = w_crop = short_side

        h_stride = w_stride = short_side

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1  # 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1  # 2
        w_grids += 1
        assert h_grids == 1 and w_grids == 3

        # print("img.shape:", img.shape)
        # for k in output_features:
        #     print(k, output_features[k].shape)
        # print("h_grids:", h_grids, "w_grids:", w_grids)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                if self.y1_y2_x1_x2 is None:
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                else:
                    y1, y2, x1, x2 = self.y1_y2_x1_x2[w_idx]
                # (0, 512, 0, 512), (0, 512, 512, 1024)
                crop_img = img[:, :, y1:y2, x1:x2]
                assert crop_img.shape[-2:] == (h_crop, w_crop), f"{crop_img.shape} from {img.shape}"
                # print("crop_img.shape:", crop_img.shape)
                crop_features = self.single_forward(crop_img, input_modal, ema_forward, timestep, **kwargs)['output_features']
                for k in crop_features:
                    k_x1 = x1 // self._out_feature_strides[k]
                    k_x2 = x2 // self._out_feature_strides[k]
                    k_y1 = y1 // self._out_feature_strides[k]
                    k_y2 = y2 // self._out_feature_strides[k]
                    # output_features[k] += F.pad(
                    #     crop_features[k],
                    #     (
                    #         k_x1,
                    #         output_features[k].shape[-1] - k_x1 - crop_features[k].shape[-1],
                    #         k_y1,
                    #         output_features[k].shape[-2] - k_y1 - crop_features[k].shape[-2],
                    #     ),
                    # )
                    # this version should save some memory
                    output_features[k][:, :, k_y1:k_y2, k_x1:k_x2] += crop_features[k]
                    count_mats[k][..., k_y1:k_y2, k_x1:k_x2] += 1
        assert all((count_mats[k] == 0).sum() == 0 for k in count_mats)

        for k in output_features:
            output_features[k] /= count_mats[k]

        # return output_features
        return {'output_features': output_features}

    def forward(self, img, input_modal='rgb', ema_forward=False, timestep=None, **kwargs):
        if (self.training and not self._slide_training) or not self._slide_inference:
            return self.single_forward(img, input_modal, ema_forward, timestep, **kwargs)
        else:
            return self.slide_forward(img, input_modal, ema_forward, timestep, **kwargs)


class AttentionFeatureExtractorBackbone(FeatureExtractorBackbone):
    def __init__(
        self,
        attention_features_res,  # {16, 32, 64}
        feature_dims,  # [320, 640, 1280]
        attention_features_location,  # ["up", "down", "mid"]

        target_attention_loss=False,
        attention_select_index=None,

        # super init parms
        feature_extractor: FeatureExtractor = None,
        out_features: List[str] = None,
        backbone_in_size: Union[int, Tuple[int]] = (512, 512),
        min_stride: int = 4,
        max_stride: int = 32,
        projection_dim: List[int] = [512, 512, 512, 512],
        bottleneck_channels: int = 512 // 4,
        num_res_blocks: int = 1,
        use_checkpoint: bool = False,
        slide_training: bool = False,
        slide_inference: bool = False,
    ):
        super().__init__(feature_extractor, out_features, backbone_in_size, min_stride, max_stride, projection_dim, num_res_blocks,
                         use_checkpoint, slide_training, slide_inference)

        self.attention_features_res = attention_features_res
        self.feature_dims = feature_dims
        self.attention_features_location = attention_features_location

        self.target_attention_loss = target_attention_loss
        self.attention_select_index = attention_select_index
        # self.feature_extractor.ldm_extractor.unet.enable_xformers_memory_efficient_attention()

        '''
        00:
        torch.Size([1, 1280, 8, 8])
        01:
        torch.Size([1, 1280, 8, 8])
        02:
        torch.Size([1, 1280, 8, 8])
        03:
        torch.Size([1, 1280, 16, 16])
        04:
        torch.Size([1, 1280, 16, 16])
        05:
        torch.Size([1, 1280, 16, 16])
        06:
        torch.Size([1, 640, 32, 32])
        07:
        torch.Size([1, 640, 32, 32])
        08:
        torch.Size([1, 640, 32, 32])
        09:
        torch.Size([1, 320, 64, 64])
        10:
        torch.Size([1, 320, 64, 64])
        11:
        torch.Size([1, 320, 64, 64])
        '''
        self.feature_projections = nn.ModuleList()
        for i, feature_dim in enumerate(self.feature_dims):  # [320, 640, 1280]
            self.feature_projections.append(
                nn.Sequential(*ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=bottleneck_channels,
                        out_channels=projection_dim[i],
                        norm="GN",
                    )
                )
            )

        self._out_feature_strides = dict()
        for s_stride in out_features:
             self._out_feature_strides[s_stride] = 2 ** int(s_stride[1])
        self._out_features = list(self._out_feature_strides.keys())


    def forward_features(self, features, input_image_size, ema_forward=False):

        self.attention_features = dict()

        features_dict = dict()
        for i in features:
            features_dict[i.shape[-1]] = i

        output_features = {}

        if self.feature_extractor.ldm_extractor.final_fuse_vae_decoder_feat:
            output_features['s0'] = features_dict[512]

        for idx, name in enumerate(self._out_features):
            output_feature = None
            stride = self._out_feature_strides[name]
            res = 512 // stride

            restored_feature = features_dict[res]

            if ema_forward:
                output_feature = self.ema_feature_projections[idx](restored_feature)
            else:
                output_feature = self.feature_projections[idx](restored_feature)
            output_features[name] = output_feature
        # [('s3', torch.Size([1, 512, 64, 64])), ('s4', torch.Size([1, 512, 32, 32])), ('s5', torch.Size([1, 512, 16, 16]))]
        
        output = {'output_features': output_features}

        return output
    