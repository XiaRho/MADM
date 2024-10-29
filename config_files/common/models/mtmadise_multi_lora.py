from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog

from modeling.meta_arch import MTMADISE
from modeling.sem_seg_head import DAFormerHead
from modeling.criterion import CmdiseCriterion

from modeling.backbone.feature_extractor import AttentionFeatureExtractorBackbone
from modeling.meta_arch.ldm_base import BasePromptTimeGenerator
from modeling.meta_arch.ldm_diffusers import LdmDiffusers


model = L(MTMADISE)(
    backbone=L(AttentionFeatureExtractorBackbone)(
        attention_features_res=None,
        feature_dims=[512, 320, 640, 1280],
        projection_dim=[512, 512, 512, 512],
        attention_features_location=None,

        feature_extractor=L(BasePromptTimeGenerator)(
            learnable_cond_prompt=True,
            learnable_cond_time=True,
            clip_state='no',
            num_timesteps=1,
            clip_model_name="ViT-L-14-336",
            ldm_extractor = L(LdmDiffusers)(
                stable_diffusion_name_or_path="~/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/"
                                            "snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/",
                encoder_block_indices=[5],
                unet_block_indices=[5, 8, 11],
                unet_block_indices_type='after',  # 'in' or 'after'
                decoder_block_indices=(),
                input_range='-1+1',
                finetune_unet='all',
            )
        ),
        num_res_blocks=1,
        out_features=["s2", "s3", "s4", "s5"],
        use_checkpoint=False,  # Ori: True
        slide_training=False,
    ),
    sem_seg_head=L(DAFormerHead)(
        in_channels=[512, 512, 512, 512],
        in_keys=['s2', 's3', 's4', 's5'],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True)
            ),
        ),
    ),
    criterion=L(CmdiseCriterion)(
        loss_weight=1.0
    ),
    sem_seg_head_sec_modal=False,
    ema_alpha=0.999,
    pseudo_threshold=0.968,
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    train_palette='???',

    num_queries=100,
    object_mask_threshold=0.0,
    overlap_threshold=0.8,
    metadata=L(MetadataCatalog.get)(name="coco_2017_train_panoptic_with_sem_seg"),
    size_divisibility=64,
    sem_seg_postprocess_before_inference=True,
    # normalize to [0, 1]
    pixel_mean=[0.0, 0.0, 0.0],
    pixel_std=[255.0, 255.0, 255.0],
    # inference
    semantic_on=True,
    instance_on=True,
    panoptic_on=True,
    test_topk_per_image=100,
)
