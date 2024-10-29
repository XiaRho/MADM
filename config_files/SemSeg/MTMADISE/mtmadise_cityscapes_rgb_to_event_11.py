from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...common.models.mtmadise_multi_lora import model
from ...common.data.cityscapes_rgb_to_dsec_event_semseg import dataloader
from ...common.train import train
from ...common.optim import AdamW as optimizer

model.lora_configs = []  # --lora_configs ''
model.target_modality = 'Event'
model.class_names = dataloader.evaluator[0].stuff_classes
model.train_palette = dataloader.evaluator[0].palette
model.sem_seg_head.num_classes = len(dataloader.evaluator[0].stuff_classes)

train.max_iter = 10000  # --max_iter 10000
sche_num_updates = 2 * train.max_iter - 1
train.grad_clip = 0.01
train.checkpointer.period = 1000  # --eval_iter 1000
train.vis_period = 250  # --vis_period 250

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        # assume 100e with batch-size 64 as original LSJ
        # Equivalent to 100 epochs.
        # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
        milestones=[int(0.88888 * sche_num_updates), int(0.96296 * sche_num_updates)],
        num_updates=sche_num_updates,
    ),
    # for warmup length we adopted COCO LSJ setting
    warmup_length=500 / 184375,
    warmup_factor=0.067,
)

optimizer.lr = 5e-6  # --lr 5e-6
optimizer.weight_decay = 0.05

train.amp.enabled = True  # --amp
dataloader.train.dataset.rare_class_sample = True  # --rare_class_sample
model.backbone.feature_extractor.same_cond_params = True  # --same_cond_params
model.rev_noise_sup = True  # --rev_noise_sup
model.rev_noise_end_iter = 8000  # --rev_noise_end_iter 8000 
model.rev_noise_gradually = True  # --rev_noise_gradually
model.max_iter = train.max_iter  # --rev_noise_gradually
model.denoise_timestep_range = [50, 51]  # --denoise_timestep_range 50 51
# --vae_decoder_loss s
model.vae_decoder_loss = 's'
model.backbone.feature_extractor.ldm_extractor.vae_decoder_loss = True
model.backbone.feature_dims[0] = 3
model.backbone.projection_dim[0] = 128
model.sem_seg_head.in_channels[0] = 128
model.backbone.feature_extractor.ldm_extractor.encoder_block_indices = []
model.backbone.out_features[0] = "s0"
model.sem_seg_head.in_keys[0] = "s0"
model.reg_uncertain = True  # reg_uncertain
model.vae_decoder_loss_type = 'L1'  # --vae_decoder_loss_type L1 
model.vae_decoder_loss_weight = [20.0]  # --vae_decoder_loss_weight 20.0

# CUDA_VISIBLE_DEVICES=0,1 python main.py --config-file config_files/SemSeg/MTMADISE/mtmadise_cityscapes_rgb_to_event_11.py --num-gpus 2 --bs 2 --tag RGB2Event
