#!/usr/bin/env python
# Our code heavily built on ODISE (https://github.com/NVlabs/ODISE), thanks to their contributions!

"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import argparse
import logging
import os.path as osp
from contextlib import ExitStack
from typing import MutableSequence

import wandb
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import JSONWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from iopath.common.s3 import S3PathHandler
from omegaconf import OmegaConf

import os
import time
from config import auto_scale_workers, instantiate_cmdise
from engine.hooks import EvalHook, VisHook
from evaluation import inference_on_dataset
from detectron2.data import DatasetCatalog

from checkpoint import ODISECheckpointer
from engine.defaults import default_setup, get_dataset_from_loader, get_model_from_module
from engine.train_loop import AMPTrainer, SimpleTrainer
from utils.events import CommonMetricPrinter, WandbWriter, WriterStack

PathManager.register_handler(S3PathHandler())

logger = logging.getLogger("odise")


def default_writers(cfg):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    if "log_dir" in cfg.train:
        log_dir = cfg.train.log_dir
    else:
        log_dir = cfg.train.output_dir
    PathManager.mkdirs(log_dir)
    ret = [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(
            cfg.train.max_iter, run_name=osp.join(cfg.train.run_name, cfg.train.run_tag)
        ),
        JSONWriter(osp.join(log_dir, "metrics.json")),
    ]
    if cfg.train.wandb.enable_writer:
        ret.append(
            WandbWriter(
                max_iter=cfg.train.max_iter,
                run_name=osp.join(cfg.train.run_name, cfg.train.run_tag),
                output_dir=log_dir,
                project=cfg.train.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=False),
                resume=cfg.train.wandb.resume,
            )
        )

    return ret


class VisualizeRunner:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def __call__(self, final_iter=False, next_iter=0):
        return do_vis(self.cfg, self.model, final_iter=final_iter, next_iter=next_iter)


def do_vis(cfg, model, *, final_iter=False, next_iter=0):

    # make a copy incase we modify it every time calling do_vis
    cfg = OmegaConf.create(cfg)

    model_ddp = get_model_from_module(model)
    model_ddp.vis_results(save_path=cfg.train.output_dir, iter_index=next_iter)

    logger.info("Visualization the results for [{}] iteration:".format(next_iter))
    return dict()


class InferenceRunner:
    def __init__(self, cfg, model, enable_wandb=False):
        self.cfg = cfg
        self.model = model
        self.enable_wandb = enable_wandb

    def __call__(self, final_iter=False, next_iter=0):
        return do_test(self.cfg, self.model, final_iter=final_iter, next_iter=next_iter, enable_wandb=self.enable_wandb)


def do_test(cfg, model, *, final_iter=False, next_iter=0, enable_wandb=False):
    all_ret = dict()
    # make a copy incase we modify it every time calling do_test
    cfg = OmegaConf.create(cfg)

    # BC for detectron
    if "evaluator" in cfg.dataloader and "test" in cfg.dataloader:
        task_final_iter_only = cfg.dataloader.get("final_iter_only", False)
        task_eval_period = cfg.dataloader.get("eval_period", 1)
        if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
            logger.info(
                f"Skip test set evaluation at iter {next_iter}, "
                f"since task_final_iter_only={task_final_iter_only}, "
                f"next_iter {next_iter} % task_eval_period {task_eval_period}"
                f"={next_iter % task_eval_period} != 0"
            )
        else:
            loader = instantiate(cfg.dataloader.test)

            if "wrapper" in cfg.dataloader:
                wrapper_cfg = cfg.dataloader.wrapper
                # look for the last wrapper
                while "model" in wrapper_cfg:
                    wrapper_cfg = wrapper_cfg.model
                wrapper_cfg.model = get_model_from_module(model)
                # poping _with_dataset_
                if wrapper_cfg.pop("_with_dataset_", False):
                    wrapper_cfg.dataset = get_dataset_from_loader(loader)
                inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
            else:
                inference_model = model

            # poping _with_dataset_
            if isinstance(cfg.dataloader.evaluator, MutableSequence):
                for i in range(len(cfg.dataloader.evaluator)):
                    if cfg.dataloader.evaluator[i].pop("_with_dataset_", False):
                        cfg.dataloader.evaluator[i].dataset = get_dataset_from_loader(loader)
                    cfg.dataloader.evaluator[i].output_dir = osp.join(cfg.train.output_dir, '{:06d}'.format(next_iter))
            else:
                if cfg.dataloader.evaluator.pop("_with_dataset_", False):
                    cfg.dataloader.evaluator.dataset = get_dataset_from_loader(loader)
                cfg.dataloader.evaluator.output_dir = osp.join(cfg.train.output_dir, '{:06d}'.format(next_iter))

            ret = inference_on_dataset(
                inference_model,
                loader,
                instantiate(cfg.dataloader.evaluator),
                use_amp=cfg.train.amp.enabled,
                logger=logger
            )
            # have already implemented in wandb
            # assert len(list(ret.keys())) == 1
            # if enable_wandb:
            #     wandb.log({"MIoU": ret[list(ret.keys())[0]]['mIoU']}, step=next_iter)
            for key in ret.keys():
                print_csv_format(ret[key])
            all_ret.update(ret)
    # if "extra_task" in cfg.dataloader:
    #     for task in cfg.dataloader.extra_task:
    #         task_final_iter_only = cfg.dataloader.extra_task[task].get("final_iter_only", False)
    #         task_eval_period = cfg.dataloader.extra_task[task].get("eval_period", 1)
    #         if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
    #             logger.info(
    #                 f"Skip {task} evaluation at iter {next_iter}, "
    #                 f"since task_final_iter_only={task_final_iter_only}, "
    #                 f"next_iter {next_iter} % task_eval_period {task_eval_period}"
    #                 f"={next_iter % task_eval_period} != 0"
    #             )
    #             continue
    #
    #         logger.info(f"Evaluating extra task: {task}")
    #         loader = instantiate(cfg.dataloader.extra_task[task].loader)
    #
    #         # poping _with_dataset_
    #         if isinstance(cfg.dataloader.extra_task[task].evaluator, MutableSequence):
    #             for i in range(len(cfg.dataloader.extra_task[task].evaluator)):
    #                 if cfg.dataloader.extra_task[task].evaluator[i].pop("_with_dataset_", False):
    #                     cfg.dataloader.extra_task[task].evaluator[
    #                         i
    #                     ].dataset = get_dataset_from_loader(loader)
    #         else:
    #             if cfg.dataloader.extra_task[task].evaluator.pop("_with_dataset_", False):
    #                 cfg.dataloader.extra_task[task].evaluator.dataset = get_dataset_from_loader(
    #                     loader
    #                 )
    #
    #         if "wrapper" in cfg.dataloader.extra_task[task]:
    #             wrapper_cfg = cfg.dataloader.extra_task[task].wrapper
    #             # look for the last wrapper
    #             while "model" in wrapper_cfg:
    #                 wrapper_cfg = wrapper_cfg.model
    #             wrapper_cfg.model = get_model_from_module(model)
    #             # poping _with_dataset_
    #             if wrapper_cfg.pop("_with_dataset_", False):
    #                 wrapper_cfg.dataset = get_dataset_from_loader(loader)
    #             inference_model = create_ddp_model(
    #                 instantiate(cfg.dataloader.extra_task[task].wrapper)
    #             )
    #         else:
    #             inference_model = model
    #
    #         ret = inference_on_dataset(
    #             inference_model,
    #             loader,
    #             instantiate(cfg.dataloader.extra_task[task].evaluator),
    #             use_amp=cfg.train.amp.enabled,
    #         )
    #         print_csv_format(ret)
    #         all_ret.update(ret)
    # logger.info("Evaluation results for all tasks:")
    # print_csv_format(all_ret)
    return all_ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    logger = logging.getLogger("odise")
    # set wandb resume before create writer
    cfg.train.wandb.resume = args.resume and ODISECheckpointer.has_checkpoint_in_dir(
        cfg.train.output_dir
    )
    # create writers at the beginning for W&B logging
    if comm.is_main_process():
        writers = default_writers(cfg)
    comm.synchronize()

    # not sure why d2 use ExitStack(), maybe easier for multiple context
    with ExitStack() as stack:
        stack.enter_context(
            WriterStack(
                logger=logger,
                writers=writers if comm.is_main_process() else None,
            )
        )
        logger.info(f"Wandb resume: {cfg.train.wandb.resume}")
        # log config again for w&b
        logger.info(f"Config:\n{LazyConfig.to_py(cfg)}")

        model = instantiate_cmdise(cfg.model)
        model.to(cfg.train.device)

        cfg.optimizer.params.model = model

        # param_groups = instantiate(cfg.optimizer.params)
        # cfg.optimizer.params = param_groups
        # optim = instantiate(cfg.optimizer)
        
        optim = instantiate(cfg.optimizer)

        train_loader = instantiate(cfg.dataloader.train)

        if cfg.train.amp.enabled:
            model = create_ddp_model(model, **cfg.train.ddp)
            trainer = AMPTrainer(model, train_loader, optim, grad_clip=cfg.train.grad_clip)
        else:
            model = create_ddp_model(model, **cfg.train.ddp)
            trainer = SimpleTrainer(model, train_loader, optim, grad_clip=cfg.train.grad_clip)

        checkpointer = ODISECheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
        )

        if 'step_2' in cfg.dataloader.train.dataset.json_path and cfg.train.init_checkpoint is not None:
            checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)

        # set wandb resume before create writer
        cfg.train.wandb.resume = args.resume and checkpointer.has_checkpoint()
        logger.info(f"Wandb resume: {cfg.train.wandb.resume}")

        trainer.register_hooks(
            [
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None,
                EvalHook(cfg.train.eval_period, InferenceRunner(cfg, model, enable_wandb=cfg.train.wandb.enable_writer)),
                # VisHook(cfg.train.vis_period, VisualizeRunner(cfg, model)),
                hooks.BestCheckpointer(checkpointer=checkpointer, **cfg.train.best_checkpointer)
                if comm.is_main_process() and "best_checkpointer" in cfg.train
                else None,
                hooks.PeriodicWriter(
                    writers=writers,
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None,
            ]
        )
        comm.synchronize()

        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
        if args.resume and checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
    comm.synchronize()
    # keep trainer.train() out of stack since it has try/catch handling
    if hasattr(cfg.train, 'stop_iter'):
        trainer.train(start_iter, cfg.train.stop_iter)
    else:
        trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.train.run_name = (
        "${train.cfg_name}_bs${dataloader.train.total_batch_size}" + f"x{comm.get_world_size()}"
    )
    if hasattr(args, "reference_world_size") and args.reference_world_size:
        cfg.train.reference_world_size = args.reference_world_size
    cfg = auto_scale_workers(cfg, comm.get_world_size())
    cfg.train.cfg_name = osp.splitext(osp.basename(args.config_file))[0]

    if hasattr(args, 'debug') and args.debug:
        args.wandb = False
        args.tag += '_TEST'
        cfg.train.checkpointer.period = 5  # 4
        cfg.train.vis_period = 2

    if hasattr(args, "output") and args.output:
        cfg.train.output_dir = args.output
    else:
        cfg.train.output_dir = osp.join("output", cfg.train.run_name)
    if hasattr(args, "tag") and args.tag:
        if not args.eval_only:
            now_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            now_time = now_time[2:8] + '-' + now_time[8:]
            args.tag = now_time + '_' + args.tag
            cfg.train.run_tag = args.tag
            cfg.train.output_dir = osp.join(cfg.train.output_dir, cfg.train.run_tag)
        else:
            assert '_eval' in args.tag, '\"_eval\" must in {}'.format(args.tag)
            cfg.train.run_tag = args.tag
            cfg.train.output_dir = osp.join(cfg.train.output_dir, cfg.train.run_tag)
        if hasattr(args, 'debug') and args.debug:
            sub_dir = cfg.train.output_dir.split('/')[-2]
            cfg.train.output_dir = cfg.train.output_dir.replace(sub_dir, '[Debug]' + sub_dir)

    if hasattr(args, "wandb") and args.wandb:
        cfg.train.wandb.enable_writer = args.wandb
        cfg.train.wandb.enable_visualizer = args.wandb
    if hasattr(args, "amp") and args.amp:
        cfg.train.amp.enabled = args.amp
    if hasattr(args, "init_from") and args.init_from:
        cfg.train.init_checkpoint = args.init_from

    cfg.train.log_dir = cfg.train.output_dir
    if hasattr(args, "log_tag") and args.log_tag:
        cfg.train.log_dir = osp.join(cfg.train.log_dir, args.log_tag)

    madm_path = '/path_to/MADM'
    if 'Cityscapes_RGB_to_DSEC_Event' in cfg.dataloader.train.dataset.json_path:
        cfg.dataloader.train.dataset.source_root_path = '/path_to/cityscapes/'
        cfg.dataloader.train.dataset.target_root_path = '/path_to/dsec_dataset/'
    elif 'Cityscapes_RGB_to_DELIVER_Depth' in cfg.dataloader.train.dataset.json_path:
        cfg.dataloader.train.dataset.source_root_path = '/path_to/cityscapes/'
        cfg.dataloader.train.dataset.target_root_path = '/path_to/DELIVER/'
    else:
        assert 'Cityscapes_RGB_to_FMB_Infrared' in cfg.dataloader.train.dataset.json_path
        cfg.dataloader.train.dataset.source_root_path = '/path_to/cityscapes/'
        cfg.dataloader.train.dataset.target_root_path = '/path_to/FMB/'

    if 'step_2' in cfg.dataloader.train.dataset.json_path:
        cfg.dataloader.train.dataset.source_root_path = madm_path + osp.dirname(cfg.dataloader.train.dataset.json_path)

    cfg.dataloader.test.dataset.source_root_path = cfg.dataloader.train.dataset.source_root_path
    cfg.dataloader.test.dataset.target_root_path = cfg.dataloader.train.dataset.target_root_path
    cfg.dataloader.train.dataset.json_path = madm_path + cfg.dataloader.train.dataset.json_path 
    cfg.dataloader.test.dataset.json_path = madm_path + cfg.dataloader.test.dataset.json_path

    if hasattr(args, 'bs') and args.bs != -1:
        assert args.bs % args.num_gpus == 0
        cfg.dataloader.train.total_batch_size = args.bs

    if hasattr(args, 'lr') and args.lr is not None:
        cfg.optimizer.lr = args.lr

    if hasattr(args, 'max_iter') and args.max_iter != -1:
        cfg.train.max_iter = args.max_iter
        sche_num_updates = 2 * cfg.train.max_iter - 1
        cfg.lr_multiplier.scheduler.num_updates = sche_num_updates
        cfg.lr_multiplier.scheduler.milestones = [int(0.88888 * sche_num_updates), int(0.96296 * sche_num_updates)]

    if hasattr(args, 'stop_iter') and args.stop_iter != -1:
        cfg.train.stop_iter = args.stop_iter

    if hasattr(args, 'unet_lr') and args.unet_lr is not None:
        cfg.optimizer.params.unet_lr = args.unet_lr

    if hasattr(args, 'eval_iter') and args.eval_iter != -1:
        cfg.train.checkpointer.period = args.eval_iter

    if hasattr(args, "enable_sem_seg_head_sec_modal") and args.enable_sem_seg_head_sec_modal:
        cfg.model.sem_seg_head_sec_modal = args.enable_sem_seg_head_sec_modal

    if hasattr(args, "norm_n1_p1") and args.norm_n1_p1:
        cfg.model.pixel_mean = [127.5, 127.5, 127.5]
        cfg.model.pixel_std = [127.5, 127.5, 127.5]

    if hasattr(args, "disable_mixup") and args.disable_mixup:
        cfg.model.enable_mixup = False

    if hasattr(args, 'remove_amp') and args.remove_amp is not None:
        cfg.dataloader.train.dataset.remove_amp = args.remove_amp
        cfg.dataloader.test.dataset.remove_amp = args.remove_amp
        cfg.model.remove_amp = args.remove_amp
        cfg.model.seed = cfg.train.seed
        cfg.model.max_iter = cfg.train.max_iter

    if hasattr(args, 'fda_fusion_val') and args.fda_fusion_val is not None:
        cfg.dataloader.train.dataset.fda_fusion_val = args.fda_fusion_val
        cfg.dataloader.test.dataset.fda_fusion_val = args.fda_fusion_val

    if hasattr(args, 'pl_crop') and args.pl_crop:
        cfg.model.pl_crop = args.pl_crop

    if hasattr(args, 'rare_class_sample') and args.rare_class_sample:
        cfg.dataloader.train.dataset.rare_class_sample = args.rare_class_sample

    if hasattr(args, 'remove_texture') and args.remove_texture is not None:
        cfg.dataloader.train.dataset.remove_texture = True
        cfg.model.remove_texture = args.remove_texture

    if hasattr(args, 'without_prompt') and args.without_prompt:
        cfg.model.backbone.feature_extractor.without_prompt = True
        cfg.model.backbone.feature_extractor.learnable_time_embed = False
    
    if hasattr(args, 'without_vae_encoder_feat') and args.without_vae_encoder_feat:
        '''
        model.backbone.feature_extractor.ldm_extractor.encoder_block_indices = [5]
        model.backbone.feature_dims = [512, 320, 640, 1280]
        model.backbone.out_features = ["s2", "s3", "s4", "s5"]
        model.sem_seg_head.in_channels = [512, 512, 512, 512]
        model.sem_seg_head.in_keys = ['s2', 's3', 's4', 's5']
        model.sem_seg_head.in_index = [0, 1, 2, 3]
        '''
        cfg.model.backbone.feature_extractor.ldm_extractor.encoder_block_indices = []
        cfg.model.backbone.feature_dims = [320, 640, 1280]  # 128: 512, 320: 64, 640: 32, 1280: 16
        cfg.model.backbone.out_features = ["s3", "s4", "s5"]
        cfg.model.sem_seg_head.in_channels = [512, 512, 512]
        cfg.model.sem_seg_head.in_keys = ['s3', 's4', 's5']
        cfg.model.sem_seg_head.in_index = [0, 1, 2]

    if hasattr(args, 'concat_corss_attention_feat_to_conv_seg') and args.concat_corss_attention_feat_to_conv_seg:
        cfg.model.backbone.concat_attention_to_conv_seg = True
        cfg.model.sem_seg_head.concat_attention_to_conv_seg = True
        cfg.model.backbone.feature_dims = [512, 320, 640, 1280]

    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        cfg.model.backbone.use_checkpoint = True
    
    if hasattr(args, 'same_cond_params') and args.same_cond_params:
        cfg.model.backbone.feature_extractor.same_cond_params = True

    if hasattr(args, 'without_prompt_alpha') and args.without_prompt_alpha:
        cfg.model.backbone.feature_extractor.without_prompt_alpha = True

    if hasattr(args, 'multi_layer_prompt') and args.multi_layer_prompt:
        cfg.model.backbone.feature_extractor.multi_layer_prompt = True

    if hasattr(args, 'target_attention_loss') and args.target_attention_loss:
        cfg.model.backbone.target_attention_loss = True
        cfg.model.backbone.feature_extractor.mix_source_target_prompt = True
    
    if hasattr(args, 'init_uncond_prompt') and args.init_uncond_prompt:
        cfg.model.backbone.feature_extractor.init_uncond_prompt = True

    if hasattr(args, 'attention_select_index') and args.attention_select_index:
        cfg.model.backbone.attention_select_index = set(args.attention_select_index)

    if hasattr(args, 'mask_prompt_ratio') and args.mask_prompt_ratio:
        cfg.model.backbone.feature_extractor.mask_prompt_ratio = args.mask_prompt_ratio

    if hasattr(args, 'detach_mask_prompt') and args.detach_mask_prompt:
        cfg.model.backbone.feature_extractor.detach_mask_prompt = args.detach_mask_prompt

    if hasattr(args, 'prompt_perturbation') and args.prompt_perturbation:
        cfg.model.backbone.feature_extractor.prompt_perturbation = args.prompt_perturbation
    
    if hasattr(args, 'MIC') and args.MIC:
        cfg.model.mic = True
    
    if hasattr(args, 'mask_ratio') and args.mask_ratio is not None:
        cfg.model.mask_ratio = args.mask_ratio

    if hasattr(args, 'warmup_lr') and args.warmup_lr: 
        from detectron2.config import LazyCall as L
        from detectron2.solver import WarmupParamScheduler
        from fvcore.common.param_scheduler import LinearParamScheduler
        cfg.lr_multiplier = L(WarmupParamScheduler)(
            scheduler=L(LinearParamScheduler)(
                start_value=1.0 / (1 - 0.0375),
                end_value=0,
            ),
            warmup_length=0.0375,
            warmup_factor=1e-6,
        )
        cfg.optimizer.weight_decay = 0.01

    if hasattr(args, 'FD') and args.FD is not None:
        cfg.model.fd = args.FD

    if hasattr(args, 'FD_attention') and args.FD_attention is not None:
        cfg.model.fd_attention = args.FD_attention
        cfg.model.backbone.attention_features_res = {16, 32}
        cfg.model.backbone.attention_features_location = ['up']

    if hasattr(args, 'prompt_confidence') and args.prompt_confidence is not None:
        cfg.model.prompt_confidence = args.prompt_confidence
        cfg.model.rand_prompt_scale = args.rand_prompt_scale
        cfg.model.backbone.feature_extractor.rand_prompt_scale = args.rand_prompt_scale

    if hasattr(args, 'finetune_without_cross_attention') and args.finetune_without_cross_attention:
        cfg.model.backbone.feature_extractor.ldm_extractor.finetune_unet = 'without cross-attention'

    if hasattr(args, 'finetune_no') and args.finetune_no:
        cfg.model.backbone.feature_extractor.ldm_extractor.finetune_unet = 'no'
    
    if hasattr(args, 'with_clip') and args.with_clip is not None:
        cfg.model.backbone.feature_extractor.clip_state = args.with_clip

    if hasattr(args, 'merge_more_target_data') and args.merge_more_target_data is not None:
        cfg.dataloader.train.dataset.merge_more_target_data = args.merge_more_target_data
    
    if hasattr(args, 'merge_with_pl_data') and args.merge_with_pl_data is not None:
        cfg.model.merge_with_pl_data = args.merge_with_pl_data
        cfg.dataloader.train.dataset.pl_data_path = args.pl_data_path
        if cfg.model.merge_with_pl_data in {'gradual_linear_mix', 'anti_gradual_linear_mix'}:
            cfg.model.max_iter = cfg.train.max_iter

    if hasattr(args, 'slide_inference') and args.slide_inference:
        cfg.model.backbone.slide_inference = True

    if hasattr(args, 'concat_pixel_shuffle') and args.concat_pixel_shuffle:
        cfg.model.backbone.feature_extractor.ldm_extractor.concat_pixel_shuffle = True

    if hasattr(args, 'vis_period') and args.vis_period is not None:
        cfg.train.vis_period = args.vis_period
    cfg.model.vis_period = cfg.train.vis_period
    cfg.model.output_dir = cfg.train.output_dir

    if hasattr(args, 'single_scale_decoder') and args.single_scale_decoder:
        cfg.model.sem_seg_head.in_channels = [512]
        cfg.model.sem_seg_head.in_keys = ['s3']
        cfg.model.sem_seg_head.in_index = [0]

        cfg.model.backbone.feature_dims = [320]
        cfg.model.backbone.out_features = ["s3"]
        cfg.model.backbone.feature_extractor.ldm_extractor.encoder_block_indices = []
        cfg.model.backbone.feature_extractor.ldm_extractor.unet_block_indices = [11]

    if hasattr(args, 'add_latent_noise') and args.add_latent_noise != -1:
        cfg.model.backbone.feature_extractor.ldm_extractor.add_latent_noise = args.add_latent_noise
    
    if hasattr(args, 'prompt_seq_len') and args.prompt_seq_len != -1:
        cfg.model.backbone.feature_extractor.prompt_seq_len = args.prompt_seq_len

    if hasattr(args, 'disable_color_aug') and args.disable_color_aug:
        cfg.model.color_aug_flag = False

    if hasattr(args, 'norm_latent_noise') and args.norm_latent_noise:
        cfg.model.backbone.feature_extractor.ldm_extractor.norm_latent_noise = True

    if hasattr(args, 'denoise_supervise') and args.denoise_supervise is not None:
        cfg.model.denoise_supervise = args.denoise_supervise
        cfg.model.denoise_interval = args.denoise_interval

    if hasattr(args, 'denoise_timestep_range') and args.denoise_timestep_range is not None:
        cfg.model.denoise_timestep_range = args.denoise_timestep_range

    if hasattr(args, 'lora_configs') and args.lora_configs is not None:
        if args.lora_configs == ['']:
            args.lora_configs = []
        cfg.model.lora_configs = args.lora_configs
    if hasattr(args, 'vae_decoder_loss') and args.vae_decoder_loss is not None:
        cfg.model.vae_decoder_loss = args.vae_decoder_loss
        cfg.model.backbone.feature_extractor.ldm_extractor.vae_decoder_loss = True
        if hasattr(args, 'final_fuse_vae_decoder_feat') and args.final_fuse_vae_decoder_feat:
            # cfg.model.backbone.feature_extractor.ldm_extractor.final_fuse_vae_decoder_feat = True
            # cfg.model.sem_seg_head.in_keys = ["s0"] + list(cfg.model.sem_seg_head.in_keys)
            cfg.model.sem_seg_head.final_fuse_vae_decoder_feat = True
        # else:
        cfg.model.backbone.feature_dims[0] = 3
        cfg.model.backbone.projection_dim[0] = 128
        cfg.model.sem_seg_head.in_channels[0] = 128
        cfg.model.backbone.feature_extractor.ldm_extractor.encoder_block_indices = []
        cfg.model.backbone.out_features[0] = "s0"
        cfg.model.sem_seg_head.in_keys[0] = "s0"
        if hasattr(args, 'reg_uncertain') and args.reg_uncertain:
            cfg.model.reg_uncertain = True
        if hasattr(args, 'reg_target_palette') and args.reg_target_palette is not None:
            cfg.model.reg_target_palette = args.reg_target_palette
        if hasattr(args, 'vae_decoder_loss_type') and args.vae_decoder_loss_type is not None:
            cfg.model.vae_decoder_loss_type = args.vae_decoder_loss_type
        if hasattr(args, 'vae_decoder_loss_weight') and args.vae_decoder_loss_weight is not None:
            cfg.model.vae_decoder_loss_weight = args.vae_decoder_loss_weight
            assert len(args.vae_decoder_loss_weight) == len(cfg.model.vae_decoder_loss)
        if hasattr(args, 'MIC_reg') and args.MIC_reg is not None:
            cfg.model.mic_reg = args.MIC_reg
            if hasattr(args, 'MIC_reg_wo_pl_val') and args.MIC_reg_wo_pl_val:
                cfg.model.MIC_reg_wo_pl_val = True

    if hasattr(args, 'baseline_wo_encoder_feat') and args.baseline_wo_encoder_feat:
        cfg.model.backbone.feature_dims = cfg.model.backbone.feature_dims[1:]
        cfg.model.backbone.projection_dim = cfg.model.backbone.projection_dim[1:]
        cfg.model.backbone.feature_extractor.ldm_extractor.encoder_block_indices = []
        cfg.model.backbone.out_features = cfg.model.backbone.out_features[1:]
        cfg.model.sem_seg_head.in_channels = cfg.model.sem_seg_head.in_channels[1:]
        cfg.model.sem_seg_head.in_keys = cfg.model.sem_seg_head.in_keys[1:]
        cfg.model.sem_seg_head.in_index = cfg.model.sem_seg_head.in_index[1:]
        
    if hasattr(args, 'mask_diff') and args.mask_diff is not None:
        cfg.model.mask_diff = args.mask_diff
        if args.mask_diff == 'circle':
            cfg.model.backbone.feature_extractor.ldm_extractor.input_channel_plus = 2
        else:
            cfg.model.backbone.feature_extractor.ldm_extractor.input_channel_plus = 1

    if hasattr(args, 'add_zero_grad') and args.add_zero_grad:
        cfg.model.add_zero_grad = args.add_zero_grad

    if hasattr(args, 'rev_noise_sup') and args.rev_noise_sup:
        cfg.model.rev_noise_sup = True
        cfg.model.rev_noise_end_iter = args.rev_noise_end_iter
        if hasattr(args, 'rev_noise_gradually') and args.rev_noise_gradually: 
            cfg.model.rev_noise_gradually = args.rev_noise_gradually
            cfg.model.max_iter = cfg.train.max_iter
            
    if hasattr(args, 'pseudo_threshold') and args.pseudo_threshold is not None:
        cfg.model.pseudo_threshold = args.pseudo_threshold

    if hasattr(args, 'noise_reg') and args.noise_reg is not None:
        cfg.model.noise_reg = args.noise_reg

    if hasattr(args, 'ema_w_unet') and args.ema_w_unet:
        cfg.model.ema_w_unet = True
    
    if hasattr(args, 'seed') and args.seed is not None:
        cfg.model.seed = args.seed
        cfg.train.seed = args.seed

    if hasattr(args, 'eval_with_noise') and args.eval_with_noise is not None:
        cfg.model.eval_with_noise = args.eval_with_noise

    if args.eval_only:
        cfg.dataloader.evaluator[0].save_eval_results_step = 1
        cfg.dataloader.evaluator[0].eval_only = True
        assert '_test' in cfg.dataloader.test.dataset.json_path
        # cfg.dataloader.test.dataset.json_path = cfg.dataloader.test.dataset.json_path.replace('_test.json', '_train_eval.json')

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    logger = setup_logger(cfg.train.log_dir, distributed_rank=comm.get_rank(), name="odise")

    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")

    if args.eval_only:
        model = instantiate_cmdise(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        ODISECheckpointer(model, cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=args.resume
        )
        with ExitStack() as stack:
            stack.enter_context(
                WriterStack(
                    logger=logger,
                    writers=default_writers(cfg) if comm.is_main_process() else None,
                )
            )
            logger.info(do_test(cfg, model, final_iter=True))
        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()
    else:
        do_train(args, cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        "odise training and evaluation script",
        parents=[default_argument_parser()],
        add_help=False,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--init-from", type=str, help="init from the given checkpoint")
    parser.add_argument("--tag", default="default", type=str, help="tag of experiment")
    parser.add_argument("--log-tag", type=str, help="tag of experiment")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--amp", action="store_true", help="Use AMP for mixed precision training")
    parser.add_argument("--reference-world-size", "--ref", type=int)

    parser.add_argument("--debug", action="store_true", help="debug code with small checkpointer.period and vis_period")
    parser.add_argument("--bs", type=int, help='batch_size', default=-1)
    parser.add_argument("--enable_sem_seg_head_sec_modal", action="store_true")
    parser.add_argument("--max_iter", type=int, default=-1)
    parser.add_argument("--stop_iter", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--unet_lr", type=float, default=None)
    parser.add_argument("--eval_iter", type=int, default=-1)
    parser.add_argument("--norm_n1_p1", action="store_true")
    parser.add_argument("--disable_mixup", action="store_true")
    parser.add_argument("--remove_amp", type=float, default=None, nargs='+')
    parser.add_argument("--fda_fusion_val", type=float, default=None, nargs='+')
    parser.add_argument("--pl_crop", action="store_true")
    parser.add_argument("--rare_class_sample", action="store_true")
    parser.add_argument("--remove_texture", type=float, default=None)
    parser.add_argument("--without_prompt", action="store_true")
    parser.add_argument("--without_vae_encoder_feat", action="store_true")
    parser.add_argument("--concat_corss_attention_feat_to_conv_seg", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--same_cond_params", action="store_true")
    parser.add_argument("--without_prompt_alpha", action="store_true")
    parser.add_argument("--multi_layer_prompt", action="store_true")
    parser.add_argument("--target_attention_loss", action="store_true")
    parser.add_argument("--init_uncond_prompt", action="store_true")
    parser.add_argument("--attention_select_index", type=int, default=None, nargs='+')
    parser.add_argument("--mask_prompt_ratio", type=float, default=None)
    parser.add_argument("--detach_mask_prompt", action="store_true")
    parser.add_argument("--prompt_perturbation", type=float, default=None)
    parser.add_argument("--MIC", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--warmup_lr", action="store_true")
    parser.add_argument("--FD", type=float, default=None)
    parser.add_argument("--FD_attention", type=float, default=None)
    parser.add_argument("--prompt_confidence", type=float, default=None)
    parser.add_argument("--rand_prompt_scale", type=float, default=0.5)
    parser.add_argument("--finetune_without_cross_attention", action="store_true")
    parser.add_argument("--finetune_no", action="store_true")
    parser.add_argument("--with_clip", type=str, choices=['no_learnable_clip', 'learnable_clip'])
    parser.add_argument("--merge_more_target_data", type=str, default=None, help="like dreambooth: use more unlabeled data")
    parser.add_argument("--merge_with_pl_data", type=str, default=None)
    parser.add_argument("--pl_data_path", type=str, default=None)
    parser.add_argument("--slide_inference", action="store_true")
    parser.add_argument("--concat_pixel_shuffle", action="store_true")
    parser.add_argument("--vis_period", type=int, default=None)
    parser.add_argument("--single_scale_decoder", action="store_true")
    parser.add_argument("--add_latent_noise", type=float, default=-1)
    parser.add_argument("--prompt_seq_len", type=int, default=-1)
    parser.add_argument("--disable_color_aug", action="store_true")
    parser.add_argument("--norm_latent_noise", action="store_true")

    parser.add_argument("--denoise_supervise", type=float, default=None)
    parser.add_argument("--denoise_timestep_range", type=int, default=None, nargs='+')
    parser.add_argument("--denoise_interval", type=int, default=None)

    parser.add_argument("--lora_configs", type=str, nargs='+', default=None)
    parser.add_argument("--vae_decoder_loss", type=str, default=None, choices=['s', 't', 'st', None])
    parser.add_argument("--vae_decoder_loss_type", type=str, default='L2', choices=['L2', 'L1'])
    parser.add_argument("--vae_decoder_loss_weight", type=float, nargs='+', default=None)
    parser.add_argument("--mask_diff", type=str, default=None)
    parser.add_argument("--final_fuse_vae_decoder_feat", action="store_true")
    parser.add_argument("--reg_uncertain", action="store_true")
    parser.add_argument("--reg_target_palette", type=str, default=None)
    parser.add_argument("--add_zero_grad", action="store_true")
    parser.add_argument("--MIC_reg", type=float, default=None)
    parser.add_argument("--rev_noise_sup", action="store_true")
    parser.add_argument("--rev_noise_end_iter", type=int, default=2500)
    parser.add_argument("--rev_noise_gradually", action="store_true")
    parser.add_argument("--pseudo_threshold", type=float, default=None)
    parser.add_argument("--noise_reg", type=float, default=None)
    parser.add_argument("--MIC_reg_wo_pl_val", action="store_true")
    parser.add_argument("--ema_w_unet", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline_wo_encoder_feat", action="store_true")
    parser.add_argument("--eval_with_noise", type=int, default=None)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
