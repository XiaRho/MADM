from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper

from evaluation.d2_evaluator import DSECSemSegEvaluator
from data.dataset.cross_modality_dataset import CrossModalityDataset
from data import (
    build_d2_test_dataloader,
    build_d2_train_dataloader,
)

from detectron2.data import MetadataCatalog

dataloader = OmegaConf.create()

dataloader.train = L(build_d2_train_dataloader)(
    dataset=L(CrossModalityDataset)(
        json_path='/data/json_file/Cityscapes_RGB_to_FMB_Infrared_train.json',
        source_root_path='???',
        target_root_path='???',
        train_or_test='train',
        source_resize_h_w=[512, 1024],
        source_crop_size_h_w=[512, 512],
        target_resize_h_w=[512, 683],  # [600, 800]
        target_crop_size_h_w=[512, 512],
        label_convert=[[0, 4], [1, 5], [2, 1], [3, 255], [4, 255], [5, 3], [6, 8], [7, 8], [8, 6], [9, 6], 
                       [10, 0], [11, 2], [12, 2], [13, 7], [14, 7], [15, 7], [16, 7], [17, 7], [18, 7]],
    ),
    total_batch_size=2,
    num_workers=4,
)

dataloader.test = L(build_d2_test_dataloader)(
    dataset=L(CrossModalityDataset)(
        json_path='/data/json_file/Cityscapes_RGB_to_FMB_Infrared_test.json',
        source_root_path='???',
        target_root_path='???',
        train_or_test='test',
        names='Cityscapes_RGB_to_FMB_Infrared',
        test_resize_h_w=[512, 512],
        label_convert=[[0, 255], [1, 4], [2, 5], [3, 1], [4, 8], [5, 8], [6, 6], [7, 0], [8, 2], [9, 7],
                       [10, 7], [11, 7], [12, 7], [13, 7], [14, 3]],
    ),
    local_batch_size=1,
    num_workers=4,
)

dataloader.evaluator = [
    L(DSECSemSegEvaluator)(
        dataset_name="${...test.dataset.names}",
        stuff_classes=['sky', 'building', 'person', 'pole', 'road',
                       'sidewalk', 'vegetation', 'car', 'traffic sign'],
        palette=[70, 130, 180, 70, 70, 70, 220, 20, 60, 153, 153, 153, 128, 64, 128, 244, 35, 232,
                 107, 142, 35, 0, 0, 142, 250, 170, 30],
        ignore_label=255,
        output_dir=None,  # modify in EvalHook (do_test)
        save_predictions_json=False,
        save_eval_results_step=2,
        convert_pred_list=None,
        enable_wandb=False
    ),
]

# dataloader.evaluator = [
#     L(COCOEvaluator)(
#         dataset_name="${...test.dataset.names}",
#         tasks=("segm",),
#     ),
#     L(SemSegEvaluator)(
#         dataset_name="${...test.dataset.names}",
#     ),
#     L(COCOPanopticEvaluator)(
#         dataset_name="${...test.dataset.names}",
#     ),
# ]
#
# dataloader.wrapper = L(OpenPanopticInference)(
#     labels=L(get_openseg_labels)(dataset="coco_panoptic", prompt_engineered=True),
#     metadata=L(MetadataCatalog.get)(name="${...test.dataset.names}"),
# )
