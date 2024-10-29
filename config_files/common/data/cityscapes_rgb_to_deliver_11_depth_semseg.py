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
        json_path='/data/json_file/Cityscapes_RGB_to_DELIVER_Depth_train.json',
        source_root_path='???',
        target_root_path='???',
        train_or_test='train',
        source_resize_h_w=[512, 1024],
        source_crop_size_h_w=[512, 512],
        target_resize_h_w=[712, 712],
        target_crop_size_h_w=[512, 512],
        label_convert=[[0, 5], [1, 6], [2, 1], [3, 9], [4, 2], [5, 4], [6, 10], [7, 10], [8, 7], [9, 7], 
                       [10, 0], [11, 3], [12, 3], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8]],
    ),
    total_batch_size=2,
    num_workers=4,
)

dataloader.test = L(build_d2_test_dataloader)(
    dataset=L(CrossModalityDataset)(
        json_path='/data/json_file/Cityscapes_RGB_to_DELIVER_Depth_test.json',
        source_root_path='???',
        target_root_path='???',
        train_or_test='test',
        names='Cityscapes_RGB_to_DELIVER_Depth',
        test_resize_h_w=[512, 512],
        label_convert=[[0, 1], [1, 2], [2, 255], [3, 3], [4, 4], [5, 5], [6, 5], [7, 6], [8, 7], [9, 8],
                       [10, 9], [11, 10], [12, 0], [13, 255], [14, 255], [15, 255], [16, 255], [17, 10], [18, 255],
                       [19, 255], [20, 255], [21, 7], [22, 8], [23, 8], [24, 8]],
    ),
    local_batch_size=1,
    num_workers=4,
)

dataloader.evaluator = [
    L(DSECSemSegEvaluator)(
        dataset_name="${...test.dataset.names}",
        stuff_classes=['sky', 'building', 'fence', 'person', 'pole', 'road',
                       'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign'],
        palette=[70, 130, 180, 70, 70, 70, 190, 153, 153, 220, 20, 60, 153, 153, 153, 128, 64, 128, 244, 35, 232,
                 107, 142, 35, 0, 0, 142, 102, 102, 156, 250, 170, 30],
        ignore_label=255,
        output_dir=None,  # modify in EvalHook (do_test)
        save_predictions_json=False,
        save_eval_results_step=10,
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
