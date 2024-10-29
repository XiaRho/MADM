from detectron2.evaluation.evaluator import DatasetEvaluator
from tabulate import tabulate
import wandb

import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import omegaconf
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize, get_local_rank
from detectron2.utils.file_io import PathManager
from utils.visualization import subplotimg
from matplotlib import pyplot as plt


class DSECSemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self, *, dataset_name, stuff_classes, palette, ignore_label, prefix="", distributed=True,
                 output_dir=None, save_predictions_json=True, save_eval_results_step=-1, convert_pred_list=None,
                 eval_only=False, **kwargs):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """

        self.save_predictions_json = save_predictions_json
        self.save_eval_results_step = save_eval_results_step
        self.eval_index = 0
        self.convert_pred_list = convert_pred_list
        self.eval_only = eval_only

        self._logger = logging.getLogger(__name__)

        # self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        # self.input_file_to_gt_file = {
        #     dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
        #     for dataset_record in DatasetCatalog.get(dataset_name)
        # }

        # meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        # try:
        #     c2d = meta['stuff_dataset_id_to_contiguous_id']
        #     self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        # except AttributeError or omegaconf.errors.ConfigKeyError:
        #     self._contiguous_id_to_dataset_id = None
        self._contiguous_id_to_dataset_id = None
        self._class_names = stuff_classes
        self._num_classes = len(stuff_classes)
        self._palette = palette
        assert len(self._palette) == 3 * self._num_classes
        self._ignore_label = ignore_label

        self.dataset_name = dataset_name
        if len(prefix) and not prefix.endswith("_"):
            prefix += "_"
        self.prefix = prefix

        if 'target_modality' in kwargs.keys():
            self.target_modality = kwargs['target_modality']
        else:
            self.target_modality = ['default']

    def reset(self):
        keys = self.target_modality
        self._conf_matrix, self._predictions = dict(), dict()
        for key in keys:
            self._conf_matrix[key] = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
            self._predictions[key] = []
        os.makedirs(self._output_dir, exist_ok=True) if self._output_dir is not None else None

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for data, output in zip(inputs, outputs):

            if self.target_modality is not None and 'modality_type' in data.keys():
                modality_type = data['modality_type']
            else:
                modality_type = 'default'

            output = output["sem_seg"][0].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int32)
            if self.convert_pred_list is not None:
                converted_pred = np.copy(pred)
                for old_id, new_id in self.convert_pred_list:
                    converted_pred[pred == old_id] = new_id
                pred = converted_pred

            # with PathManager.open(self.input_file_to_gt_file[data["file_name"]], "rb") as f:
            if 'target_label' in data.keys():
                gt = np.int32(np.array(data["target_label"]))
                if gt.shape[0] == 1:
                    gt = gt[0]
            else:
                with PathManager.open(data["file_name"], "rb") as f:
                    gt = np.array(Image.open(f), dtype=np.int32)  # [H, W] np.int32
            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix[modality_type] += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix[modality_type].size,
            ).reshape(self._conf_matrix[modality_type].shape)

            if self.save_predictions_json:
                self._predictions[modality_type].extend(self.encode_json_sem_seg(pred, data["file_name"]))
            if self.save_eval_results_step != -1 and self.eval_index % self.save_eval_results_step == 0:
                self.save_vis_results(image=data['target_second_modality'], pred=pred, gt=gt, save_name=data['pred_save_name'])
            self.eval_index += 1
            # if self.eval_index == 10 - 1:
            #     break

    def save_vis_results(self, image, pred, gt, save_name=None):
        if not self.eval_only:
            rows, cols = 1, 3
            _, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False,
                                  gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0})

            image = image / 255

            subplotimg(axs[0][0], image, title='image')
            subplotimg(axs[0][1], pred, title='pred', palette=self._palette)
            subplotimg(axs[0][2], gt, title='gt', palette=self._palette)

            for ax in axs.flat:
                ax.axis('off')
            if save_name is None:
                plt.savefig(os.path.join(self._output_dir, '{:06d}_rank{}.png'.format(self.eval_index, get_local_rank())))
            else:
                plt.savefig(os.path.join(self._output_dir, save_name))
            plt.close()
        else:
            image_path = os.path.join(self._output_dir, 'image')
            pred_path = os.path.join(self._output_dir, 'pred')
            pred_color_path = os.path.join(self._output_dir, 'pred_color')
            gt_path = os.path.join(self._output_dir, 'gt')

            os.makedirs(image_path, exist_ok=True)
            os.makedirs(pred_path, exist_ok=True)
            os.makedirs(pred_color_path, exist_ok=True)
            os.makedirs(gt_path, exist_ok=True)

            save_name = '{:06d}_rank{}.png'.format(self.eval_index, get_local_rank())

            image = image.cpu().numpy()  # [3, 512, 512]
            image = np.transpose(np.uint8(image), (1, 2, 0))  # [512, 512, 3] uint8
            Image.fromarray(image).save(os.path.join(image_path, save_name))

            Image.fromarray(pred).save(os.path.join(pred_path, save_name))  # pred = [512, 512]

            pred = Image.fromarray(pred.astype(np.uint8)).convert('P')
            pred.putpalette(self._palette)
            pred = pred.convert('RGB')
            pred.save(os.path.join(pred_color_path, save_name))

            gt = Image.fromarray(gt.astype(np.uint8)).convert('P')
            gt.putpalette(self._palette)
            gt = gt.convert('RGB')
            gt.save(os.path.join(gt_path, save_name))

    def evaluate(self):
        results = self.ori_evaluate()
        if results is None:
            return
        
        prefix_results = dict()
        for key in results.keys():
            results_per_category = []
            for name in self._class_names:
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                results_per_category.append((str(name), float(results[key]["sem_seg_{}".format(key)][f"IoU-{name}"])))

            # tabulate it
            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "IoU"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category IoU {}: \n".format(key) + table)
            # print("Per-category IoU: \n" + table)
            prefix_results[key] = OrderedDict()
            for k, v in results[key].items():
                prefix_results[key][f"{self.dataset_name}/{self.prefix}{k}"] = v

        return prefix_results

    def ori_evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        results = dict()
        
        # if self._distributed:
        #     synchronize()
        #     for key in self.target_modality:
        #         conf_matrix_list = all_gather(self._conf_matrix[key])
        #         self._predictions[key] = all_gather(self._predictions[key])
        #         self._predictions[key] = list(itertools.chain(*self._predictions[key]))
        #         if not is_main_process():
        #             return
        #         self._conf_matrix[key] = np.zeros_like(self._conf_matrix[key])
        #         for conf_matrix in conf_matrix_list:
        #             self._conf_matrix[key] += conf_matrix

        for key in self.target_modality:
            if self._output_dir and self.save_predictions_json:
                # PathManager.mkdirs(self._output_dir)
                file_path = os.path.join(self._output_dir, "sem_seg_{}_predictions.json".format(key))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(self._predictions[key]))
            acc = np.full(self._num_classes, np.nan, dtype=np.float)
            iou = np.full(self._num_classes, np.nan, dtype=np.float)
            tp = self._conf_matrix[key].diagonal()[:-1].astype(np.float)
            pos_gt = np.sum(self._conf_matrix[key][:-1, :-1], axis=0).astype(np.float)
            class_weights = pos_gt / np.sum(pos_gt)
            pos_pred = np.sum(self._conf_matrix[key][:-1, :-1], axis=1).astype(np.float)
            acc_valid = pos_gt > 0
            acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
            iou_valid = (pos_gt + pos_pred) > 0
            union = pos_gt + pos_pred - tp
            iou[acc_valid] = tp[acc_valid] / union[acc_valid]
            macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
            miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
            fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
            pacc = np.sum(tp) / np.sum(pos_gt)

            res = {}
            res["mIoU"] = 100 * miou
            res["fwIoU"] = 100 * fiou
            for i, name in enumerate(self._class_names):
                res["IoU-{}".format(name)] = 100 * iou[i]
            res["mACC"] = 100 * macc
            res["pACC"] = 100 * pacc
            for i, name in enumerate(self._class_names):
                res["ACC-{}".format(name)] = 100 * acc[i]

            if self._output_dir:
                file_path = os.path.join(self._output_dir, "sem_seg_{}_evaluation.pth".format(key))
                with PathManager.open(file_path, "wb") as f:
                    torch.save(res, f)
            results[key] = OrderedDict({"sem_seg_{}".format(key): res})
            self._logger.info(results[key])

        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

