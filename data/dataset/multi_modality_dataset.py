from PIL import Image
import random
import numpy as np
import torchvision.transforms as standard_transforms
import torch
from torch.utils import data
import json
import os
import logging
import torch.nn as nn
from data.dataset.cross_modality_dataset import get_rcs_class_probs


class MultiModalityDataset(torch.utils.data.Dataset):

    rcs_class_temp = 0.01
    rcs_min_crop_ratio = 0.5
    rcs_min_pixels = 3000

    def __init__(self, json_path, source_root_path, target_root_path, target_modality, source_resize_h_w=None, source_crop_size_h_w=None,
                 target_resize_h_w=None, target_crop_size_h_w=None, test_resize_h_w=None, train_or_test='train',
                 label_convert=None, rare_class_sample=False, **kwargs):

        self.source_resize_h_w = [0, 0] if source_resize_h_w is None else source_resize_h_w
        self.source_crop_size_h_w = [0, 0] if source_crop_size_h_w is None else source_crop_size_h_w
        self.target_modality = target_modality

        self.target_resize_h_w = dict()
        self.target_crop_size_h_w = dict()
        self.target_root_path = dict()
        self.test_resize_h_w = dict()
        if train_or_test == 'train':
            self.label_convert = label_convert
        else:
            self.label_convert = dict()
        for i, modal in enumerate(self.target_modality):
            if target_resize_h_w is not None:
                self.target_resize_h_w[modal] = target_resize_h_w[i]
            if target_crop_size_h_w is not None:
                self.target_crop_size_h_w[modal] = target_crop_size_h_w[i]
            if test_resize_h_w is not None:
                self.test_resize_h_w[modal] = test_resize_h_w[i]
            self.target_root_path[modal] = target_root_path[i]
            if train_or_test == 'test':
                self.label_convert[modal] = label_convert[i]

        self.json_path = json_path
        assert 'Cityscapes_RGB_to_Depth_and_Event' in self.json_path or \
               'XXXXXXX' in self.json_path

        self.source_root_path = source_root_path
        self.train_or_test = train_or_test
        assert self.train_or_test in {'train', 'test'}
        
        self.rare_class_sample = rare_class_sample

        with open(json_path) as f:
            self.json = json.load(f)

        if self.train_or_test == 'train':
            self.source_data_length = len(self.json['source_data']['RGB'])
        else:
            self.source_data_length = 1

        self.target_data_length = len(self.json['target_data']['second_modality'])
        assert sorted(list(self.json['target_split'].keys())) == sorted(target_modality)

        self.target_data_remain = dict()
        for modality in self.json['target_split'].keys():
            self.target_data_remain[modality] = [i for i in range(self.json['target_split'][modality][0], self.json['target_split'][modality][1] + 1)]

        self.to_tensor_transform = standard_transforms.Compose([standard_transforms.ToTensor()])
        self.HorizontalFlip = standard_transforms.RandomHorizontalFlip(p=1)

        if self.rare_class_sample:
            self.logger = logging.getLogger("odise")
            self.init_rare_class_sample()
        # self.source_data_length, self.target_data_length = 1000, 1000

    def __len__(self):
        return self.source_data_length * self.target_data_length

    def init_rare_class_sample(self):
        self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(self.source_root_path, self.rcs_class_temp)
        self.logger.info(f'RCS Classes: {self.rcs_classes}')
        self.logger.info(f'RCS ClassProb: {self.rcs_classprob}')
        # print(f'RCS Classes: {self.rcs_classes}')
        # print(f'RCS ClassProb: {self.rcs_classprob}')
        with open(os.path.join(self.source_root_path, 'samples_with_class.json'), 'r') as of:
            samples_with_class_and_n = json.load(of)
        samples_with_class_and_n = {
            int(k): v
            for k, v in samples_with_class_and_n.items()
            if int(k) in self.rcs_classes
        }
        self.samples_with_class = {}
        for c in self.rcs_classes:
            self.samples_with_class[c] = []
            for file, pixels in samples_with_class_and_n[c]:
                if pixels > self.rcs_min_pixels:
                    self.samples_with_class[c].append(file.split('/')[-1])
            assert len(self.samples_with_class[c]) > 0
        self.file_to_idx = {}
        for i, label_name in enumerate(self.json['source_data']['label']):
            self.file_to_idx[label_name.split('/')[-1]] = i

    def get_source_data(self, source_idx):
        flip_flag = True if random.random() < 0.5 else False
        x = random.randint(0, self.source_resize_h_w[1] - self.source_crop_size_h_w[1])
        y = random.randint(0, self.source_resize_h_w[0] - self.source_crop_size_h_w[0])

        source_rgb_path = os.path.join(self.source_root_path, self.json['source_data']['RGB'][source_idx])
        source_label_path = os.path.join(self.source_root_path, self.json['source_data']['label'][source_idx])

        source_rgb_tensor = self.load_aug_data(source_rgb_path, self.source_resize_h_w[1], self.source_resize_h_w[0],
                                               self.source_crop_size_h_w[1], self.source_crop_size_h_w[0], x, y,
                                               flip_flag, data_or_label='data')

        source_label_tensor = self.load_aug_data(source_label_path, self.source_resize_h_w[1],
                                                 self.source_resize_h_w[0],
                                                 self.source_crop_size_h_w[1], self.source_crop_size_h_w[0], x, y,
                                                 flip_flag, data_or_label='label')
        
        return {'rgb': source_rgb_tensor, 'label': source_label_tensor}

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.get_source_data(source_idx=i1)
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['label'] == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = self.get_source_data(source_idx=i1)
        return s1
    

    def load_aug_data(self, data_path, resize_width=0, resize_height=0, crop_width=0, crop_height=0, random_x=0,
                      random_y=0, flip_flag=False, data_or_label='data', test_modality_type=None):
        
        data_pil = Image.open(data_path)

        if self.train_or_test == 'train':
            resample_type = Image.BILINEAR if data_or_label == 'data' else Image.NEAREST
            if 'dsec' in data_path and data_pil.size == (640, 480):
                data_pil = data_pil.crop(box=(0, 0, 640, 440))
            data_pil = data_pil.resize(size=(resize_width, resize_height), resample=resample_type)
            data_pil = data_pil.crop(box=(random_x, random_y, random_x + crop_width, random_y + crop_height))
            if flip_flag:
                data_pil = self.HorizontalFlip(data_pil)
        elif self.test_resize_h_w[test_modality_type] is not None:
            resample_type = Image.BILINEAR if data_or_label == 'data' else Image.NEAREST
            data_pil = data_pil.resize(
                size=(self.test_resize_h_w[test_modality_type][1], self.test_resize_h_w[test_modality_type][0]), 
                resample=resample_type
            )

        # data_tensor = self.to_tensor_transform(data_pil) * 255
        data_np = np.array(data_pil)
        if len(data_np.shape) == 2:
            data_np = data_np[None]
        else:
            data_np = np.transpose(data_np, (2, 0, 1))
        if data_or_label == 'data':
            if data_np.shape[0] == 4:
                data_np = data_np[:3]
            elif data_np.shape[0] == 1:
                data_np = np.repeat(data_np, 3, axis=0)
            data_np = np.float32(data_np)
        else:
            if 'DELIVER' in data_path:
                data_np = data_np[0:1]
        data_tensor = torch.from_numpy(data_np)

        if data_or_label == 'label':
            if 'DELIVER' in data_path:
                mask = (data_tensor == 255)
                data_tensor -= 1
                data_tensor[mask] = 255
            data_tensor = data_tensor.type(torch.long)

        return data_tensor

    def convert_label(self, label_tensor, label_convert):
        _label_tensor = torch.clone(label_tensor)
        for old_id, new_id in label_convert:
            _label_tensor[label_tensor == old_id] = new_id
        return _label_tensor

    def __getitem__(self, idx):
        """
        :param idx: index
        :return: Dict{'source_rgb'/'target_rgb'/'target_second_modality': [3, 512, 512]  torch.float32  0~255,
                      'source_label': [1, 512, 512]  torch.int64_0  (num_classes-1)+255,
                      'width': W, 'height': H}
        """
        source_idx = idx % self.source_data_length
        target_idx = idx % self.target_data_length

        if self.train_or_test == 'train':

            if self.rare_class_sample:
                source_data = self.get_rare_class_sample()
            else:
                source_data = self.get_source_data(source_idx=source_idx)
            source_rgb_tensor, source_label_tensor = source_data['rgb'], source_data['label']

            if self.label_convert is not None:
                source_label_tensor = self.convert_label(source_label_tensor, label_convert=self.label_convert)

            target_second_modality = {}
            for modality in self.json['target_split'].keys():
                # select from remain samples
                target_idx = random.choice(self.target_data_remain[modality])
                self.target_data_remain[modality].remove(target_idx)
                if len(self.target_data_remain[modality]) == 0:
                    self.target_data_remain[modality] = [i for i in range(self.json['target_split'][modality][0], self.json['target_split'][modality][1] + 1)]

                flip_flag = True if random.random() < 0.5 else False
                x = random.randint(0, self.target_resize_h_w[modality][1] - self.target_crop_size_h_w[modality][1])
                y = random.randint(0, self.target_resize_h_w[modality][0] - self.target_crop_size_h_w[modality][0])

                target_second_modality_path = os.path.join(self.target_root_path[modality], self.json['target_data']['second_modality'][target_idx])

                target_second_modality[modality] = self.load_aug_data(
                    target_second_modality_path, self.target_resize_h_w[modality][1], self.target_resize_h_w[modality][0],
                    self.target_crop_size_h_w[modality][1], self.target_crop_size_h_w[modality][0], x, y, flip_flag,
                    data_or_label='data',
                )

            output_dict = {
                'source_rgb': source_rgb_tensor, 'source_label': source_label_tensor,
                'target_second_modality': target_second_modality,
                'width': self.target_crop_size_h_w[modality][1], 'height': self.target_crop_size_h_w[modality][0]
            }
            # output_dict = {'source_rgb': 1, 'width': 512, 'height': 512}
            return output_dict
        else:
            modality_type = None
            for modality in self.json['target_split'].keys():
                _min, _max = self.json['target_split'][modality]
                if _min <= target_idx <= _max:
                    modality_type = modality
                    break
            if modality_type is None:
                raise ValueError('target_idx: {} not in {}'.format(target_idx, self.json['target_split']))
                
            target_second_modality_path = os.path.join(self.target_root_path[modality], self.json['target_data']['second_modality'][target_idx])
            target_label_path = os.path.join(self.target_root_path[modality], self.json['target_data']['label'][target_idx])

            target_second_modality_tensor = self.load_aug_data(
                target_second_modality_path,
                data_or_label='data',
                test_modality_type=modality_type
            )

            output = {'target_second_modality': target_second_modality_tensor, 'file_name': target_label_path, 'modality_type': modality_type,
                    'width': target_second_modality_tensor.shape[-2], 'height': target_second_modality_tensor.shape[-1]}
            
            if self.test_resize_h_w[modality_type] is not None:
                target_label_tensor = self.load_aug_data(target_label_path, data_or_label='label', test_modality_type=modality_type)
                output['target_label'] = target_label_tensor

                if self.label_convert[modality_type] is not None:
                    output['target_label'] = self.convert_label(output['target_label'], label_convert=self.label_convert[modality_type])
                output['target_label'] = output['target_label']

            words = self.json['target_data']['label'][target_idx].split('/')
            if 'DELIVER' in target_second_modality_path:
                output['pred_save_name'] = '{}_{}_{}_{}'.format(words[-4], words[-3], words[-2], words[-1])
            elif 'dsec' in target_second_modality_path:
                output['pred_save_name'] = '{}_{}'.format(words[-3], words[-1])
            else:
                raise NotImplementedError('pred_save_name in {}'.format(target_second_modality_path))
            
            return output
