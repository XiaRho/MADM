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


def remove_array_amp(img_np, L, fusion_val=None):
    '''
    :param img_np: np.array [3, 512, 512]  0~255 float
    :param L: 0.005 range of pixel
    :return: np.array [3, 512, 512]
    '''
    fft_img_np = np.fft.fft2(img_np, axes=(-2, -1))
    amp_img, pha_img = np.abs(fft_img_np), np.angle(fft_img_np)

    # mask some amp area
    amp_img = np.fft.fftshift(amp_img, axes=(-2, -1))
    _, h, w = amp_img.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    amp_img[:, h1:h2, w1:w2] = 0
    amp_img = np.fft.ifftshift(amp_img, axes=(-2, -1))

    # generate new img
    new_img_np = amp_img * np.exp(1j * pha_img)
    new_img_np = np.fft.ifft2(new_img_np, axes=(-2, -1))
    new_img_np = np.real(new_img_np)

    # scale the range
    _min, _max = np.min(new_img_np), np.max(new_img_np)
    new_img_np = (new_img_np - _min) / (_max - _min) * 255

    # fusion with ori_img
    if fusion_val is not None:
        new_img_np = fusion_val * new_img_np + (1 - fusion_val) * img_np
    return new_img_np


def remove_tensor_amp(img_tensor, L, fusion_val=None):
    '''
    :param img_tensor: torch.tensor [3, 512, 512]  0~255 float
    :param L: 0.005 range of pixel
    :return: torch.tensor [3, 512, 512]
    '''
    fft_img_tensor = torch.fft.fft2(img_tensor, dim=(-2, -1))
    amp_img, pha_img = torch.abs(fft_img_tensor), torch.angle(fft_img_tensor)

    # mask some amp area
    amp_img = torch.fft.fftshift(amp_img, dim=(-2, -1))
    _, h, w = amp_img.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    amp_img[:, h1:h2, w1:w2] = 0
    amp_img = torch.fft.ifftshift(amp_img, dim=(-2, -1))

    # generate new img
    new_img_tensor = amp_img * torch.exp(1j * pha_img)
    new_img_tensor = torch.fft.ifft2(new_img_tensor, dim=(-2, -1))
    new_img_tensor = torch.real(new_img_tensor)

    # scale the range
    _min, _max = torch.min(new_img_tensor), torch.max(new_img_tensor)
    new_img_tensor = (new_img_tensor - _min) / (_max - _min) * 255

    # fusion with ori_img
    if fusion_val is not None:
        new_img_tensor = fusion_val * new_img_tensor + (1 - fusion_val) * img_tensor
    return new_img_tensor


def get_rcs_class_probs(data_root, temperature):
    with open(os.path.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@torch.no_grad()
def remove_batch_tensor_amp(img_tensor, L_range, fusion_val=None):
    '''
    :param img_tensor: torch.tensor [B, 3, 512, 512]  0~1 float
    :param L_range: 0.01~0.1 range of pixel
    :return: torch.tensor [B, 3, 512, 512]
    '''
    assert len(img_tensor.shape) == 4
    bs = img_tensor.shape[0]
    new_img_tensor = []
    for i in range(bs):
        img_tensor_ = img_tensor[i] * 255
        img_tensor_ = remove_tensor_amp(img_tensor_, L=random.uniform(L_range[0], L_range[1]), fusion_val=fusion_val)
        new_img_tensor.append(img_tensor_ / 255)
    return torch.stack(new_img_tensor)


class Diff(nn.Module):
    def __init__(self):
        super(Diff, self).__init__()
        diff_kernels = [
            torch.tensor([[3, -1],
                          [-1, -1]], dtype=torch.float)]
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, padding=1, bias=False, padding_mode='reflect')
        with torch.no_grad():
            for i, kernel in enumerate(diff_kernels):
                self.conv1.weight[i].copy_(kernel.unsqueeze(0))

    @torch.no_grad()
    def forward(self, x):
        return self.conv1(x)[:, :, :-1, :-1]


class CrossModalityDataset(torch.utils.data.Dataset):

    rcs_class_temp = 0.01
    rcs_min_crop_ratio = 0.5
    rcs_min_pixels = 3000
    edges_log_add = -1  # 0.01
    edges_min_clip = 0.02  # 0.01
    edges_max_clip = 0.95
    local_region_num = 10

    def __init__(self, json_path, source_root_path, target_root_path, source_resize_h_w=None, source_crop_size_h_w=None,
                 target_resize_h_w=None, target_crop_size_h_w=None, test_resize_h_w=None, train_or_test='train',
                 label_convert=None, remove_amp=None, fda_fusion_val=None, rare_class_sample=False, remove_texture=False,
                 merge_more_target_data=None, pl_data_path=None,
                 **kwargs):

        self.source_resize_h_w = [0, 0] if source_resize_h_w is None else source_resize_h_w
        self.source_crop_size_h_w = [0, 0] if source_crop_size_h_w is None else source_crop_size_h_w
        self.target_resize_h_w = [0, 0] if target_resize_h_w is None else target_resize_h_w
        self.target_crop_size_h_w = [0, 0] if target_crop_size_h_w is None else target_crop_size_h_w
        self.test_resize_h_w = test_resize_h_w

        self.merge_more_target_data = merge_more_target_data
        self.pl_data_path = pl_data_path
        
        self.json_path = json_path
        assert 'DELIVER_RGB2Depth' in self.json_path or \
               'DELIVER_Depth2RGB' in self.json_path or \
               'DSEC_RGB2REvents' in self.json_path or \
               'cityscapes_dsec' in self.json_path or \
               'Cityscapes_RGB_to_DELIVER_Depth' in self.json_path or \
               'Cityscapes_RGB_to_DSEC_Event' in self.json_path or \
               'Cityscapes_RGB_to_DSEC_19_Event' in self.json_path or \
               'Cityscapes_RGB_to_VKITTI2_Depth' in self.json_path or \
               'GTA5_RGB_to_Cityscapes_RGB' in self.json_path or \
               'GTA5_RGB_to_DELIVER_Depth' in self.json_path or \
               'Cityscapes_RGB_to_FMB_Infrared' in self.json_path

        # labels in deliver need to be need handled additionally
        self.deliver_label_process = False
        if 'DELIVER_RGB2Depth' in self.json_path or 'DELIVER_Depth2RGB' in self.json_path:
            self.deliver_label_process = True
        elif 'to_DELIVER_Depth' in self.json_path and train_or_test == 'test':
            self.deliver_label_process = True

        self.source_root_path = source_root_path
        self.target_root_path = target_root_path
        self.train_or_test = train_or_test
        assert self.train_or_test in {'train', 'test'}
        self.label_convert = label_convert
        self.remove_amp = remove_amp
        self.fda_fusion_val = fda_fusion_val
        self.rare_class_sample = rare_class_sample
        self.remove_texture = remove_texture
        assert not (self.remove_amp and self.remove_texture)
        if self.remove_amp is not None:
            self.remove_amp = list(self.remove_amp)
            assert isinstance(self.remove_amp, list) and len(self.remove_amp) == 2
        if self.fda_fusion_val is not None:
            self.fda_fusion_val = list(self.fda_fusion_val)
            assert isinstance(self.fda_fusion_val, list) and len(self.fda_fusion_val) == 2
        if self.remove_texture:
            self.edge_conv = Diff()

        with open(json_path) as f:
            self.json = json.load(f)

        if self.train_or_test == 'train':
            source_image_sample = Image.open(os.path.join(self.source_root_path, self.json['source_data']['RGB'][0]))
            self.source_data_width = source_image_sample.width
            self.source_data_height = source_image_sample.height
            self.source_data_length = len(self.json['source_data']['RGB'])
        else:
            self.source_data_length = 1

        target_image_sample = Image.open(self.target_root_path + self.json['target_data']['second_modality'][0])
        self.target_data_width = target_image_sample.width
        self.target_data_height = target_image_sample.height
        if self.merge_more_target_data is not None:
            assert 'Cityscapes_RGB_to_DELIVER_Depth' in self.json_path
            new_target_path = os.path.join(self.target_root_path, self.merge_more_target_data)
            new_images = os.listdir(new_target_path)
            for image_name in new_images:
                self.json['target_data']['second_modality'].append(os.path.join(self.merge_more_target_data, image_name))

        self.target_data_length = len(self.json['target_data']['second_modality'])

        self.to_tensor_transform = standard_transforms.Compose([standard_transforms.ToTensor()])
        self.HorizontalFlip = standard_transforms.RandomHorizontalFlip(p=1)

        if self.rare_class_sample:
            self.logger = logging.getLogger("odise")
            self.init_rare_class_sample()

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
                                               flip_flag, data_or_label='data', remove_amp=False)
        
        if self.pl_data_path is not None:
            source_pl_data_path = os.path.join(self.pl_data_path, self.json['source_data']['label'][source_idx].split('gtFine/train/')[-1])
            source_pl_data_tensor = self.load_aug_data(source_pl_data_path, self.source_resize_h_w[1], self.source_resize_h_w[0],
                                                       self.source_crop_size_h_w[1], self.source_crop_size_h_w[0], x, y,
                                                       flip_flag, data_or_label='data', remove_amp=False)
        else:
            source_pl_data_tensor = None

        # random remove amp
        if self.remove_amp is not None:
            source_rgb_pha_tensor = self.load_aug_data(source_rgb_path, self.source_resize_h_w[1],
                                                       self.source_resize_h_w[0],
                                                       self.source_crop_size_h_w[1], self.source_crop_size_h_w[0], x, y,
                                                       flip_flag, data_or_label='data', remove_amp=True)
        else:
            source_rgb_pha_tensor = None

        source_label_tensor = self.load_aug_data(source_label_path, self.source_resize_h_w[1],
                                                 self.source_resize_h_w[0],
                                                 self.source_crop_size_h_w[1], self.source_crop_size_h_w[0], x, y,
                                                 flip_flag, data_or_label='label')
        
        return {'rgb': source_rgb_tensor, 'label': source_label_tensor, 'rgb_pha': source_rgb_pha_tensor, 'pl_data': source_pl_data_tensor}

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

    def extract_edge_info(self, img_tensor):
        img_tensor = img_tensor / 255
        img_tensor = torch.mean(img_tensor, dim=0, keepdim=True)
        # img_tensor = torch.log(img_tensor + edges_log_add)
        img_tensor = self.edge_conv(img_tensor[None])[0]
        img_tensor[torch.abs(img_tensor) < self.edges_min_clip] = 0
        if torch.numel(img_tensor[img_tensor > 0]) != 0:
            if self.edges_max_clip == 1:
                max_clip_thres = torch.max(img_tensor)
            else:
                max_clip_thres = torch.quantile(img_tensor[img_tensor > 0], self.edges_max_clip)
            img_tensor = torch.clamp(img_tensor, min=-max_clip_thres, max=max_clip_thres) / max_clip_thres
        else:
            img_tensor[:] = 0
        img_tensor = (img_tensor + 1) * 127.5
        return img_tensor.repeat(3, 1, 1)

    def extract_edge_info_local(self, img_tensor):
        _, h, w = img_tensor.shape
        h_step = h / self.local_region_num
        w_step = w / self.local_region_num
        l_r_u_d = []
        for x in range(self.local_region_num):
            start_x, end_x = round(x * w_step), round((x + 1) * w_step)
            for y in range(self.local_region_num):
                start_y, end_y = round(y * h_step), round((y + 1) * h_step)
                l_r_u_d.append((start_x, end_x, start_y, end_y))

        for left, right, upper, down in l_r_u_d:
            img_tensor[:, upper: down, left: right] = self.extract_edge_info(img_tensor[:, upper: down, left: right])
        return img_tensor

    def load_aug_data(self, data_path, resize_width=0, resize_height=0, crop_width=0, crop_height=0, random_x=0,
                      random_y=0, flip_flag=False, data_or_label='data', remove_amp=False, remove_texture=False):
        data_pil = Image.open(data_path)
        if self.remove_amp is not None and data_or_label == 'data':
            data_pil = data_pil.convert('L').convert('RGB')

        if self.train_or_test == 'train':
            resample_type = Image.BILINEAR if data_or_label == 'data' else Image.NEAREST
            if 'DSEC_RGB' in self.json_path and data_pil.size == (640, 480):
                data_pil = data_pil.crop(box=(0, 0, 640, 440))
            data_pil = data_pil.resize(size=(resize_width, resize_height), resample=resample_type)
            data_pil = data_pil.crop(box=(random_x, random_y, random_x + crop_width, random_y + crop_height))
            if flip_flag:
                data_pil = self.HorizontalFlip(data_pil)
        elif self.test_resize_h_w is not None:
            resample_type = Image.BILINEAR if data_or_label == 'data' else Image.NEAREST
            data_pil = data_pil.resize(size=(self.test_resize_h_w[1], self.test_resize_h_w[0]), resample=resample_type)

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
            # FDA
            if remove_amp is not False:
                if remove_amp is True:
                    if self.train_or_test == 'train':
                        remove_range = random.uniform(self.remove_amp[0], self.remove_amp[1])
                    else:
                        remove_range = (self.remove_amp[0] + self.remove_amp[1]) / 2
                else:
                    assert isinstance(remove_amp, float)
                    remove_range = remove_amp
                if self.fda_fusion_val is not None:
                    if self.train_or_test == 'train':
                        fda_fusion_val = random.uniform(self.fda_fusion_val[0], self.fda_fusion_val[1])
                    else:
                        fda_fusion_val = (self.fda_fusion_val[0] + self.fda_fusion_val[1]) / 2
                else:
                    fda_fusion_val = None
                data_np = np.float32(remove_array_amp(data_np, L=remove_range, fusion_val=fda_fusion_val))
        else:
            if self.deliver_label_process:
                data_np = data_np[0:1]
        data_tensor = torch.from_numpy(data_np)

        if data_or_label == 'label':
            if self.deliver_label_process:
                mask = (data_tensor == 255)
                data_tensor -= 1
                data_tensor[mask] = 255
            data_tensor = data_tensor.type(torch.long)

        if data_or_label == 'data' and remove_texture:
            data_tensor = self.extract_edge_info_local(data_tensor)

        return data_tensor

    def convert_label(self, label_tensor):
        _label_tensor = torch.clone(label_tensor)
        for old_id, new_id in self.label_convert:
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
            source_rgb_tensor, source_label_tensor, source_rgb_pha_tensor = source_data['rgb'], source_data['label'], source_data['rgb_pha']
            if self.pl_data_path is not None:
                source_pl_data_tensor = source_data['pl_data']

            flip_flag = True if random.random() < 0.5 else False
            x = random.randint(0, self.target_resize_h_w[1] - self.target_crop_size_h_w[1])
            y = random.randint(0, self.target_resize_h_w[0] - self.target_crop_size_h_w[0])
            # target_rgb_path = self.target_root_path + self.json['target_data']['RGB'][target_idx]
            target_second_modality_path = os.path.join(self.target_root_path, self.json['target_data']['second_modality'][target_idx])

            target_second_modality_tensor = self.load_aug_data(
                target_second_modality_path, self.target_resize_h_w[1], self.target_resize_h_w[0],
                self.target_crop_size_h_w[1], self.target_crop_size_h_w[0], x, y, flip_flag,
                data_or_label='data', remove_amp=False
            )
            # random remove amp
            if self.remove_amp is not None:
                target_second_modality_pha_tensor = self.load_aug_data(
                    target_second_modality_path, self.target_resize_h_w[1], self.target_resize_h_w[0],
                    self.target_crop_size_h_w[1], self.target_crop_size_h_w[0], x, y, flip_flag,
                    data_or_label='data', remove_amp=True
                )
                source_rgb_pha_tensor += (torch.mean(target_second_modality_pha_tensor) - torch.mean(source_rgb_pha_tensor))
                source_rgb_pha_tensor = torch.clip(source_rgb_pha_tensor, 0, 255)

            if self.remove_texture:
                target_second_modality_pha_tensor = self.load_aug_data(
                    target_second_modality_path, self.target_resize_h_w[1], self.target_resize_h_w[0],
                    self.target_crop_size_h_w[1], self.target_crop_size_h_w[0], x, y, flip_flag,
                    data_or_label='data', remove_texture=True
                )

            if self.label_convert is not None:
                source_label_tensor = self.convert_label(source_label_tensor)

            output_dict = {
                'source_rgb': source_rgb_tensor, 'source_label': source_label_tensor,
                'target_second_modality': target_second_modality_tensor,
                'width': self.target_crop_size_h_w[1], 'height': self.target_crop_size_h_w[0]
            }
            if self.remove_amp is not None:
                output_dict['source_rgb_pha'] = source_rgb_pha_tensor
                output_dict['target_second_modality_pha'] = target_second_modality_pha_tensor
            elif self.remove_texture:
                output_dict['target_second_modality_pha'] = target_second_modality_pha_tensor
            elif self.pl_data_path is not None:
                output_dict['source_pl_data'] = source_pl_data_tensor

            return output_dict
        else:
            target_second_modality_path = os.path.join(self.target_root_path, self.json['target_data']['second_modality'][target_idx])
            target_label_path = os.path.join(self.target_root_path, self.json['target_data']['label'][target_idx])

            target_second_modality_tensor = self.load_aug_data(
                target_second_modality_path,
                data_or_label='data',
                remove_amp=False
            )

            output = {'target_second_modality': target_second_modality_tensor, 'file_name': target_label_path,
                      'width': target_second_modality_tensor.shape[-2], 'height': target_second_modality_tensor.shape[-1]}
            if self.test_resize_h_w is not None:
                target_label_tensor = self.load_aug_data(target_label_path, data_or_label='label')
                output['target_label'] = target_label_tensor

            if self.label_convert is not None:
                output['target_label'] = self.convert_label(output['target_label'])

            words = self.json['target_data']['label'][target_idx].split('/')
            if 'Cityscapes_RGB_to_DELIVER_Depth' in self.json_path or 'GTA5_RGB_to_DELIVER_Depth' in self.json_path:
                output['pred_save_name'] = '{}_{}_{}_{}'.format(words[-4], words[-3], words[-2], words[-1])
            elif 'Cityscapes_RGB_to_DSEC_Event' in self.json_path or 'Cityscapes_RGB_to_DSEC_19_Event' in self.json_path:
                output['pred_save_name'] = '{}_{}'.format(words[-3], words[-1])
            elif 'Cityscapes_RGB_to_VKITTI2_Depth' in self.json_path:
                output['pred_save_name'] = words[-1]
            elif 'GTA5_RGB_to_Cityscapes_RGB' in self.json_path:
                output['pred_save_name'] = words[-1]
            elif 'Cityscapes_RGB_to_FMB_Infrared' in self.json_path:
                output['pred_save_name'] = words[-1]
            else:
                raise NotImplementedError('pred_save_name in {}'.format(self.json_path))

            return output