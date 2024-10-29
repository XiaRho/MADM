# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import numpy as np
import torch
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import cv2

Cityscapes_palette = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153,
    153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130,
    180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
    0, 0, 230, 119, 11, 32
]

DSEC_palette = [70, 130, 180, 70, 70, 70, 190, 153, 153, 220, 20, 60, 153, 153, 153, 128, 64, 128,
                244, 35, 232, 107, 142, 35, 0, 0, 142, 102, 102, 156, 250, 170, 30]


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def subplotimg(ax,
               img,
               title,
               range_in_title=False,
               palette=None,
               norm_mean=None,
               norm_std=None,
               heat_map=False,
               **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
            if norm_std is not None and norm_mean is not None:
                for i in range(3):
                    img[:, :, i] = img[:, :, i] * norm_std[0, i, 0, 0].cpu() + norm_mean[0, i, 0, 0].cpu()

        if palette is not None:
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette)

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    if heat_map:
        img = cv2.applyColorMap(np.uint8(255 * img), cv2.COLORMAP_JET)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

    ax.imshow(img, **kwargs)
    ax.set_title(title)

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def show_image_attention_maps(image_attention_maps, image, relevnace_res=16, attention_norm=False):
    if torch.is_tensor(image):
        image = Image.fromarray(np.transpose(np.uint8(image.cpu().numpy() * 255), (1, 2, 0)))

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_attention_maps = image_attention_maps.reshape(1, 1, image_attention_maps.shape[-1], image_attention_maps.shape[-1])
    image_attention_maps = image_attention_maps.cuda() # because float16 precision interpolation is not supported on cpu
    image_attention_maps = torch.nn.functional.interpolate(image_attention_maps, size=relevnace_res ** 2, mode='bilinear')
    image_attention_maps = image_attention_maps.cpu() # send it back to cpu
    if attention_norm:
        image_attention_maps = (image_attention_maps - image_attention_maps.min()) / (image_attention_maps.max() - image_attention_maps.min())
    image_attention_maps = image_attention_maps.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_attention_maps)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)  # np.array-np.uint8-[H, W, 3]: 0~255
    vis = np.transpose(vis, (2, 0, 1))
    vis = np.float32(vis) / 255  # np.array-np.float32-[3, H, W]: 0.0~1.0
    return torch.from_numpy(vis)  

