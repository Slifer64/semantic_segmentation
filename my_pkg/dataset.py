import numpy as np
import os
import torch
from typing import List, Optional, Dict, Tuple
import copy
import my_pkg.data_transforms as data_tf
from my_pkg.data_types import *


# ================================================
# ============  SegmentationDataset  =============
# ================================================

class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, data_samples: List[str], data_transforms=None):
        if isinstance(data_transforms, list):
            data_transforms = data_tf.Compose(data_transforms)
        self.transforms = data_transforms
        self.samples = copy.deepcopy(data_samples)
        self.in_cfg = {'rgb.png': InputImage}
        self.out_cfg = {'mask.png': SegmentationMask}

    @classmethod
    def from_folder(cls, path: str, data_transforms=None):
        # load all image files, sorting them to ensure that they are aligned
        data_samples = [os.path.join(path, i) for i in list(sorted(os.listdir(path)))
                        if os.path.isdir(os.path.join(path, i))]
        return cls(data_samples=data_samples, data_transforms=data_transforms)

    def __call__(self, idx, return_raw=True):

        sample_path = self.samples[idx]

        inputs = {input_.data_name: input_.load(sample_path, filename) for filename, input_ in self.in_cfg.items()}
        outputs = {output_.data_name: output_.load(sample_path, filename) for filename, output_ in self.out_cfg.items()}

        if self.transforms is not None:
            inputs, outputs = self.transforms(inputs, outputs)

        # net_inputs = torch.concat([v.torch_tensor() for v in inputs.values()], dim=0)
        if return_raw:
            inputs = {name: input_.torch_tensor() for name, input_ in inputs.items()}
            outputs = {name: out.torch_tensor() for name, out in outputs.items()}

        return inputs, outputs

    def __getitem__(self, idx):
        return self(idx)

    def __len__(self):
        return len(self.samples)

    def subset(self, indices) -> 'SegmentationDataset':
        return SegmentationDataset(data_samples=[self.samples[i] for i in indices], data_transforms=self.transforms)


# ================================================
# ============  MaskRCNNDataset  =============
# ================================================

import cv2
from PIL import Image


def mask_rcnn_collate_fn(batch):
    batched_imgs = [img for img, _ in batch]
    batched_targets = [targets for _, targets in batch]
    return batched_imgs, batched_targets


class MaskRCNNDataset(torch.utils.data.Dataset):

    def __init__(self, data_samples: List[str]):
        self.samples = copy.deepcopy(data_samples)

    @classmethod
    def from_folder(cls, path: str):
        data_samples = [os.path.join(path, i) for i in list(sorted(os.listdir(path)))
                        if os.path.isdir(os.path.join(path, i))]
        return cls(data_samples=data_samples)

    def __call__(self, idx):

        sample_path = self.samples[idx]

        img_path = os.path.join(sample_path, 'rgb.png')
        mask_path = os.path.join(sample_path, 'mask.png')

        img = torch.as_tensor(cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32).permute(2, 0, 1)
        # img = torch.tensor(np.asanyarray(Image.open(img_path).convert("RGB")), dtype=torch.float32).permute(2, 0, 1)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        ids = np.unique(mask)[1:]

        bin_masks = (mask[None, ...] == ids[:, None, None]).astype(np.uint8)

        masks = []
        bbox = []
        labels = []
        area = []
        for id, m in zip(ids, bin_masks):
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                bbox.append([x, y, x+w, y+h])
                labels.append(id)
                area.append(h*w)
                masks.append(cv2.drawContours(np.zeros(img.shape[1:], dtype=np.uint8), [c], 0, color=1, thickness=-1)) # color=1 to get binary mask

        masks = np.stack(masks, axis=0)

        num_objs = len(bbox)

        target = {}
        target["boxes"] = torch.as_tensor(bbox, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        return img, target


    def __getitem__(self, idx):
        return self(idx)

    def __len__(self):
        return len(self.samples)

    def subset(self, indices) -> 'MaskRCNNDataset':
        return MaskRCNNDataset(data_samples=[self.samples[i] for i in indices])