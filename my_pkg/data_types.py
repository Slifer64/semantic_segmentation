import os.path
import PIL.Image
import numpy as np
import abc
import torch
from typing import List, Tuple, Dict, Union
import torchvision.transforms.functional as torchvision_F
import torchvision.transforms as torchvision_T

__all__ = [
    'BaseDataType',
    'SegmentationMask',
    'InputImage',
]


class BaseDataType:

    def __init__(self):
        self.transforms = {}

    def __call__(self, transform, *args, **kwargs) -> 'BaseDataType':
        """
        Applies a transform to the underlying data.

        Arguments:
        transform -- string with name of the transform
        *args, **kwargs -- arguments to be passed to the transform
        """

        # if the transform is not supported (implemented) skip it
        if transform not in self.transforms:
            return self
        # else apply the transform
        else:
            return self.transforms[transform](*args, **kwargs)

    def get_type(self) -> str:
        """
        Returns the class type as a string.
        """
        return self.__class__.__name__

    @abc.abstractmethod
    def numpy(self) -> np.array:
        """
        Returns the underlying raw data as an np.array.
        """
        pass

    @abc.abstractmethod
    def torch_tensor(self) -> torch.Tensor:
        """
        Returns the underlying raw data as a torch.Tensor.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path):
        """
        Loads the data from the specified path. The filename will be chosen automatically based on the data-type,
        unless specified explicitly from additional input arguments.

        Arguments:
        path -- string, path where the file is located.
        """
        pass

    @abc.abstractmethod
    def save(self, path):
        """
        Loads the data from the specified path.

        Arguments:
        path -- string, path where the file is located.
        """
        pass


class SegmentationMask(BaseDataType):

    data_name = 'seg_mask'
    filename = 'seg_mask.png'

    def __init__(self, seg_mask: Union[np.array, PIL.Image.Image]):
        super().__init__()
        self.data = torch.tensor(np.asarray(seg_mask), dtype=torch.long)[None, ...] # add 1 channel

        self.transforms = {
            'hflip': self.hflip,
            'center_crop': self.center_crop,
            'resize': self.resize,
            'pad': self.pad,
            'rotate': self.rotate,
            'translate': self.translate,
            'scale': self.scale,
            'perspective': self.perspective,
        }

    @classmethod
    def empty(cls, size):
        return cls(-np.ones(size))

    @classmethod
    def load(cls, path, filename=None) -> 'SegmentationMask':
        if filename is None:
            filename = cls.filename
        filename = os.path.join(path, filename)
        if not os.path.exists(filename):
            return None
        return cls(PIL.Image.open(filename))

    def save(self, path, filename=None):
        if filename is None:
            filename = self.filename
        torchvision_F.to_pil_image(self.data.type(torch.int)).save(os.path.join(path, filename), format='png')
    
    def numpy(self) -> np.array:
        return self.torch_tensor().numpy()

    def torch_tensor(self) -> torch.Tensor:
        return self.data[0]

    def size(self) -> List[int]:
        return list(self.data.shape)

    # =========== Transforms ==============

    def hflip(self) -> 'SegmentationMask':
        self.data = torchvision_F.hflip(self.data)
        return self

    def center_crop(self, size) -> 'SegmentationMask':
        self.data = torchvision_F.center_crop(self.data, size)
        return self

    def resize(self, size, not_used) -> 'SegmentationMask':
        self.data = torchvision_F.resize(self.data, size, torchvision_T.InterpolationMode.NEAREST)
        return self

    def pad(self, padding, fill=0) -> 'SegmentationMask':
        self.data = torchvision_F.pad(self.data, padding, fill)
        return self

    def rotate(self, angle) -> 'SegmentationMask':
        self.data = torchvision_F.affine(img=self.data, angle=angle, translate=[0, 0], scale=1.0, shear=0, fill=0.0,
                                   interpolation=torchvision_T.InterpolationMode.NEAREST)
        return self

    def translate(self, translate) -> 'SegmentationMask':
        self.data = torchvision_F.affine(img=self.data, angle=0, translate=translate, scale=1.0, shear=0, fill=0.0,
                                   interpolation=torchvision_T.InterpolationMode.NEAREST)
        return self

    def scale(self, scale) -> 'SegmentationMask':
        self.data = torchvision_F.affine(img=self.data, angle=0, translate=[0, 0], scale=scale, shear=0, fill=0.0,
                                   interpolation=torchvision_T.InterpolationMode.NEAREST)
        return self

    def perspective(self, startpoints: List[List[float]], endpoints: List[List[float]], fill=0) -> 'SegmentationMask':
        """ Applies perspective transformation to 'self'.

        Args:
            startpoints: List[List[float]], normalized in [0, 1) corner start-points.
            endpoints: List[List[float]], normalized in [0, 1) corner end-points.

        Returns:
            self
        """
        h, w = self.size()[-2:]
        scale_points = lambda points: [temp.int().tolist() for temp in torch.as_tensor(points) * torch.tensor([[w, h]])]
        startpoints, endpoints = map(scale_points, (startpoints, endpoints))
        self.data = torchvision_F.perspective(self.data, startpoints, endpoints, torchvision_T.InterpolationMode.NEAREST, fill=fill)
        return self

class BaseRGBimage(BaseDataType):

    data_name = None
    filename = None

    def __init__(self, img: Union[np.array, PIL.Image.Image]):
        super().__init__()
        self.data = torchvision_F.to_tensor(img)

        # supported (implemented) transforms
        self.transforms = {
            'hflip': self.hflip,
            'center_crop': self.center_crop,
            'resize': self.resize,
            'pad': self.pad,
            'rotate': self.rotate,
            'translate': self.translate,
            'scale': self.scale,
            'perspective': self.perspective,
        }

    @classmethod
    def load(cls, path, filename=None) -> 'BaseRGBimage':
        if filename is None:
            filename = cls.filename
        filename = os.path.join(path, filename)
        if not os.path.exists(filename):
            return None
        return cls(PIL.Image.open(filename))

    def save(self, path, filename=None):
        if filename is None:
            filename = self.filename
        torchvision_F.to_pil_image(self.data).save(os.path.join(path, filename), format='png')

    def numpy(self) -> np.array:
        return self.data.permute(1, 2, 0).numpy()

    def torch_tensor(self) -> torch.Tensor:
        return self.data

    def size(self) -> List[int]:
        return list(self.data.shape)

    # =========== Transforms ==============

    def hflip(self) -> 'BaseRGBimage':
        self.data = torchvision_F.hflip(self.data)
        return self

    def center_crop(self, size) -> 'BaseRGBimage':
        self.data = torchvision_F.center_crop(self.data, size)
        return self

    def resize(self, size, interpolation) -> 'BaseRGBimage':
        self.data = torchvision_F.resize(self.data, size, interpolation)
        return self

    def pad(self, padding, fill) -> 'BaseRGBimage':
        self.data = torchvision_F.pad(self.data, padding, fill)
        return self

    def rotate(self, angle, fill=0.0) -> 'BaseRGBimage':
        self.data = torchvision_F.affine(img=self.data, angle=angle, translate=[0, 0], scale=1.0, shear=0, fill=fill,
                                   interpolation=torchvision_T.InterpolationMode.NEAREST)
        return self

    def translate(self, translate, fill=0.0) -> 'BaseRGBimage':
        self.data = torchvision_F.affine(img=self.data, angle=0, translate=translate, scale=1.0, shear=0, fill=fill,
                                   interpolation=torchvision_T.InterpolationMode.NEAREST)
        return self

    def scale(self, scale, fill=0.0) -> 'BaseRGBimage':
        self.data = torchvision_F.affine(img=self.data, angle=0, translate=[0, 0], scale=scale, shear=0, fill=fill,
                                   interpolation=torchvision_T.InterpolationMode.NEAREST)
        return self

    def perspective(self, startpoints: List[List[float]], endpoints: List[List[float]], fill=0.0) -> 'BaseRGBimage':
        """ Applies perspective transformation to 'self'.

        Args:
            startpoints: List[List[float]], normalized in [0, 1) corner start-points.
            endpoints: List[List[float]], normalized in [0, 1) corner end-points.

        Returns:
            self
        """
        h, w = self.size()[-2:]
        scale_points = lambda points: [temp.int().tolist() for temp in torch.as_tensor(points) * torch.tensor([[w, h]])]
        startpoints, endpoints = map(scale_points, (startpoints, endpoints))
        self.data = torchvision_F.perspective(self.data, startpoints, endpoints, torchvision_T.InterpolationMode.NEAREST, fill=fill)
        return self

# ========= Specific Types ===========

class InputImage(BaseRGBimage):
    data_name = 'rgb'
    filename = 'rgb.png'

    def __init__(self, img: Union[np.array, PIL.Image.Image]):
        img = np.array(img)[..., 0:3] # make sure only 3 channels are read
        super().__init__(img)

        # add extra supported transforms
        self.transforms.update({
            'grayscale': self.grayscale,
            'grayscale_to_rgb': self.grayscale_to_rgb,
            'normalize': self.normalize,
            'color_jitter': self.color_jitter,
            'gaussian_noise': self.gaussian_noise,
        })

    def normalize(self, mean, std) -> 'InputImage':
        self.data = torchvision_F.normalize(self.data, mean=mean, std=std)
        return self

    def grayscale(self) -> 'InputImage':
        self.data = torchvision_T.Grayscale()(self.data)
        return self

    def grayscale_to_rgb(self) -> 'InputImage':
        self.data = self.data.repeat(3, 1, 1)
        return self

    def color_jitter(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.data = torchvision_T.ColorJitter(brightness=brightness, contrast=contrast,
                                              saturation=saturation, hue=hue)(self.data)
        return self

    def gaussian_noise(self, mean, std) -> 'InputImage':
        self.data = self.data + torch.randn(self.data.size()) * std + mean
        self.data = torch.clip(self.data, 0.0, 1.0)
        return self
