"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt


@register()
class RandomRotateOBB(T.Transform):
    def __init__(self, degrees=15.0, p=0.5, fill=0) -> None:
        super().__init__()
        self.degrees = float(degrees)
        self.p = p
        self.fill = fill

    def _build_obb_from_boxes(self, boxes, angle_rad, image_size):
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 5))
        h, w = image_size
        center = boxes.new_tensor([w / 2.0, h / 2.0])
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        wh = torch.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=-1)

        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rot = boxes.new_tensor([[c, -s], [s, c]])
        centers = torch.stack([cx, cy], dim=-1)
        centers = (centers - center) @ rot.T + center

        angles = boxes.new_full((boxes.shape[0], 1), angle_rad)
        return torch.cat([centers, wh, angles], dim=-1)

    def _rotate_boxes(self, boxes, angle_rad, image_size):
        if boxes.numel() == 0:
            return boxes
        h, w = image_size
        center = boxes.new_tensor([w / 2.0, h / 2.0])
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        rot = boxes.new_tensor([[c, -s], [s, c]])

        corners = torch.stack([
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ], dim=1)
        rotated = (corners - center) @ rot.T + center

        x_min = rotated[..., 0].min(dim=1).values.clamp(min=0, max=w)
        y_min = rotated[..., 1].min(dim=1).values.clamp(min=0, max=h)
        x_max = rotated[..., 0].max(dim=1).values.clamp(min=0, max=w)
        y_max = rotated[..., 1].max(dim=1).values.clamp(min=0, max=h)
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        image, target = sample[:2]
        angle_rad = 0.0
        if torch.rand(1) < self.p:
            angle_deg = float((torch.rand(1) * 2 - 1) * self.degrees)
            angle_rad = math.radians(angle_deg)
            image = F.rotate(image, angle=angle_deg, interpolation=F.InterpolationMode.BILINEAR, fill=self.fill)
            if "boxes" in target:
                target["boxes"] = self._rotate_boxes(target["boxes"], angle_rad, image.size[::-1])

        if "boxes" in target:
            target["obb_boxes"] = self._build_obb_from_boxes(target["boxes"], angle_rad, image.size[::-1])

        if len(sample) == 2:
            return image, target
        return image, target, *sample[2:]


@register()
class ConvertOBB(T.Transform):
    def __init__(self, normalize=False) -> None:
        super().__init__()
        self.normalize = normalize

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        image, target = sample[:2]
        if "obb_boxes" in target and self.normalize:
            h, w = image.shape[-2], image.shape[-1]
            scale = target["obb_boxes"].new_tensor([w, h, w, h, 1.0])
            target["obb_boxes"] = target["obb_boxes"] / scale
        if len(sample) == 2:
            return image, target
        return image, target, *sample[2:]
