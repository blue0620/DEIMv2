"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import math

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
        sp = F.get_size(flat_inputs[0])
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
class ConvertOBB(T.Transform):
    def __init__(self, normalize=True) -> None:
        super().__init__()
        self.normalize = normalize

    def __call__(self, *inputs: Any) -> Any:
        outputs = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2 or not isinstance(outputs[1], dict):
            return outputs

        image, target = outputs[0], outputs[1]
        if 'rboxes' in target and self.normalize:
            h, w = F.get_size(image)
            scale = torch.tensor([w, h, w, h, 1.0], dtype=target['rboxes'].dtype, device=target['rboxes'].device)
            target['rboxes'] = target['rboxes'] / scale

        return outputs


@register()
class RandomRotateAABBToOBB(T.Transform):
    """Create OBB targets from AABB targets with random training-time rotation.

    This transform is useful when the source annotation is standard COCO bbox
    (xyxy) and no explicit `rboxes` are provided. It keeps the original `boxes`
    untouched and writes `target['rboxes']` in pixel-space `cx, cy, w, h, angle(rad)`.
    """

    def __init__(self, max_angle: float = 45.0, p: float = 0.5, overwrite: bool = False) -> None:
        super().__init__()
        self.max_angle = float(max_angle)
        self.p = float(p)
        self.overwrite = overwrite

    def __call__(self, *inputs: Any) -> Any:
        outputs = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2 or not isinstance(outputs[1], dict):
            return outputs

        target = outputs[1]
        if 'boxes' not in target:
            return outputs
        if ('rboxes' in target) and (not self.overwrite):
            return outputs

        boxes = target['boxes']
        if boxes.numel() == 0:
            target['rboxes'] = torch.zeros((0, 5), dtype=boxes.dtype, device=boxes.device)
            return outputs

        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

        if torch.rand(1).item() < self.p:
            angle_deg = (torch.rand((boxes.shape[0],), device=boxes.device, dtype=boxes.dtype) * 2.0 - 1.0) * self.max_angle
            angle = angle_deg * (math.pi / 180.0)
        else:
            angle = torch.zeros((boxes.shape[0],), device=boxes.device, dtype=boxes.dtype)

        target['rboxes'] = torch.stack([cx, cy, w, h, angle], dim=-1)
        return outputs
