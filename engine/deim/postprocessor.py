"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        obb_boxes = outputs.get('pred_obb_boxes', None)
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        obb_pred, obb_points = None, None
        if obb_boxes is not None:
            scale = torch.cat([orig_target_sizes[:, [1, 0, 1, 0]], torch.ones_like(orig_target_sizes[:, :1])], dim=-1)
            obb_pred = obb_boxes * scale.unsqueeze(1)
            obb_points = self._obb_to_corners(obb_pred)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            if obb_pred is not None:
                obb_pred = obb_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, obb_pred.shape[-1]))
                obb_points = obb_points.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 2))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            if obb_pred is not None:
                return labels, boxes, scores, obb_pred
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        if obb_pred is None:
            for lab, box, sco in zip(labels, boxes, scores):
                result = dict(labels=lab, boxes=box, scores=sco)
                results.append(result)
        else:
            for lab, box, sco, obb, obb_pt in zip(labels, boxes, scores, obb_pred, obb_points):
                result = dict(labels=lab, boxes=box, scores=sco, obb_boxes=obb, obb_points=obb_pt)
                results.append(result)

        return results

    @staticmethod
    def _obb_to_corners(obb_boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h, a = [obb_boxes[..., i] for i in range(5)]
        half_w, half_h = w / 2.0, h / 2.0
        base = torch.stack([
            torch.stack([-half_w, -half_h], dim=-1),
            torch.stack([half_w, -half_h], dim=-1),
            torch.stack([half_w, half_h], dim=-1),
            torch.stack([-half_w, half_h], dim=-1),
        ], dim=-2)
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        rot = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ], dim=-2)
        corners = torch.matmul(base, rot.transpose(-1, -2))
        corners[..., 0] += cx.unsqueeze(-1)
        corners[..., 1] += cy.unsqueeze(-1)
        return corners


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
