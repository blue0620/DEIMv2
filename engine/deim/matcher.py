"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict, List

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from ..core import register
import numpy as np


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


def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[0] * b[1] - a[1] * b[0]


def _line_intersection(p1: torch.Tensor, p2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    r = p2 - p1
    s = q2 - q1
    denom = _cross2d(r, s)
    if torch.abs(denom) < 1e-8:
        return (p2 + q1) / 2.0
    t = _cross2d(q1 - p1, s) / denom
    return p1 + t * r


def _polygon_area(poly: torch.Tensor) -> torch.Tensor:
    if poly.shape[0] < 3:
        return poly.new_tensor(0.0)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, shifts=-1)) - torch.dot(y, torch.roll(x, shifts=-1)))


def _clip_polygon(subject: torch.Tensor, clipper: torch.Tensor) -> torch.Tensor:
    if subject.shape[0] < 3 or clipper.shape[0] < 3:
        return subject.new_zeros((0, 2))

    output: List[torch.Tensor] = [p for p in subject]
    clip_sign = 1.0 if (_polygon_area(clipper) > 0) else -1.0

    def inside(p: torch.Tensor, cp1: torch.Tensor, cp2: torch.Tensor) -> bool:
        return (clip_sign * _cross2d(cp2 - cp1, p - cp1)).item() >= -1e-7

    for i in range(clipper.shape[0]):
        if not output:
            break
        cp1 = clipper[i]
        cp2 = clipper[(i + 1) % clipper.shape[0]]
        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            e_inside = inside(e, cp1, cp2)
            s_inside = inside(s, cp1, cp2)
            if e_inside:
                if not s_inside:
                    output.append(_line_intersection(s, e, cp1, cp2))
                output.append(e)
            elif s_inside:
                output.append(_line_intersection(s, e, cp1, cp2))
            s = e

    if not output:
        return subject.new_zeros((0, 2))
    return torch.stack(output, dim=0)


def _pairwise_rotated_iou(obb1: torch.Tensor, obb2: torch.Tensor) -> torch.Tensor:
    corners1 = _obb_to_corners(obb1).detach().cpu()
    corners2 = _obb_to_corners(obb2).detach().cpu()
    area1 = (obb1[:, 2] * obb1[:, 3]).detach().cpu().clamp_min(1e-8)
    area2 = (obb2[:, 2] * obb2[:, 3]).detach().cpu().clamp_min(1e-8)
    ious = torch.zeros((obb1.shape[0], obb2.shape[0]), dtype=obb1.dtype)

    for i in range(obb1.shape[0]):
        for j in range(obb2.shape[0]):
            inter_poly = _clip_polygon(corners1[i], corners2[j])
            inter_area = _polygon_area(inter_poly)
            union = area1[i] + area2[j] - inter_area
            ious[i, j] = inter_area / union.clamp_min(1e-8)

    return ious.to(device=obb1.device)


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_angle = weight_dict.get('cost_angle', 0.0)
        self.cost_riou = weight_dict.get('cost_riou', 0.0)
        self.angle_smooth_l1_beta = weight_dict.get('angle_smooth_l1_beta', 0.1)

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 \
            or self.cost_angle != 0 or self.cost_riou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        has_obb = ("pred_obb_boxes" in outputs) and all(("obb_boxes" in v) for v in targets)
        if has_obb:
            out_obb = outputs["pred_obb_boxes"].flatten(0, 1)
            tgt_obb = torch.cat([v["obb_boxes"] for v in targets])

            angle_delta = out_obb[:, 4].unsqueeze(1) - tgt_obb[:, 4].unsqueeze(0)
            angle_periodic = torch.sin(angle_delta)
            cost_angle = F.smooth_l1_loss(
                angle_periodic,
                torch.zeros_like(angle_periodic),
                reduction='none',
                beta=self.angle_smooth_l1_beta,
            )
            cost_riou = -_pairwise_rotated_iou(out_obb, tgt_obb)
        else:
            cost_angle = 0.0
            cost_riou = 0.0

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score
            class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]

            # # Compute iou
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
            if has_obb:
                C = C + self.cost_angle * cost_angle + self.cost_riou * cost_riou
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            if has_obb:
                C = C + self.cost_angle * cost_angle + self.cost_riou * cost_riou

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list
