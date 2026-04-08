"""Utility ops for oriented bounding boxes (OBB)."""

import math
import torch
from torch import Tensor


def normalize_angle_half_pi(angle: Tensor) -> Tensor:
    """Normalize angle to [-pi/2, pi/2)."""
    pi = math.pi
    return torch.remainder(angle + pi / 2, pi) - pi / 2


def obb_cxcywha_to_corners(boxes: Tensor) -> Tensor:
    """Convert [cx, cy, w, h, angle(rad)] boxes to 4 corner points [N, 4, 2]."""
    cx, cy, w, h, a = boxes.unbind(-1)
    cos, sin = torch.cos(a), torch.sin(a)

    dw = w * 0.5
    dh = h * 0.5
    x = torch.stack([-dw, dw, dw, -dw], dim=-1)
    y = torch.stack([-dh, -dh, dh, dh], dim=-1)

    rot_x = x * cos.unsqueeze(-1) - y * sin.unsqueeze(-1)
    rot_y = x * sin.unsqueeze(-1) + y * cos.unsqueeze(-1)

    corners = torch.stack([rot_x + cx.unsqueeze(-1), rot_y + cy.unsqueeze(-1)], dim=-1)
    return corners


def obb_to_aabb_xyxy(boxes: Tensor) -> Tensor:
    corners = obb_cxcywha_to_corners(boxes)
    min_xy = corners.min(dim=-2).values
    max_xy = corners.max(dim=-2).values
    return torch.cat([min_xy, max_xy], dim=-1)


def probiou_obb(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-9) -> Tensor:
    """Pairwise probabilistic IoU between OBBs in cxcywha format.

    Returns matrix [N, M].
    """

    def _get_cov(boxes: Tensor):
        w, h, a = boxes[:, 2], boxes[:, 3], boxes[:, 4]
        a_var = (w.pow(2) / 12.0).clamp_min(eps)
        b_var = (h.pow(2) / 12.0).clamp_min(eps)
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        A = a_var * cos_a.pow(2) + b_var * sin_a.pow(2)
        B = a_var * sin_a.pow(2) + b_var * cos_a.pow(2)
        C = (a_var - b_var) * cos_a * sin_a
        return A, B, C

    x1, y1 = boxes1[:, 0], boxes1[:, 1]
    x2, y2 = boxes2[:, 0], boxes2[:, 1]
    A1, B1, C1 = _get_cov(boxes1)
    A2, B2, C2 = _get_cov(boxes2)

    x = x1[:, None] - x2[None, :]
    y = y1[:, None] - y2[None, :]

    A = A1[:, None] + A2[None, :]
    B = B1[:, None] + B2[None, :]
    C = C1[:, None] + C2[None, :]

    den = (A * B - C.pow(2)).clamp_min(eps)

    t1 = (A * y.pow(2) + B * x.pow(2)) / den * 0.25
    t2 = (C * x * y) / den * 0.5

    det1 = (A1 * B1 - C1.pow(2)).clamp_min(eps)
    det2 = (A2 * B2 - C2.pow(2)).clamp_min(eps)
    t3 = 0.5 * torch.log((den / (4.0 * (det1[:, None] * det2[None, :]).sqrt() + eps)) + eps)

    bd = (t1 + t2 + t3).clamp_min(eps)
    hd = (1.0 - torch.exp(-bd) + eps).sqrt()
    return (1.0 - hd).clamp(0.0, 1.0)
