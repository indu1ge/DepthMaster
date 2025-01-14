# Last modified: 2025-01-14
#
# Copyright 2025 Ziyang Song, USTC. All rights reserved.
#
# This file has been modified from the original version.
# Original copyright (c) 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/indu1ge/DepthMaster#-citation
# More information about the method can be found at https://indu1ge.github.io/DepthMaster_page
# --------------------------------------------------------------------------


import pandas as pd
import torch
from typing import Tuple
import torch.nn.functional as F
# from src.util.loss import SSIM
from skimage.metrics import structural_similarity
import numpy as np
from skimage import feature
from scipy import ndimage


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss

def si_boundary_F1(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask=None,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
) -> float:
    predicted_depth = predicted_depth.squeeze()
    # predicted_depth = (predicted_depth + 1)
    target_depth = target_depth.squeeze()
    assert predicted_depth.ndim == target_depth.ndim == 2
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    # print(target_depth.min())
    f1_scores = torch.Tensor(
        [
            boundary_f1(invert_depth(predicted_depth), invert_depth(target_depth), t, valid_mask)
            # boundary_f1(predicted_depth, target_depth, t)
            for t in thresholds
        ]
    )
    return torch.sum(f1_scores * weights)

def get_thresholds_and_weights(
    t_min: float, t_max: float, N: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate thresholds and weights for the given range.

    Args:
    ----
        t_min (float): Minimum threshold.
        t_max (float): Maximum threshold.
        N (int): Number of thresholds.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Array of thresholds and corresponding weights.

    """
    thresholds = torch.linspace(t_min, t_max, N)
    weights = thresholds / thresholds.sum()
    return thresholds, weights

def invert_depth(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverts a depth map with numerical stability.

    Args:
    ----
        depth (np.ndarray): Depth map to be inverted.
        eps (float): Minimum value to avoid division by zero (default is 1e-6).

    Returns:
    -------
    np.ndarray: Inverted depth map.

    """
    inverse_depth = 1.0 / depth.clip(min=eps)
    return inverse_depth

def boundary_f1(
    pr: torch.Tensor,
    gt: torch.Tensor,
    t: float,
    valid_mask: torch.Tensor,
    return_p: bool = False,
    return_r: bool = False,
) -> float:
    """Calculate Boundary F1 score.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth depth matrix.
        t (float): Threshold for comparison.
        return_p (bool, optional): If True, return precision. Defaults to False.
        return_r (bool, optional): If True, return recall. Defaults to False.

    Returns:
    -------
        float: Boundary F1 score, or precision, or recall depending on the flags.

    """
    ap, bp, cp, dp = fgbg_depth(pr, t, valid_mask)
    ag, bg, cg, dg = fgbg_depth(gt, t, valid_mask)

    r = 0.25 * (
        torch.count_nonzero(ap & ag) / max(torch.count_nonzero(ag), 1)
        + torch.count_nonzero(bp & bg) / max(torch.count_nonzero(bg), 1)
        + torch.count_nonzero(cp & cg) / max(torch.count_nonzero(cg), 1)
        + torch.count_nonzero(dp & dg) / max(torch.count_nonzero(dg), 1)
    )
    p = 0.25 * (
        torch.count_nonzero(ap & ag) / max(torch.count_nonzero(ap), 1)
        + torch.count_nonzero(bp & bg) / max(torch.count_nonzero(bp), 1)
        + torch.count_nonzero(cp & cg) / max(torch.count_nonzero(cp), 1)
        + torch.count_nonzero(dp & dg) / max(torch.count_nonzero(dp), 1)
    )
    if r + p == 0:
        return 0.0
    if return_p:
        return p
    if return_r:
        return r
    return 2 * (r * p) / (r + p)

def fgbg_depth(
    d: torch.Tensor, t: float, valid_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find foreground-background relations between neighboring pixels.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for comparison.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations.

    """
    h, w = d.shape
    invalid_h = torch.zeros(h, 1).bool().to(valid_mask.device)
    invalid_w = torch.zeros(1, w).bool().to(valid_mask.device)
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    right_is_big_enough = torch.cat([right_is_big_enough, invalid_h], dim=1) & valid_mask
    left_is_big_enough = torch.cat([invalid_h, left_is_big_enough], dim=1) & valid_mask
    bottom_is_big_enough = torch.cat([bottom_is_big_enough, invalid_w], dim=0) & valid_mask
    top_is_big_enough = torch.cat([invalid_w, top_is_big_enough], dim=0) & valid_mask
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def gauss(x, sigma):
    y = torch.exp(-(x**2) / (2 * sigma**2)) / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma**2)
    return y


def gaussgradient(im, sigma):
    epsilon = torch.tensor(1e-2)
    halfsize = int(torch.ceil(sigma * torch.sqrt(-2 * torch.log(torch.sqrt(2 * torch.tensor(torch.pi)) * sigma * epsilon))))
    size = 2 * halfsize + 1
    hx = torch.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / torch.sqrt(torch.sum(torch.abs(hx) * torch.abs(hx)))
    hx = hx.to(im.device)
    hy = hx.t().to(im.device)

    # gx = scipy.ndimage.convolve(im, hx, mode="nearest")
    # gy = scipy.ndimage.convolve(im, hy, mode="nearest")
    gx = F.conv2d(im.unsqueeze(0).unsqueeze(0), hx.unsqueeze(0).unsqueeze(0), padding=halfsize)
    gy = F.conv2d(im.unsqueeze(0).unsqueeze(0), hy.unsqueeze(0).unsqueeze(0), padding=halfsize)


    return gx.squeeze(0).squeeze(0), gy.squeeze(0).squeeze(0)


def gradient_loss(pred, target, valid_mask=None):

    # min_d = target[valid_mask].min()
    # max_d = target[valid_mask].max()
    # pred = (pred - pred.min())/ (pred.max() - pred.min())
    # target = (target - min_d)/ (max_d - min_d)
    
    _min, _max = torch.quantile(
        target[valid_mask],
        torch.tensor([0.02, 0.98]).to(pred.device),
    )
    target_norm = (target - _min) / (_max - _min)
    target_norm = torch.clip(target_norm, 0, 1)
    target_norm[~valid_mask] = 0.
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())

    pred_x, pred_y = gaussgradient(pred_norm, torch.tensor(1.4))
    target_x, target_y = gaussgradient(target_norm, torch.tensor(1.4))

    pred_amp = torch.sqrt(pred_x**2 + pred_y**2)
    target_amp = torch.sqrt(target_x**2 + target_y**2)

    error_map = (pred_amp - target_amp) ** 2
    mask = target_amp > 0.05
    if valid_mask is not None:
        loss = torch.mean(error_map[valid_mask & mask])
    else:
        loss = torch.mean(error_map[mask])
    return loss


def grad_sim(pred: torch.Tensor, target: torch.Tensor, valid_mask=None):
    _min, _max = torch.quantile(
        target[valid_mask],
        torch.tensor([0.02, 0.98]).to(pred.device),
    )
    target_norm = (target - _min) / (_max - _min)
    target_norm = torch.clip(target_norm, 0, 1)
    target_norm[~valid_mask] = 0.
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
    grad_pred = torch.sqrt(grad(pred_norm))
    grad_gt = torch.sqrt(grad(target_norm))
    error_map = (grad_pred - grad_gt)**2
    mask = grad_gt > 0.05
    
    if valid_mask is not None:
        valid_mask1 = valid_mask[1:, 1:]
        valid_mask2 = valid_mask[:-1, :-1]
        valid_mask = valid_mask1 & valid_mask2 & mask
        loss = torch.mean(error_map[valid_mask])
    else:
        loss = torch.mean(error_map[mask])
    return loss




def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[1:, 1:] - x[1:, :-1]
    diff_y = x[1:, 1:] - x[:-1, 1:]
    mag = diff_x**2 + diff_y**2
    # # angle_ratio
    # angle = torch.atan(diff_y / (diff_x + 1e-10))
    # result = torch.cat([mag, angle], dim=1)
    return mag


    
    
def psnr(pred: torch.Tensor, target: torch.Tensor, valid_mask=None):
    _min, _max = torch.quantile(
        target[valid_mask],
        torch.tensor([0.02, 0.98]).to(pred.device),
    )
    target_norm = (target - _min) / (_max - _min)
    target_norm = torch.clip(target_norm, 0, 1)
    target_norm[~valid_mask] = 0.
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
    mse = ((((pred_norm - target_norm)) ** 2)[valid_mask]).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(pred: torch.Tensor, target: torch.Tensor, valid_mask=None):
    _min, _max = torch.quantile(
        target[valid_mask],
        torch.tensor([0.02, 0.98]).to(pred.device),
    )
    target_norm = (target - _min) / (_max - _min)
    target_norm = torch.clip(target_norm, 0, 1)
    target_norm[~valid_mask] = 0.
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())

    ssim, S = structural_similarity(pred_norm.cpu().numpy(), target_norm.cpu().numpy(), win_size=3, gradient=False, data_range=1.0, multichannel=False, channel_axis=None, gaussian_weights=False, full=True)

    return S[valid_mask.cpu().numpy()].mean()

def compute_depth_boundary_error(edges,pred):
    # skip dbe for this image if there is no ground truth distinc edge
    if np.sum(edges) == 0:
        dbe_acc = 0
        dbe_com = 0
    
    else:
         # normalize est depth map from 0 to 1
         pred_normalized = pred.copy().astype('f')
         pred_normalized[pred_normalized==0]=np.nan
         pred_normalized = pred_normalized - np.nanmin(pred_normalized)
         pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    
         # apply canny filter 
         edges_est = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)
         #plt.imshow(edges_est)
         
         # compute distance transform for chamfer metric
         D_gt = ndimage.distance_transform_edt(1-edges)
         D_est = ndimage.distance_transform_edt(1-edges_est)

         max_dist_thr = 10.; # Threshold for local neighborhood
         
         mask_D_gt = D_gt<max_dist_thr; # truncate distance transform map
         
         E_fin_est_filt = edges_est*mask_D_gt; # compute shortest distance for all predicted edges
         
         if np.sum(E_fin_est_filt) == 0: # assign MAX value if no edges could be found in prediction
             dbe_acc = max_dist_thr
             dbe_com = max_dist_thr
         else:
             dbe_acc = np.nansum(D_gt*E_fin_est_filt)/np.nansum(E_fin_est_filt) # accuracy: directed chamfer distance
             dbe_com = np.nansum(D_est*edges)/np.nansum(edges) # completeness: directed chamfer distance (reversed)
                
    if dbe_acc and dbe_com:
        return 2* dbe_acc*dbe_com / (dbe_acc + dbe_acc)
    else:
        return 0