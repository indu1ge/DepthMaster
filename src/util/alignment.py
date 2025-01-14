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

import numpy as np
import torch

def align_depth_medium_mask(
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    max_resolution=None,
):
    ori_shape = gt.shape[-2:]  # input shape
    batch_size = gt.shape[0]
    # print(gt.shape)

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(gt)
            valid_mask = downscaler(valid_mask).bool()


    scale_ls = []
    shift_ls = []

    for i in range(batch_size):
        # print('yes')

        gt_masked = gt[i][valid_mask[i]]
        shift = torch.median(gt_masked).unsqueeze(0)
        scale = torch.mean(torch.abs(gt_masked - shift)).unsqueeze(0)
        # print(scale)

        scale_ls.append(scale)
        shift_ls.append(shift)
        # print(len(scale_ls))
        
    scale = torch.concat(scale_ls, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    shift = torch.concat(shift_ls, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    return scale, shift


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


# ******************** disparity space ********************
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)


def align_depth_least_square_torch_mask(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
    max_resolution=None,
):
    ori_shape = pred.shape[-2:]  # input shape
    batch_size = gt.shape[0]

    # gt = gt_arr.squeeze()  # [B, H, W]
    # pred = pred_arr.squeeze()
    # valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(gt)
            pred = downscaler(pred)
            valid_mask = downscaler(valid_mask).bool()

    assert (
        gt.shape == pred.shape
    ), f"{gt.shape}, {pred.shape}"

    scale_ls = []
    shift_ls = []

    for i in range(batch_size):

        gt_masked = gt[i][valid_mask[i]].view(-1, 1)
        pred_masked = pred[i][valid_mask[i]].view(-1, 1)

        # torch solver
        ones = torch.ones_like(pred_masked)
        A = torch.cat([pred_masked, ones], dim=-1)
        X, *_ = torch.linalg.lstsq(A, gt_masked)

        scale, shift = X[0, :].detach(), X[1, :].detach()
        scale_ls.append(scale)
        shift_ls.append(shift)
    scale = torch.concat(scale_ls, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    shift = torch.concat(shift_ls, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    return scale, shift
