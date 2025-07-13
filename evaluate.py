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

import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from depthmaster import DepthMasterPipeline
from depthmaster.modules.unet_2d_condition_s2 import UNet2DConditionModel
from src.util.seeding import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from src.util import metric
from src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from src.util.metric import MetricTracker

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
    "si_boundary_F1"
]

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt/eval",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity", "least_square_sqrt_disp"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()

    '''-----------------------------------------------------------------------------------------------------------------------'''

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    ensemble_size = args.ensemble_size

    alignment = args.alignment
    alignment_max_res = args.alignment_max_res

    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # ------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe = DepthMasterPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, f'unet'))
    pipe.unet = unet

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # -------------------- Per-sample metric file head --------------------
    per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    # -------------------- Evaluate --------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            # Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)

            # Predict depth
            pipe_out = pipe(
                input_image,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=True,
                resample_method=resample_method,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            depth_raw_ts = batch["depth_raw_linear"].squeeze()
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            rgb_name = batch["rgb_relative_path"][0]

            depth_raw = depth_raw_ts.numpy()
            valid_mask = valid_mask_ts.numpy()

            depth_raw_ts = depth_raw_ts.to(device)
            valid_mask_ts = valid_mask_ts.to(device)

            # Align with GT using least square
            if "least_square" == alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=alignment_max_res,
                )
            elif "least_square_disparity" == alignment:
                # convert GT depth -> GT disparity
                gt_disparity, gt_non_neg_mask = depth2disparity(
                    depth=depth_raw, return_mask=True
                )
                # LS alignment in disparity space
                pred_non_neg_mask = depth_pred > 0
                valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                disparity_pred, scale, shift = align_depth_least_square(
                    gt_arr=gt_disparity,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_nonnegative_mask,
                    return_scale_shift=True,
                    max_resolution=alignment_max_res,
                )
                # convert to depth
                disparity_pred = np.clip(
                    disparity_pred, a_min=1e-3, a_max=None
                )  # avoid 0 disparity
                depth_pred = disparity2depth(disparity_pred)
            elif "least_square_sqrt_disp" == alignment:
                # convert GT depth -> GT sqrt disparity
                gt_disparity, gt_non_neg_mask = depth2disparity(
                    depth=depth_raw, return_mask=True
                )
                gt_sqrt_disp = np.sqrt(gt_disparity)
                gt_non_neg_mask = (gt_sqrt_disp > 0) & gt_non_neg_mask

                # LS alignment in sqrt space
                pred_non_neg_mask = depth_pred > 0
                valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask
                depth_sqrt_disp_pred, scale, shift = align_depth_least_square(
                    gt_arr=gt_sqrt_disp,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_nonnegative_mask,
                    return_scale_shift=True,
                )
                # convert to depth
                disparity_pred = depth_sqrt_disp_pred ** 2

                # convert to depth
                disparity_pred = np.clip(
                    disparity_pred, a_min=1e-3, a_max=None
                )  # avoid 0 disparity
                depth_pred = disparity2depth(disparity_pred)
        
            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
            )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate (using CUDA if available)
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(device)

            for met_func in metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)
            
            # Save per-sample metric
            with open(per_sample_filename, "a+") as f:
                f.write(batch["rgb_relative_path"][0] + ",")
                f.write(",".join(sample_metric))
                f.write("\n")

    print("Evaluate Results:")
    for key in metric_tracker.result().keys():
        print(f"{key}={metric_tracker.result()[key]}")
    
    
    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {output_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    if alignment:
        metrics_filename += f"-{alignment}"
    metrics_filename += ".txt"

    _save_to = os.path.join(output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
