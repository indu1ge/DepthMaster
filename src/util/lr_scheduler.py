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


class IterExponential:
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha


if "__main__" == __name__:
    lr_scheduler = IterExponential(
        total_iter_length=50000, final_ratio=0.01, warmup_steps=200
    )
    lr_scheduler = IterExponential(
        total_iter_length=50000, final_ratio=0.01, warmup_steps=0
    )

    x = np.arange(100000)
    alphas = [lr_scheduler(i) for i in x]
    import matplotlib.pyplot as plt

    plt.plot(alphas)
    plt.savefig("lr_scheduler.png")
