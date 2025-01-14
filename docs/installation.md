# Get Started
## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch. We ran our experiments with PyTorch 2.3.0, CUDA 11.8, Python 3.10 and Ubuntu 18.04. We recommend using the same configuration to avoid environment conflicts.

**Note:**
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install [Anaconda](https://www.anaconda.com/download#downloads) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) from the official website.

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name depthmaster python==3.10
conda activate depthmaster
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.


```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Installation

We recommend that users follow our practices for installation.


**Step 1.** Clone repository.

```shell
git clone https://github.com/indu1ge/DepthMaster.git
cd DepthMaster
```

**Step 2.** Install requirements.

```shell
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

pip install opencv-python  transformers matplotlib safetensors accelerate tensorboard datasets scipy einops pytorch_lightning omegaconf diffusers peft

pip3 install h5py scikit-image tqdm bitsandbytes wandb tabulate

```

Download checkpoints for [Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2/tree/main).
<!-- ```shell
mkdir ckpt
cd ckpt
wget https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth # mpvit-small
``` -->