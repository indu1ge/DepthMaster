import torch

def get_z0(
        scheduler,
        sample: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    
    t = timesteps.to(sample.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    model_output = noise

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
        model_output, _ = torch.split(model_output, sample.shape[1], dim=1)

    # 1. compute alphas, betas
    alpha_prod_t = alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    alpha_prod_t = alpha_prod_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    beta_prod_t = beta_prod_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    return pred_original_sample