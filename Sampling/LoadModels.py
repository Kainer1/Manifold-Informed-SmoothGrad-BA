"""Original loaders for the retained latent and pixel diffusion models."""

import os
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    IFImg2ImgPipeline,
    IFImg2ImgSuperResolutionPipeline,
    StableDiffusionImg2ImgPipeline,
    Transformer2DModel,
)

# Set this to the original OpenAI 256x256 ImageNet checkpoint before using ADM.
# Installation of guided-diffusion and the checkpoint source are in README.md.
ADM_WEIGHTS_PATH = Path("/path/to/256x256_diffusion.pt")


def load_openai_adm(device):
    if not ADM_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"ADM weights not found at {ADM_WEIGHTS_PATH}. "
            "Set ADM_WEIGHTS_PATH in Sampling/LoadModels.py to 256x256_diffusion.pt."
        )
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    config = model_and_diffusion_defaults()
    config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": True,
            "diffusion_steps": 1000,
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
    model, diffusion = create_model_and_diffusion(**config)
    state_dict = torch.load(ADM_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    if config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def load_deepfloyd_if(device):
    pipe_i = IFImg2ImgPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        token=os.environ.get("HF_TOKEN"),
    )
    pipe_i.to(device)

    pipe_ii = IFImg2ImgSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=pipe_i.text_encoder,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        token=os.environ.get("HF_TOKEN"),
    )
    pipe_ii.to(device)

    return pipe_i, pipe_ii


def load_dit_imagenet(device):
    model_id = "facebook/dit-xl-2-256"

    transformer = Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        token=os.environ.get("HF_TOKEN"),
        use_safetensors=False,
    )
    transformer.to(device).eval()

    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", token=os.environ.get("HF_TOKEN")
    )

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        token=os.environ.get("HF_TOKEN"),
        use_safetensors=False,
    )
    vae.to(device).eval()

    return vae, transformer, scheduler


def load_sd2_img2img(device):
    """Load Stable Diffusion 2.0-base for fixed 256x256 img2img sampling."""
    model_id = "sd2-community/stable-diffusion-2-base"
    device_type = torch.device(device).type

    load_kwargs = {
        "torch_dtype": torch.float16 if device_type == "cuda" else torch.float32,
        "use_safetensors": True,
        "safety_checker": None,
        "feature_extractor": None,
        "token": os.environ.get("HF_TOKEN"),
    }
    if device_type == "cuda":
        load_kwargs["variant"] = "fp16"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **load_kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_vae_slicing()

    return pipe
