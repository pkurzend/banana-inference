# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import os

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, AutoencoderKL


HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")



model_path = 'stabilityai/stable-diffusion-2-1-base'
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer=tokenizer, safety_checker=None, torch_dtype=torch.float16,  revision='fp16', use_auth_token=HF_AUTH_TOKEN)

model_path = 'stabilityai/stable-diffusion-2-1-base'
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer=tokenizer, safety_checker=None, torch_dtype=torch.float16,  revision=None, use_auth_token=HF_AUTH_TOKEN)


