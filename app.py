import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from CLIPTokenizerWithEmbeddings import CLIPTokenizerWithEmbeddings


import json
import os
import s3fs

import uuid    



# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    model_path = 'stabilityai/stable-diffusion-2-1-base'

    tokenizer = CLIPTokenizerWithEmbeddings.from_pretrained(model_path, subfolder="tokenizer")

    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer=tokenizer, safety_checker=None, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN).to("cuda")

    model.tokenizer.load_embedding('<flora-marble>', './FloralMarble-400.pt', model.text_encoder)
    model.tokenizer.load_embedding('<flora-marble250>', './FloralMarble-250.pt', model.text_encoder)
    model.tokenizer.load_embedding('<flora-marble150>', './FloralMarble-150.pt', model.text_encoder)
    model.tokenizer.load_embedding('<photo-helper>', './PhotoHelper.pt', model.text_encoder)
    model.tokenizer.load_embedding('<lysergian-dreams>', './LysergianDreams-3600.pt', model.text_encoder)
    model.tokenizer.load_embedding('<urban-jungle>', './UrbanJungle.pt', model.text_encoder)
    model.tokenizer.load_embedding('<cinema-helper>', './CinemaHelper.pt', model.text_encoder)
    model.tokenizer.load_embedding('<neg-mutation>', './NegMutation-2400.pt', model.text_encoder)
    model.tokenizer.load_embedding('<car-helper>', './CarHelper.pt', model.text_encoder)
    model.tokenizer.load_embedding('<hyper-fluid>', './HyperFluid.pt', model.text_encoder)
    model.tokenizer.load_embedding('<double-exposure>', './dblx.pt', model.text_encoder)
    model.tokenizer.load_embedding('<pencil-graphite>', './ppgra.pt', model.text_encoder)
    model.tokenizer.load_embedding('<viking-punk>', './VikingPunk.pt', model.text_encoder)
    model.tokenizer.load_embedding('<gigachad>', './GigaChad.pt', model.text_encoder)
    model.tokenizer.load_embedding('<glass-case>', './kc16-v4-5000.pt', model.text_encoder)
    model.tokenizer.load_embedding('<action-helper>', './ActionHelper.pt', model.text_encoder)
    


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    from dotenv import load_dotenv
    load_dotenv() # take environment variables from .env.

    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")



    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', '')
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)

    user_id = model_inputs.get("user_id",None)
    model_id = model_inputs.get("model_id",None)



    if user_id == None:
        return {'message': "No user_id provided"}

    if model_id == None:
        return {'message': "No model_id provided"}

    if prompt == None:
        return {'message': "No prompt provided"}


    
    s3_file = s3fs.S3FileSystem(key=os.getenv("KEY"), secret=os.getenv("SECRET"), client_kwargs={'endpoint_url':'https://s3.eu-central-1.amazonaws.com'})
     
    local_path = f"{model_id}/"
    os.makedirs(local_path, exist_ok=True)

    s3_text_encoder_path = f"stable-diffusion-finetunings/{user_id}/{model_id}/text_encoder/"
    s3_file.get(s3_text_encoder_path, local_path+'text_encoder/', recursive=True)

    s3_unet_path = f"stable-diffusion-finetunings/{user_id}/{model_id}/unet/"
    s3_file.get(s3_unet_path, local_path+'text_encoder/', recursive=True)    
 

    model_path = local_path

    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
    )
    model.text_encoder = text_encoder


    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
    )

    model.unet = unet


    # s3_model_index_path = f"stable-diffusion-finetunings/{user_id}/data/{model_id}/model_index.json "
    # s3_file.get(s3_model_index_path, local_path+'model_index.json ', recursive=True)
      
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    

    
    # Run the model
    with torch.inference_mode():
        image = model(prompt,height=height, negative_prompt=negative_prompt, width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64, 'model_id' : model_id, 'user_id' : user_id}



