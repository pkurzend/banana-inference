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
    


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    from dotenv import load_dotenv
    load_dotenv() # take environment variables from .env.

    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # Parse out your arguments
    # prompt = model_inputs.get('prompt', None)
    # negative_prompt = model_inputs.get('negative_prompt', '')
    # height = model_inputs.get('height', 768)
    # width = model_inputs.get('width', 768)
    # num_inference_steps = model_inputs.get('num_inference_steps', 50)
    # guidance_scale = model_inputs.get('guidance_scale', 7.5)
    # input_seed = model_inputs.get("seed",None)

    # with open(f"{user_id}/{model_id}/model_inputs.json", "w") as f:
    #     json.dump(model_inputs, f, indent=4)
    
    OUTPUT_DIR = "stable_diffusion_weights"
    concepts_list = model_inputs.get("concepts_list",None)
    pretrained_model_name_or_path = model_inputs.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-2")
    resolution = model_inputs.get("resolution", 768)
    train_batch_size = model_inputs.get("train_batch_size", 2)
    train_text_encoder = model_inputs.get("train_text_encoder", True)
    gradient_accumulation_steps = model_inputs.get("gradient_accumulation_steps", 1)
    center_crop = model_inputs.get("center_crop", True)
    learning_rate = model_inputs.get("learning_rate", 2e-6)
    num_class_images = model_inputs.get("num_class_images", 50)
    max_train_steps = model_inputs.get("max_train_steps", 1200)
    center_crop = model_inputs.get("center_crop", True)
    center_crop = model_inputs.get("center_crop", True)


    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', '')
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)

    user_id = model_inputs.get("user_id",None)
    model_id = uuid.uuid4().hex


    if user_id == None:
        return {'message': "No user_id provided"}

    if concepts_list == None:
        return {'message': "No concepts_list provided"}

    if prompt == None:
        return {'message': "No prompt provided"}


    
    s3_file = s3fs.S3FileSystem(key=os.getenv("KEY"), secret=os.getenv("SECRET"), client_kwargs={'endpoint_url':'https://s3.eu-central-1.amazonaws.com'})
     

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)
        if c.get('class_data_dir'):
            os.makedirs(c["class_data_dir"], exist_ok=True)

        local_path = f"{c['instance_data_dir']}/"
        s3_path = f"stable-diffusion-finetunings/{user_id}/data/{c['instance_data_dir']}/"
        s3_file.get(s3_path, local_path, recursive=True) 
        
 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    os.system('ls -la')

    command = f"""
    accelerate launch --mixed_precision=fp16 train_dreambooth21.py \
    --pretrained_model_name_or_path={pretrained_model_name_or_path} \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir={OUTPUT_DIR} \
    --revision="fp16" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --seed=1337 \
    --resolution={resolution} \
    --train_batch_size={train_batch_size} \
    {'--train_text_encoder' if train_text_encoder else ''} \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --gradient_accumulation_steps={gradient_accumulation_steps} \
    {'--center_crop' if center_crop else ''} \
    --learning_rate={learning_rate} \
    --lr_scheduler="polynomial" \
    --lr_warmup_steps=0 \
    --num_class_images={num_class_images} \
    --sample_batch_size=4 \
    --max_train_steps={max_train_steps} \
    --save_interval={max_train_steps*10} \
    --concepts_list="concepts_list.json"
    """



    os.system(command)

    os.system('ls -la')

    files_and_folders = os.listdir(OUTPUT_DIR)
    filtered_files = []
    for item in files_and_folders:
        try:
            filtered_files.append(int(item))
        except:
            pass
    model_folder = max(filtered_files)

    model_path = OUTPUT_DIR + '/' + '10'#str(model_folder)


    local_path = f"{model_path}/"
    s3_path = f"stable-diffusion-finetunings/{user_id}/{model_id}"
    s3_file.put(local_path, s3_path, recursive=True) 

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



    # TRAIN_STEPS = 6000
    # !accelerate launch train_dreambooth21.py \
    # --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
    # --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    # --output_dir=$OUTPUT_DIR \
    # --revision="fp16" \
    # --with_prior_preservation --prior_loss_weight=1.0 \
    # --seed=1337 \
    # --resolution=768 \
    # --train_batch_size=2 \
    # --train_text_encoder \
    # --mixed_precision="fp16" \
    # --use_8bit_adam \
    # --gradient_checkpointing \
    # --gradient_accumulation_steps=3 \
    # --center_crop \
    # --learning_rate=1e-6 \
    # --lr_scheduler="polynomial" \
    # --lr_warmup_steps=0 \
    # --num_class_images=500 \
    # --sample_batch_size=4 \
    # --max_train_steps=$TRAIN_STEPS \
    # --save_interval=2000 \
    # --concepts_list="concepts_list.json"


