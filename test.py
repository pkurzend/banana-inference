# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'prompt': 'Hello I am a [MASK] model.'}

model_inputs = {
    "prompt": "xyz jonny depp",
    "negative_prompt": "blurry, toy, cartoon, animated, underwater, photoshop, bad form, close",
    "height": 768,
    "width": 768,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": None,


    "user_id" : "philip",
    "pretrained_model_name_or_path" : "stabilityai/stable-diffusion-2-1-base",
    "resolution" : 512,
    "train_batch_size" : 2,
    "train_text_encoder" : True,
    "gradient_accumulation_steps" : 1,
    "center_crop" : True,
    "learning_rate" : 2e-6,
    "num_class_images" : 5,
    "max_train_steps" : 10,
    "concepts_list" : [    {
        "instance_prompt":      "xyz jonny depp",
        "class_prompt":         "van darkholme face, portrait, cinematic lighting, best quality, high detail, detailed face, masterpiece, high quality, styled hair, eye contact, beautiful eyes, stern face, epic composition, bright background, sparks, neon, 4 k",
        "instance_data_dir":    "jonny_depp",
        "class_data_dir":       "woman_face"
    }],
}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())


