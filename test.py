# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests



model_inputs = {
    "prompt": "xyz jonny depp",
    "negative_prompt": "blurry, toy, cartoon, animated, underwater, photoshop, bad form, close",
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": None,
    "user_id" : "philip",
    "model_id" : "f85ee87b6dbf4409b00faa4cd9660a83"

}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())


