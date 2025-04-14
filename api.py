import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import argparse

# Import the necessary components from the existing code
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.hooks import apply_group_offloading
from transformers import T5EncoderModel, CLIPTextModel
from src.pipeline_tryon import FluxTryonPipeline
from optimum.quanto import freeze, qfloat8, quantize

# Import utility functions from infer.py
from infer import crop_to_multiple_of_16, resize_and_pad_to_size, resize_by_height

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
pipe = None

def load_models(device=device, torch_dtype=torch_dtype, group_offloading=False):
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    # Enable memory efficient attention
    text_encoder = CLIPTextModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=torch_dtype,)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=torch_dtype,)
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=torch_dtype,)
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=torch_dtype)
    
    pipe = FluxTryonPipeline.from_pretrained(
        bfl_repo,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    )
    
    quantize(pipe.text_encoder_2, weights=qfloat8)
    freeze(pipe.text_encoder_2)    

    # Enable memory efficient attention and VAE optimization
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(
        "loooooong/Any2anyTryon",
        weight_name="dev_lora_any2any_alltasks.safetensors",
        adapter_name="tryon",
    )
    pipe.remove_all_hooks()

    if group_offloading:
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#group-offloading
        apply_group_offloading(
            pipe.transformer,
            offload_type="leaf_level",
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            use_stream=True,
        )
        apply_group_offloading(
            pipe.text_encoder, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
        apply_group_offloading(
            pipe.vae, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )

    pipe.to(device=device)
    return pipe

@torch.no_grad()
def generate_image(prompt, model_image, garment_image, height=512, width=384, seed=0, guidance_scale=3.5, num_inference_steps=30):
    height, width = int(height), int(width)
    width = width - (width % 16)  
    height = height - (height % 16)

    concat_image_list = [np.zeros((height, width, 3), dtype=np.uint8)]
    has_model_image = model_image is not None
    has_garment_image = garment_image is not None
    
    if has_model_image:
        if has_garment_image:
            # if both model and garment image are provided, ensure model image and target image have the same size
            input_height, input_width = model_image.size[1], model_image.size[0]
            model_image, lp, tp, rp, bp = resize_and_pad_to_size(model_image, width, height)
        else:
            model_image = resize_by_height(model_image, height)
        concat_image_list.append(model_image)
    
    if has_garment_image:
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)

    image = np.concatenate([np.array(img) for img in concat_image_list], axis=1)
    image = Image.fromarray(image)
    
    mask = np.zeros_like(np.array(image))
    mask[:,:width] = 255
    mask_image = Image.fromarray(mask)
    
    assert height==image.height, "ensure same height"
    
    output = pipe(
        prompt,
        image=image,
        mask_image=mask_image,
        strength=1.,
        height=height,
        width=image.width,
        target_width=width,
        tryon=has_model_image and has_garment_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator().manual_seed(seed),
        output_type="latent",
    ).images
    
    latents = pipe._unpack_latents(output, image.height, image.width, pipe.vae_scale_factor)
    latents = latents[:,:,:,:width//pipe.vae_scale_factor]
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    if has_model_image and has_garment_image:
        image = image.crop((lp, tp, image.width-rp, image.height-bp)).resize((input_width, input_height))
    
    return image

def image_to_base64(image):
    """Convert a PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/api/tryon', methods=['POST'])
def tryon_endpoint():
    try:
        # Check if files are in the request
        if 'model_image' not in request.files and 'garment_image' not in request.files:
            return jsonify({'error': 'No model or garment image provided'}), 400
        
        # Get images from request
        model_image = None
        garment_image = None
        
        if 'model_image' in request.files:
            model_file = request.files['model_image']
            model_image = Image.open(io.BytesIO(model_file.read()))
        
        if 'garment_image' in request.files:
            garment_file = request.files['garment_image']
            garment_image = Image.open(io.BytesIO(garment_file.read()))
        
        # Get parameters from request
        prompt = request.form.get('prompt', '<MODEL> a person with fashion garment. <GARMENT> a garment. <TARGET> model with fashion garment')
        height = int(request.form.get('height', 576))
        width = int(request.form.get('width', 576))
        seed = int(request.form.get('seed', 0))
        guidance_scale = float(request.form.get('guidance_scale', 3.5))
        num_inference_steps = int(request.form.get('num_inference_steps', 15))
        
        # Generate the image
        result_image = generate_image(
            prompt=prompt,
            model_image=model_image,
            garment_image=garment_image,
            height=height,
            width=width,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        
        # Convert to base64
        base64_image = image_to_base64(result_image)
        
        return jsonify({
            'status': 'success',
            'image': base64_image
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API is running'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--group_offloading', action="store_true")
    args = parser.parse_args()
    
    # Load the model
    print("Loading model...")
    pipe = load_models(group_offloading=args.group_offloading)
    print("Model loaded successfully!")
    
    # Start the Flask app
    app.run(host=args.host, port=args.port, debug=False) 