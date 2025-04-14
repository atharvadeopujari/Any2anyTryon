import torch
import numpy as np
from PIL import Image
import gradio as gr

import os
import json
import argparse
import tempfile
import uuid
import subprocess

from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.hooks import apply_group_offloading
from transformers import T5EncoderModel, CLIPTextModel
from src.pipeline_tryon import FluxTryonPipeline
from optimum.quanto import freeze, qfloat8, quantize

device = torch.device("cuda")
torch_dtype = torch.bfloat16 # torch.float16

def load_models(device=device, torch_dtype=torch_dtype,group_offloading=False):
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    # Enable memory efficient attention
    text_encoder = CLIPTextModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=torch_dtype,)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=torch_dtype,)
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=torch_dtype,)
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=torch_dtype)
    # transformer = FluxTransformer2DModel.from_single_file("Kijai/flux-fp8/flux1-dev-fp8.safetensors", torch_dtype=torch_dtype)
    pipe = FluxTryonPipeline.from_pretrained(
        bfl_repo,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    )#.to(device="cpu", dtype=torch_dtype)
    # pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True) # Do not use this if resolution can change
    # # quantize transformer cause severe degration
    # quantize(pipe.transformer, weights=qfloat8)
    # freeze(pipe.transformer)
    
    # Removed quantization and freezing to improve quality
    # quantize(pipe.text_encoder_2, weights=qfloat8)
    # freeze(pipe.text_encoder_2)    
    # pipe.to(device=device)

    # With 80GB GPU, we can disable memory optimizations for better quality
    # pipe.enable_attention_slicing()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # No need for CPU offloading with large GPU memory
    # pipe.enable_model_cpu_offload()
    
    pipe.load_lora_weights(
        "loooooong/Any2anyTryon",
        weight_name="dev_lora_any2any_tryon.safetensors",
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
        # apply_group_offloading(
        #     pipe.text_encoder_2, 
        #     offload_device=torch.device("cpu"),
        #     onload_device=torch.device(device),
        #     offload_type="leaf_level",
        #     use_stream=True,
        # )
        apply_group_offloading(
            pipe.vae, 
            offload_device=torch.device("cpu"),
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )

    pipe.to(device=device)
    return pipe

def crop_to_multiple_of_16(img):
    width, height = img.size
    
    # Calculate new dimensions that are multiples of 8
    new_width = width - (width % 16)  
    new_height = height - (height % 16)
    
    # Calculate crop box coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    
    return cropped_img

def resize_and_pad_to_size(image, target_width, target_height):
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Calculate aspect ratios
    target_ratio = target_width / target_height
    orig_ratio = orig_width / orig_height
    
    # Calculate new dimensions while maintaining aspect ratio
    if orig_ratio > target_ratio:
        # Image is wider than target ratio - scale by width
        new_width = target_width
        new_height = int(new_width / orig_ratio)
    else:
        # Image is taller than target ratio - scale by height
        new_height = target_height
        new_width = int(new_height * orig_ratio)
        
    # Resize image
    resized_image = image.resize((new_width, new_height))
    
    # Create white background image of target size
    padded_image = Image.new('RGB', (target_width, target_height), 'white')
    
    # Calculate padding to center the image
    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2
    
    # Paste resized image onto padded background
    padded_image.paste(resized_image, (left_padding, top_padding))
    
    return padded_image, left_padding, top_padding, target_width - new_width - left_padding, target_height - new_height - top_padding

def resize_by_height(image, height):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # image is a PIL image
    image = image.resize((int(image.width * height / image.height), height))
    return crop_to_multiple_of_16(image)

# @spaces.GPU()
@torch.no_grad
def generate_image(prompt, model_image, garment_image, height=512, width=384, seed=0, guidance_scale=3.5, show_type="follow model image", num_inference_steps=30):
    height, width = int(height), int(width)
    width = width - (width % 16)  
    height = height - (height % 16)

    concat_image_list = [np.zeros((height, width, 3), dtype=np.uint8)]
    has_model_image = model_image is not None
    has_garment_image = garment_image is not None
    if has_model_image:
        if has_garment_image:
            # if both model and garment image are provided, ensure model image and target image have the same size
            input_height, input_width = model_image.shape[:2]
            model_image, lp, tp, rp, bp = resize_and_pad_to_size(Image.fromarray(model_image), width, height)
        else:
            model_image = resize_by_height(model_image, height)
        # model_image = resize_and_pad_to_size(Image.fromarray(model_image), width, height)
        concat_image_list.append(model_image)
    if has_garment_image:
        # if has_model_image:
        #     garment_image = resize_and_pad_to_size(Image.fromarray(garment_image), width, height)
        # else:
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)

    image = np.concatenate([np.array(img) for img in concat_image_list], axis=1)
    image = Image.fromarray(image)
    
    mask = np.zeros_like(image)
    mask[:,:width] = 255
    mask_image = Image.fromarray(mask)
    
    assert height==image.height, "ensure same height"
    # with torch.cuda.amp.autocast(): # this cause black image
    # with torch.no_grad():
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
    if show_type!="all outputs":
        latents = latents[:,:,:,:width//pipe.vae_scale_factor]
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    output = image
    if show_type=="follow model image" and has_model_image and has_garment_image:
        output = output.crop((lp, tp, output.width-rp, output.height-bp)).resize((input_width, input_height))
    
    return output

def update_dimensions(model_image, garment_image, height, width, auto_ar):
    if not auto_ar:
        return height, width
    if model_image is not None:
        height = model_image.shape[0]
        width = model_image.shape[1]
    elif garment_image is not None:
        height = garment_image.shape[0]
        width = garment_image.shape[1]
    else:
        height = 512
        width = 384

    # Set max dimensions and minimum size
    max_height = 1024
    max_width = 1024
    min_size = 384

    # Scale down if exceeds max dimensions while maintaining aspect ratio
    if height > max_height or width > max_width:
        aspect_ratio = width / height
        if height > max_height:
            height = max_height
            width = int(height * aspect_ratio)
        if width > max_width:
            width = max_width
            height = int(width / aspect_ratio)

    # Scale up if below minimum size while maintaining aspect ratio
    if height < min_size and width < min_size:
        aspect_ratio = width / height
        if height < width:
            height = min_size
            width = int(height * aspect_ratio)
        else:
            width = min_size
            height = int(width / aspect_ratio)

    return height, width

model1 = Image.open("asset/images/model/model1.png") 
model2 = Image.open("asset/images/model/model2.jpg")
model3 = Image.open("asset/images/model/model3.png") 
model4 = Image.open("asset/images/model/model4.png")

garment1 = Image.open("asset/images/garment/garment1.jpg") 
garment2 = Image.open("asset/images/garment/garment2.jpg")
garment3 = Image.open("asset/images/garment/garment3.jpg") 
garment4 = Image.open("asset/images/garment/garment4.jpg")

def enhance_with_realesrgan(image, outscale=3.5, face_enhance=True):
    """
    Enhance an image using Real-ESRGAN
    
    Args:
        image (PIL.Image): The image to enhance
        outscale (float): Output scale factor
        face_enhance (bool): Whether to enhance faces (always True now)
    
    Returns:
        PIL.Image: The enhanced image
    """
    # Create temporary files for input and output
    temp_dir = tempfile.gettempdir()
    random_id = str(uuid.uuid4())
    input_path = os.path.join(temp_dir, f"input_{random_id}.png")
    output_path = os.path.join(temp_dir, f"output_{random_id}.png")
    
    # Save input image
    image.save(input_path)
    
    # Prepare command - always use face enhancement
    cmd = [
        "python", "inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", input_path,
        "-o", temp_dir,
        "--outscale", str(outscale),
        "--suffix", random_id,
        "--face_enhance"  # Always enable face enhancement
    ]
    
    # Run Real-ESRGAN
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load enhanced image
        enhanced_path = os.path.join(temp_dir, f"output_{random_id}.png")
        if os.path.exists(enhanced_path):
            enhanced_image = Image.open(enhanced_path)
            
            # Clean up
            try:
                os.remove(input_path)
                os.remove(enhanced_path)
            except:
                pass
            
            return enhanced_image
        else:
            print(f"Enhanced image not found at {enhanced_path}")
            return image
    except subprocess.CalledProcessError as e:
        print(f"Error running Real-ESRGAN: {e.stderr.decode()}")
        return image
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        return image

def launch_demo():
    with gr.Blocks() as demo:   
        gr.Markdown("# Any2AnyTryon")
        gr.Markdown("Demo(experimental) for [Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks](https://arxiv.org/abs/2501.15891) ([Code](https://github.com/logn-2024/Any2anyTryon)).") 
        with gr.Row():
            with gr.Column():
                model_image = gr.Image(label="Model Image", type="numpy", interactive=True,)
                with gr.Row():
                    garment_image = gr.Image(label="Garment Image", type="numpy", interactive=True,)
                    with gr.Column():
                        with gr.Row():
                            height = gr.Number(label="Height", value=768, precision=0)
                            width = gr.Number(label="Width", value=576, precision=0)
                        seed = gr.Number(label="Seed", value=0, precision=0)
                        with gr.Accordion("Advanced Settings", open=False):
                            guidance_scale = gr.Number(label="Guidance Scale", value=3.5)
                            num_inference_steps = gr.Number(label="Inference Steps", value=15)
                            show_type = gr.Radio(label="Show Type",choices=["follow model image", "follow height & width", "all outputs"],value="follow model image")
                            auto_ar = gr.Checkbox(label="Detect Image Size(From Uploaded Images)", value=False, visible=True,)
                            
                            # Real-ESRGAN Enhancement Settings - removed face_enhance checkbox
                            enable_enhancement = gr.Checkbox(label="Enable Real-ESRGAN Enhancement", value=True)
                            esrgan_scale = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, label="Enhancement Scale")
                btn = gr.Button("Generate")
            
            with gr.Column():
                output = gr.Image(label="Generated Image")
                # Removed example prompts since we're using a fixed prompt
                example_model = gr.Examples(
                    examples=[
                        model1, model2, model3, model4
                    ],
                    inputs=model_image,
                    label="Example Model Images"
                )
                example_garment = gr.Examples(
                    examples=[
                        garment1, garment2, garment3, garment4
                    ],
                    inputs=garment_image,
                    label="Example Garment Images"
                )

        # Update dimensions when images change
        model_image.change(fn=update_dimensions, 
                        inputs=[model_image, garment_image, height, width, auto_ar],
                        outputs=[height, width])
        garment_image.change(fn=update_dimensions,
                            inputs=[model_image, garment_image, height, width, auto_ar], 
                            outputs=[height, width])
        
        # Use a wrapper function that adds the fixed prompt - always use face enhancement
        def generate_with_fixed_prompt(model_image, garment_image, height, width, seed, guidance_scale, show_type, num_inference_steps, enable_enhancement, esrgan_scale):
            fixed_prompt = "<MODEL> a person with fashion garment. <GARMENT> a garment. <TARGET> model with fashion garment"
            
            # Generate the image using the Any2AnyTryon model
            generated_image = generate_image(fixed_prompt, model_image, garment_image, height, width, seed, guidance_scale, show_type, num_inference_steps)
            
            # Enhance the generated image using Real-ESRGAN if enabled - always use face enhancement
            if enable_enhancement:
                try:
                    print(f"Enhancing image with Real-ESRGAN (scale: {esrgan_scale}, face_enhance: True)...")
                    enhanced_image = enhance_with_realesrgan(generated_image, outscale=esrgan_scale, face_enhance=True)
                    print("Enhancement complete!")
                    return enhanced_image
                except Exception as e:
                    print(f"Error during enhancement: {str(e)}")
                    # If enhancement fails, return the original generated image
                    return generated_image
            else:
                return generated_image
            
        btn.click(fn=generate_with_fixed_prompt,
                inputs=[model_image, garment_image, height, width, seed, guidance_scale, show_type, num_inference_steps, enable_enhancement, esrgan_scale],
                outputs=output)

        demo.title = "FLUX Image Generation Demo"
        demo.description = "Generate images using FLUX model with LoRA"
        
        examples = [
            # tryon
            [
                model1,
                garment1,
                768, 576,
                0,  # seed
                3.5,  # guidance_scale
                "follow model image",  # show_type
                15,  # num_inference_steps
                True,  # enable_enhancement
                2.0,  # esrgan_scale
            ],
            [
                model2,
                garment2,
                768, 576,
                0,  # seed
                3.5,  # guidance_scale
                "follow model image",  # show_type
                15,  # num_inference_steps
                True,  # enable_enhancement
                2.0,  # esrgan_scale
            ],
            [
                model3,
                garment3,
                768, 576,
                0,  # seed
                3.5,  # guidance_scale
                "follow model image",  # show_type
                15,  # num_inference_steps
                True,  # enable_enhancement
                2.0,  # esrgan_scale
            ],
            [
                model4,
                garment4,
                768, 576,
                0,  # seed
                3.5,  # guidance_scale
                "follow model image",  # show_type
                15,  # num_inference_steps
                True,  # enable_enhancement
                2.0,  # esrgan_scale
            ],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[model_image, garment_image, height, width, seed, guidance_scale, show_type, num_inference_steps, enable_enhancement, esrgan_scale],
            outputs=output,
            fn=generate_with_fixed_prompt,
            cache_examples=False,
            examples_per_page=20
        )
    demo.queue().launch(share=True, show_error=False,
        server_name="0.0.0.0"
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_offloading', action="store_true")
    args=parser.parse_args()
    
    # Check for Real-ESRGAN dependencies
    if not os.path.exists("inference_realesrgan.py"):
        print("Warning: inference_realesrgan.py not found. Real-ESRGAN enhancement will not work.")
        print("Run setup_realesrgan.sh to set up Real-ESRGAN.")
    
    if not os.path.exists("weights/RealESRGAN_x4plus.pth"):
        print("Warning: RealESRGAN model weights not found. Enhancement will not work.")
        print("Run setup_realesrgan.sh to download the model weights.")
    
    pipe = load_models(group_offloading=args.group_offloading)
    launch_demo()