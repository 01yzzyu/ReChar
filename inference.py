import argparse
import os
import torch
import gc
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from utils import BLOCKS, filter_lora, scale_lora

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion XL with LoRA and IP-Adapter.")
    parser.add_argument("--prompt", type=str, required=True, help="Base prompt for image generation.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated images.")
    parser.add_argument("--structure_LoRA", type=str, default=None, help="Path for the structure LoRA model.")
    parser.add_argument("--style_LoRA", type=str, default=None, help="Path for the style LoRA model.")
    parser.add_argument("--structure_alpha", type=float, default=1.0, help="Alpha to scale structure LoRA weights.")
    parser.add_argument("--style_alpha", type=float, default=1.0, help="Alpha to scale style LoRA weights.")
    parser.add_argument("--control_image_path", type=str, required=True, help="Path to the control image for ControlNet.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt.")
    return parser.parse_args()

def load_lora_to_unet(pipe, structure_lora_model_id=None, style_lora_model_id=None, structure_alpha=1.0, style_alpha=1.0):
    try:
        structure_LoRA = {}
        style_LoRA = {}

        if structure_lora_model_id:
            structure_LoRA_sd, _ = pipe.lora_state_dict(structure_lora_model_id)
            structure_LoRA = filter_lora(structure_LoRA_sd, BLOCKS['structure'])
            structure_LoRA = scale_lora(structure_LoRA, structure_alpha)

        if style_lora_model_id:
            style_LoRA_sd, _ = pipe.lora_state_dict(style_lora_model_id)
            style_LoRA = filter_lora(style_LoRA_sd, BLOCKS['style'])
            style_LoRA = scale_lora(style_LoRA, style_alpha)

        res_lora = {**structure_LoRA, **style_LoRA}
        pipe.load_lora_into_unet(res_lora, None, pipe.unet)
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA into UNet: {e}")

def unload_lora_from_unet(pipe):
    pipe.unload_lora_weights()

def main():
    args = parse_args()
    
    device = "cuda"
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9", 
        controlnet=controlnet,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    # Prepare input images
    control_image_path = args.control_image_path
    control_image = Image.open(control_image_path).convert("RGB")
    control_image = control_image.resize((1024, 1024))
    control_np = np.array(control_image)
    control_edges = cv2.Canny(control_np, 100, 200)
    control_edges = np.stack([control_edges] * 3, axis=-1)
    control_image = Image.fromarray(control_edges)

    # Load LoRA
    load_lora_to_unet(pipe, args.structure_LoRA, args.style_LoRA, args.structure_alpha, args.style_alpha)

    # Generate images
    processor = IPAdapterMaskProcessor()
    masks = processor.preprocess(
        [control_image, Image.new("L", (1024, 1024), 255)], height=1024, width=1024
    )

    try:
        image = pipe(
            prompt=args.prompt,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            ip_adapter_image=[control_image, control_image],
            image=control_image,
            controlnet_conditioning_scale=0.8,
            cross_attention_kwargs={"ip_adapter_masks": masks},
            guidance_scale=17.5,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(48),
            num_images_per_prompt=args.num_images_per_prompt,
        ).images[0]

        image_path = os.path.join(output_dir, f"{args.prompt.replace(' ', '_')}.png")
        image.save(image_path, format="PNG")
        print(f"Image saved to {image_path}")
    except RuntimeError as e:
        print(f"Error during image generation: {e}")

    # Clean up
    unload_lora_from_unet(pipe)
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
