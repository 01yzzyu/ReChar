import torch
import os
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import cv2
import numpy as np
import gc
from blora_utils import BLOCKS, filter_lora, scale_lora
from tqdm import tqdm
import time

def parse_args():
    """
    Parse command-line arguments for the image generation process.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate images using StableDiffusionXLControlNetPipeline")
    parser.add_argument("--prompt", type=str, required=True, help="The main prompt for image generation")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--content_B_LoRA_path", type=str, required=True, help="Path to the content B-LoRA model")
    parser.add_argument("--style_B_LoRA_path", type=str, required=True, help="Path to the style B-LoRA model")
    parser.add_argument("--control_image_path", type=str, required=True, help="Path to the control image")
    parser.add_argument("--style_prompt", type=str, required=True, help="The style prompt for image generation")
    parser.add_argument("--content_alpha", type=float, default=1.0, help="Scaling factor for content B-LoRA")
    parser.add_argument("--style_alpha", type=float, default=1.0, help="Scaling factor for style B-LoRA")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.5, help="ControlNet conditioning scale")
    parser.add_argument("--guidance_scale", type=float, default=17.5, help="Guidance scale for image generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--ip_adapter_scale", nargs='+', type=float, default=[1.5, 0.8], help="IP-Adapter scaling factors")
    return parser.parse_args()

def generate_images(args):
    """
    Generate images based on the provided arguments using StableDiffusionXLControlNetPipeline.

    Args:
        args: Parsed arguments containing all necessary parameters.
    """
    # Set device
    device = "cuda"

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
    ).to(device)

    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)

    # Load VAE
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Load SDXL ControlNet pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9", 
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16, 
        image_encoder=image_encoder, 
        variant="fp16"
    ).to(device)

    # Prepare mask processor
    processor = IPAdapterMaskProcessor()

    def load_b_lora_to_unet(pipe, content_lora_model_id: str = '', style_lora_model_id: str = '', content_alpha: float = 1.0, style_alpha: float = 1.0) -> None:
        """
        Load B-LoRA weights into the UNet.

        Args:
            pipe: The pipeline object.
            content_lora_model_id: Path to the content B-LoRA model.
            style_lora_model_id: Path to the style B-LoRA model.
            content_alpha: Scaling factor for content B-LoRA.
            style_alpha: Scaling factor for style B-LoRA.
        """
        try:
            # Load Content B-LoRA
            if content_lora_model_id:
                content_B_LoRA_sd, _ = pipe.lora_state_dict(content_lora_model_id)
                content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS['content'])
                content_B_LoRA = scale_lora(content_B_LoRA, content_alpha)
            else:
                content_B_LoRA = {}

            # Load Style B-LoRA
            if style_lora_model_id:
                style_B_LoRA_sd, _ = pipe.lora_state_dict(style_lora_model_id)
                style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
                style_B_LoRA = scale_lora(style_B_LoRA, style_alpha)
            else:
                style_B_LoRA = {}

            # Merge B-LoRAs
            res_lora = {**content_B_LoRA, **style_B_LoRA}

            # Load into UNet
            pipe.load_lora_into_unet(res_lora, None, pipe.unet)
        except Exception as e:
            raise type(e)(f'Failed to load B-LoRA weights: {e}')

    def unload_b_lora_from_unet(pipe):
        """
        Unload B-LoRA weights from the UNet.

        Args:
            pipe: The pipeline object.
        """
        pipe.unload_lora_weights()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare ControlNet input image
    control_image = Image.open(args.control_image_path).convert("RGB")
    control_image = control_image.resize((1024, 1024))
    control_np = np.array(control_image)
    control_edges = cv2.Canny(control_np, 100, 200)
    control_edges = control_edges[:, :, None]
    control_edges = np.concatenate([control_edges, control_edges, control_edges], axis=2)
    control_image = Image.fromarray(control_edges)

    # Generate content image (using content B-LoRA)
    load_b_lora_to_unet(pipe, content_lora_model_id=args.content_B_LoRA_path, content_alpha=args.content_alpha)
    content_prompt = "A [v3] content"
    content_image = pipe(
        prompt=content_prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        image=control_image,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        num_images_per_prompt=1,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale
    ).images[0]

    # Unload content B-LoRA
    unload_b_lora_from_unet(pipe)

    # Generate style image (using style B-LoRA)
    load_b_lora_to_unet(pipe, style_lora_model_id=args.style_B_LoRA_path, style_alpha=args.style_alpha)
    style_image = pipe(
        prompt=args.style_prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        image=control_image,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed + 1),
        num_images_per_prompt=1,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale
    ).images[0]

    # Load IP-Adapter
    pipe.load_ip_adapter(
        ["ostris/ip-composition-adapter", "h94/IP-Adapter"],
        subfolder=["", "sdxl_models"],
        weight_name=[
            "ip_plus_composition_sdxl.safetensors",
            "ip-adapter_sdxl_vit-h.safetensors",
        ],
        image_encoder_folder=None,
    )

    # Set IP-Adapter scale
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    # Create content mask (using Canny edge detection)
    content_np = np.array(content_image)
    content_edges = cv2.Canny(content_np, 100, 200)
    content_mask = Image.fromarray(content_edges)

    # Create style mask (white mask)
    style_mask = Image.new('L', (1024, 1024), 255)

    # Process masks
    masks = processor.preprocess([content_mask, style_mask], height=1024, width=1024)

    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()  # Trigger garbage collection

    try:
        image = pipe(
            prompt=args.prompt,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            ip_adapter_image=[content_image, style_image],            
            image=control_image,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            cross_attention_kwargs={"ip_adapter_masks": masks},
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + 2),
            num_images_per_prompt=1
        ).images[0]

        # Save the generated image
        image_path = os.path.join(args.output_dir, f"generated_image.png")
        image.save(image_path, format='PNG')
        print(f"Generated image saved to {image_path}")

    except RuntimeError as e:
        print(f"Error during image generation: {e}")
        return None

def main():
    """
    Main function to parse arguments and call the image generation function.
    """
    args = parse_args()
    generate_images(args)

if __name__ == "__main__":
    main()
