# ReChar

This is the project page of **ReChar: Revitalising Characters with Structure-Preserved and User-Specified Aesthetic Enhancements.**
<a href="https://01yzzyu.github.io/rechar.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a> 

![mech](https://github.com/01yzzyu/ReChar/blob/main/assets/framework_00.png)	

ReChar integrates three distinct yet interrelated modules: (1) a character structure extraction module, which is designed to preserve the integrity of the character's form, (2) an element generation module, responsible for producing user-defined decorative elements based on textual input, and (3) a style extraction module, aimed at capturing the visual style from a reference image provided by the user. These components are subsequently fused in a controllable synthesis step, which enables flexible and user-customized image generation. To provide a clearer understanding of our approach, we will illustrate the generation process of an instance through a detailed case study.


This repository contains the official implementation of the ReChar method, which enables implicit style-content separation of a single input image for Revitalising Character task. 
Rechar leverages the power of Stable Diffusion XL (SDXL) and Low-Rank Adaptation (LoRA) to disentangle the style and content components of an image, facilitating applications such as image style transfer, text-based image stylization, and consistent style generation.

## 🔧 Important Update 🔧
There were some issues with the new versions of diffusers and PEFT that caused the fine-tuning process to not converge as quickly as desired. In the meantime, we have uploaded the original training script that we used in the paper.

Please note that we used a previous version of diffusers (0.25.0) and did not use PEFT.

## Getting Started

### Prerequisites
- Python 3.11.6+
- PyTorch 2.1.1+
- Other dependencies (specified in `requirements.txt`)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/01yzzyu/ReChar.git
   cd ReChar
   ```

2. Install the required dependencies:
   ```
   Linux
   pip install -r requirements.txt
   ```

   ```
   Window
   Immediately after creating the conda environment, install CUDA PyTorch:

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   Then, pip install -r requirements.txt as normal.

   After that:

   pip uninstall bitsandbytes

   python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
   ```

### Usage

1. **Fine-tuning**

   To train the ReChar for a given input image, run:
   ```
   accelerate launch finetune_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="<path/to/example_images>" \
    --output_dir="<path/to/output_dir>" \
    --instance_prompt="<prompt>" \
    --resolution=1024 \
    --rank=64 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --seed="0" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision="fp16"
      ```
This will optimize the weights for the structure extraction and style extraction and store them in  `output_dir`.
Parameters that need to replace  `instance_data_dir`, `output_dir`, `instance_prompt` (in our paper we use `A [v]`)


2. **Inference**   

   For image stylization based on a reference style image (1) and character structure image (2), run:
   ```
   python inference.py --prompt="A <c> in <s> style" --content_B_LoRA="<path/to/content_B-LoRA>" --style_B_LoRA="<path/to/style_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the content of the first B-LoRA and the style of the second B-LoRA.
   Note that you need to replace `c` and `s` in the prompt according to the optimization prompt.

   Several additional parameters that you can set in the `inference.py` file include:
   1. `--content_alpha`, `--style_alpha` for controlling the strength of the adapters.
   2. `--num_images_per_prompt` for specifying the number of output images.



## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact the authors at [yangzhy21@gmail.com](mailto:yangzhy21@gmail.com).
