"""
Image generation using Stable Diffusion with the Hugging Face diffusers library.

Uses the stabilityai/stable-diffusion-2-1 model with PyTorch.
Generates an image from a text prompt and saves it to disk.

Requirements:
  uv pip install diffusers transformers accelerate torch
"""

import torch
from diffusers import StableDiffusionPipeline


def main():
    model_id = "stabilityai/stable-diffusion-2-1"
    print(f"Loading model: {model_id}")
    print("(First run will download ~5 GB of model weights)\n")

    # Use float16 for GPU, float32 for CPU
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
    elif torch.backends.mps.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("mps")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)

    prompt = "a serene mountain landscape at sunset, oil painting style"
    print(f"Generating image for prompt: '{prompt}'")

    image = pipe(prompt, num_inference_steps=30).images[0]
    output_path = "generated_landscape.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
