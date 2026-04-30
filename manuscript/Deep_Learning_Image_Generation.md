# Part IV - Overviews of Image Generation, Reinforcement Learning, and Recommendation Systems

This final part of this book consists of overviews of three important topics that I cover briefly, with perhaps more material added in the next edition of this book.

# Overview of Image Generation

I have never used deep learning image generation at work but I have fun experimenting with both code and model examples, as well as turn-key web apps like DALL·E. In this chapter we look at two approaches to generating images from text prompts using PyTorch.

The requirements for this chapter are:

```bash
uv pip install torch diffusers transformers accelerate
```

The examples for this chapter are in the directory **source-code/deep_learning_image_generation**.

## Image Generation Using Stable Diffusion and PyTorch

Stable Diffusion is an open-source deep learning model for text-to-image generation. The Hugging Face **diffusers** library makes it straightforward to load and run Stable Diffusion models using PyTorch. Here is a complete example that generates an image from a text prompt:

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1"
print(f"Loading model: {model_id}")

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
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("generated_landscape.png")
```

The first time you run this code, the model weights (5 GB) will be downloaded to **.cache/huggingface** in your. Subsequent runs use the cached model.

The code automatically detects available hardware: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU. GPU acceleration dramatically speeds up image generation — from minutes on CPU to seconds on a modern GPU.

You can experiment with different prompts, and the `num_inference_steps` parameter controls the quality/speed tradeoff (higher = better quality, slower). Here is sample output running on Apple Silicon:

```bash
$ python image_generation.py
Loading model: stabilityai/stable-diffusion-2-1
(First run will download ~5 GB of model weights)

Generating image for prompt: 'a serene mountain landscape at sunset,
oil painting style'
Image saved to: generated_landscape.png
```

### Understanding the Diffusion Process

Stable Diffusion works by a process called **denoising diffusion**:

1. Start with pure random noise (a tensor of random values).
2. Gradually remove noise over many steps, guided by the text prompt.
3. The result is an image that matches the prompt description.

The text prompt is converted to an embedding vector using a text encoder (CLIP), which guides the denoising process at each step. This is why the same prompt can generate different images with different random seeds.

## Mini-DALL·E: A Lightweight Alternative

For a lighter-weight alternative, Brett Kuprel's [Mini-Dalle model](https://github.com/kuprel/min-dalle) is a reduced size port of DALL·E Mini to PyTorch. It requires less GPU memory and can run on more modest hardware:

```bash
uv pip install min-dalle
```

```python
import torch
from min_dalle import MinDalle

model = MinDalle(
    dtype=torch.float32,
    device='cuda',    # use 'cpu' if no GPU
    is_mega=True,
    is_reusable=True
)

text = "parrot sitting on old man's shoulder"

image = model.generate_image(
    text=text,
    seed=-123,
    grid_size=2,
    temperature=1.5,
    top_k=256,
    supercondition_factor=12
)

image.save(text.replace(" ", "_") + ".png")
```

If **is_mega** is true then a larger model is constructed. If **is_reusable** is true then the same model is reused to create additional images.

You can try changing the temperature (increase for more randomness and differences from training examples), random seed, and text prompt.

Here is a sample generated image:

{width: "50%"}
![Generated image](generated_landscape.png)


The three Python model files in the GitHub repository comprise about 600 lines of code making this a fairly short complete Attention Network/Transformer example. If you are interested in the implementation please read the original paper from Open AI [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) before reading the [code for the models](https://github.com/kuprel/min-dalle/tree/main/min_dalle/models).

## Recommended Reading for Image Generation

You can get more information on DALL·E and later versions from [https://openai.com/blog/dall-e/](https://openai.com/blog/dall-e/). You will get much higher quality images using OpenAI's DALL·E web service.

For more advanced image generation with PyTorch, explore:

- The [Hugging Face diffusers documentation](https://huggingface.co/docs/diffusers/) for Stable Diffusion variants, ControlNet, and image-to-image generation.
- [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for higher quality image generation.
- The [PyTorch image generation tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) for understanding GANs from scratch.

