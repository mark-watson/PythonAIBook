# Part IV - Overviews of Image Generation, Reinforcement Learning, and Recommendation Systems

This final part of this book consists of overviews of three important topics that I cover briefly, with perhaps more material added in the next edition of this book.

# Overview of Image Generation

I have never used deep learning image generation at work but I have fun experimenting with both code and model examples, as well as turn-key web apps like DALL·E. In this chapter we look at two approaches to generating images from text prompts: running a model locally with PyTorch, and calling Google's Imagen 4 cloud API.

{width: "80%"}
![Architecture diagram for the Deep Learning Image Generation example](FIG_deep_learning_image_generation.jpg)

The requirements for this chapter are:

```bash
uv add torch diffusers transformers accelerate google-genai Pillow
```

The examples for this chapter are in the directory **source-code/deep_learning_image_generation**.

## Image Generation Using Stable Diffusion and PyTorch

Stable Diffusion is an open-source deep learning model for text-to-image generation. The Hugging Face **diffusers** library makes it straightforward to load and run Stable Diffusion models using PyTorch. Here is a complete example that generates an image from a text prompt:

```python
import torch
from diffusers import DiffusionPipeline

# A smaller model (~1GB) for faster downloading
model_id = "segmind/tiny-sd"
print(f"Loading model: {model_id}")

# Use float16 for GPU/MPS, float32 for CPU
if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
elif torch.backends.mps.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("mps")
else:
    pipe = DiffusionPipeline.from_pretrained(model_id)

prompt = "a serene mountain landscape at sunset, oil painting style"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("generated_landscape.png")
```

The first time you run this code, the model weights (about 1.1 GB) will be downloaded to **~/.cache/huggingface** in your home directory. Subsequent runs use the cached model.

The code automatically detects available hardware: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU. GPU acceleration dramatically speeds up image generation, from minutes on CPU to seconds on a modern GPU.

You can experiment with different prompts, and the `num_inference_steps` parameter controls the quality/speed tradeoff (higher = better quality, slower). Here is sample output running on Apple Silicon:

```bash
$ python image_generation.py
Loading model: segmind/tiny-sd
(First run will download about 1.1 GB of model weights)

Generating image for prompt: 'a serene mountain landscape at sunset, oil painting style'
Image saved to: generated_landscape.png
```

### Understanding the Diffusion Process

Stable Diffusion works by a process called **denoising diffusion**:

1. Start with pure random noise (a tensor of random values).
2. Gradually remove noise over many steps, guided by the text prompt.
3. The result is an image that matches the prompt description.

The text prompt is converted to an embedding vector using a text encoder (CLIP), which guides the denoising process at each step. This is why the same prompt can generate different images with different random seeds.

## Image Generation Using Google's Imagen API

While running models locally gives you full control and privacy, cloud-based image generation APIs offer higher quality results with virtually no setup. Google's **Imagen 4** model is accessible through the Gemini API using the **google-genai** SDK.

The entire example is remarkably concise:

```python
import io
import os

from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = (
    "a serene mountain landscape at sunset,"
    " oil painting style"
)
print(f"Generating image for prompt: '{prompt}'")

response = client.models.generate_images(
    model="imagen-4.0-fast-generate-001",
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images=1,
    ),
)

for generated_image in response.generated_images:
    image = Image.open(
        io.BytesIO(generated_image.image.image_bytes)
    )
    image.save("gemini_generated_landscape.png")
    print("Image saved to: gemini_generated_landscape.png")
```

Compared to the local Stable Diffusion approach, the Gemini API example requires no GPU, no multi-gigabyte model downloads, and no hardware-specific configuration. You just need a `GOOGLE_API_KEY` (available free from [Google AI Studio](https://aistudio.google.com/)).

The `generate_images` method returns image data as raw bytes, which we decode using PIL's `Image.open` with an `io.BytesIO` wrapper. The Imagen 4 model family includes three variants: **Fast** (optimized for speed), **Standard** (balanced), and **Ultra** (maximum fidelity up to 2K resolution). We use the Fast variant here since it produces good results with low latency.

Here is sample output:

```bash
$ python gemini_image_generation.py
Generating image for prompt: 'a serene mountain landscape at sunset, oil painting style'
Image saved to: gemini_generated_landscape.png
```

Here is a sample generated image using Imagen 4:

{width: "50%"}
![Gemini Imagen 4 generated landscape](FIG_emini_generated_landscape.png)

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


If you are interested in the implementation please read the original paper from Open AI [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) before reading the [code for the models](https://github.com/kuprel/min-dalle/tree/main/min_dalle/models).

## Recommended Reading for Image Generation

You can get more information on DALL·E and later versions from [https://openai.com/blog/dall-e/](https://openai.com/blog/dall-e/). You will get much higher quality images using OpenAI's DALL·E web service.

For more advanced image generation with PyTorch, explore:

- The [Hugging Face diffusers documentation](https://huggingface.co/docs/diffusers/) for Stable Diffusion variants, ControlNet, and image-to-image generation.
- [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for higher quality image generation.
- The [PyTorch image generation tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) for understanding GANs from scratch.
- The [Google Imagen documentation](https://ai.google.dev/gemini-api/docs/imagen) for cloud-based image generation with the Gemini API.


## Optional Practice Problems

To help reinforce and expand your understanding of deep learning image generation, try completing the following exercises. You can modify the scripts in the `source-code/deep_learning_image_generation` directory.

### 1. Easy: Parameter Tuning and Prompt Styling
**Objective:** Explore how style keywords and model hyperparameters affect generated images.
- **Tasks:**
  1. Open [image_generation.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/deep_learning_image_generation/image_generation.py). Modify the text prompt to compare three different artistic styles for the same subject (e.g., "a futuristic city skyline" in *photorealistic*, *pixel art*, and *watercolor* styles).
  2. For the local Stable Diffusion model, experiment with the `num_inference_steps` parameter. Run the script with values of `5`, `15`, `30`, and `50`. Measure the execution time for each run and visually inspect the output. What is the minimum number of steps required to produce a recognizable, high-quality image?

### 2. Medium: Batch Generation and Reproducibility
**Objective:** Control randomness and generate multiple image variations in a single run.
- **Tasks:**
  1. Modify [gemini_image_generation.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/deep_learning_image_generation/gemini_image_generation.py) to request `3` images instead of `1` in the `GenerateImagesConfig` object. Update the loop to save all three images as separate files (e.g., `gemini_generated_landscape_1.png`, etc.).
  2. Extend [image_generation.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/deep_learning_image_generation/image_generation.py) to make generation reproducible. Instantiate a PyTorch random number generator with a specific seed and pass it to the pipeline:
     ```python
     generator = torch.Generator(device=pipe.device).manual_seed(42)
     image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
     ```
     Verify that running the script multiple times with the same seed produces the exact same image, whereas changing the seed (e.g., to `43`) produces a distinct composition.

### 3. Hard: Image-to-Image Translation (Img2Img)
**Objective:** Use a local image as the starting point for a new generation.
- **Tasks:**
  1. Create a new script `image_to_image.py` in the source directory.
  2. Instead of starting from random noise, load an existing image (like `generated_landscape.png`) and use a text prompt to transform it. You will need to import `AutoPipelineForImage2Image` (or `StableDiffusionImg2ImgPipeline`) from `diffusers`.
     ```python
     from diffusers import AutoPipelineForImage2Image
     from diffusers.utils import load_image

     # Load pipeline
     pipe = AutoPipelineForImage2Image.from_pretrained(
         "segmind/tiny-sd", torch_dtype=torch.float16
     )
     # Send to GPU / MPS / CPU as in the original script
     ```
  3. Load your input image, resize it if necessary, and run the pipeline with a prompt like "a serene mountain landscape in winter with heavy snow, oil painting style".
  4. Experiment with the `strength` parameter (which ranges from `0.0` to `1.0`). Observe how a strength of `0.2` keeps the image almost identical to the original, while a strength of `0.8` allows the model to completely reimagine the landscape.

