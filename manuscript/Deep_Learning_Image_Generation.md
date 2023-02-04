# Part IV - Overviews of Image Generation, Reinforcement Learning, and Recommendation Systems

This final part of this book consists of overviews of three important topics that I cover briefly, with perhaps more material added in the next edition of this book.

# Overview of Image Generation

I have never used deep learning image generation at work but I have fun experimenting with both code and model examples, as well as turn-key web apps like DALL·E. We will use Brett Kuprel's [Mini-Dalle model](https://github.com/kuprel/min-dalle) GitHub repository that is a reduced size port of DALL·E Mini to PyTorch.

You can run this example directly on [Google Colab](https://colab.research.google.com/drive/1FxTaCCVtLWUfvHKvcgnwAerJtq5a6KSX?usp=sharing). Here is a listing of the example code in this notebook:

```python
!pip install min-dalle

import os

from IPython.display import display, update_display
import torch
from min_dalle import MinDalle

dtype = "float32" 

model = MinDalle(
    dtype=getattr(torch, dtype),
    device='cuda',
    is_mega=True, 
    is_reusable=True
)

directory = "/content"

for root, subdirectories, files in os.walk(directory):

    for filename in files:
        if filename.endswith(".png"):
          path_img = os.path.join(root, filename)
          os.remove(path_img)
          
text = "parrot sitting on old man's shoulder"

image_stream = model.generate_image_stream(
    text=text,
    seed=-123,
    grid_size=2,
    progressive_outputs=True,
    is_seamless=False,
    temperature=1.5,
    top_k=int(256),
    supercondition_factor=float(12)
)

for image in image_stream:
    display(image, display_id=1)
    # optional:
    image.save("./"+text.replace(" ", "_")+".png")
```

The pre-trained model files will be downloaded the first time you run this code. We create a class instance in lines 11-16. If **is_mega** is true then a larger model is constructed. If **is_reusable** is true then the same model is reused to create additional images.

The example prompt text "parrot sitting on old man's shoulder" set in line 27 can be changed to whatever you want to try.

You can try changing the temperature (increase for more randomness and differences from training examples), random seed, and text prompt. This is a generated image containing four images (because we set the output image grid size to 2):

{width: "50%"}
![](omparrot.png)

I reduced the above image size by a factor of four in order to keep the size of this eBook fairly small. When you run this example you will get higher resolution images.

You will get different results even without changing the random seed or parameters. Here is sample output from the second time I ran this example on Google Colab:

{width: "50%"}
![](omparrot-2.png)

I also reduced the last image size by a factor of four for inclusion in this chapter.

The three Python model files in the GitHub repository comprise about 600 lines of code making this a fairly short complete Attention Network/Transformer example. We will not walk through the code here but if your are interested in the implementation please read the original paper from Open AI [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) before reading the [code for the models](https://github.com/kuprel/min-dalle/tree/main/min_dalle/models). 

## Recommended Reading for Image Generation

The example program is small enough to run on Google Colab or on your laptop (you may want to reduce the value **top_k=int(256)** to 128 if you are not using a GPU with 16G of video RAM.

You can get more information on DALL·E and DALL·E 2 from [https://openai.com/blog/dall-e/](https://openai.com/blog/dall-e/). You will get much higher quality images using OpenAI's DALL·E web service.

We won't cover StyleGAN (created by researchers at NVIDIA) here because it is almost two year old technology as I am writing this chapter but I recommend experimenting with it using the [TensorFlow/Keras StyleGAN example](https://keras.io/examples/generative/stylegan/). StyleGAN can progressively increase the resolution of images. StyleGAN can also mix styles from multiple images to create a new image.
