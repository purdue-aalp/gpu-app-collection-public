#!/usr/bin/env python3
"""
Stable Diffusion XL Inference Tracing via diffusers + torch_hook.

Traces UNet down/mid/up blocks and cross-attention layers.

Usage:
  python3 sdxl_inference_trace.py --num_steps 20 --image_size 1024
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of denoising steps")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Output image size (1024 for SDXL)")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to generate")
    args = parser.parse_args()

    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from diffusers import StableDiffusionXLPipeline

    print(f"=== Stable Diffusion XL Inference Trace Config ===")
    print(f"  Num denoising steps: {args.num_steps}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Num images: {args.num_images}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    print("Loading Stable Diffusion XL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # Fix diffusers compat: SDXL checkpoint sets name="Conv2d_0" in upsamplers
    # but the weight is loaded as self.conv. Forward checks self.name and
    # tries self.Conv2d_0 which doesn't exist. Add alias.
    for block in pipe.unet.up_blocks:
        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            for upsampler in block.upsamplers:
                if hasattr(upsampler, 'conv') and not hasattr(upsampler, 'Conv2d_0'):
                    upsampler.Conv2d_0 = upsampler.conv
    pipe = pipe.to("cuda")

    # Disable CUDA graphs for NVBit compatibility
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False) if False else pipe.unet

    unet = pipe.unet

    def print_layer_names(model):
        print("Available UNet layers:")
        for name, module in model.named_modules():
            if name and name.count('.') <= 1:
                print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(unet)

    # SDXL UNet structure:
    #   down_blocks: 3 down-sampling blocks (CrossAttnDownBlock2D)
    #   mid_block: middle block with cross-attention
    #   up_blocks: 3 up-sampling blocks (CrossAttnUpBlock2D)
    layers_to_trace = [
        # Down path
        "down_blocks.0.resnets.0",          # early ResNet (DownBlock2D, no attention)
        "down_blocks.1.resnets.0",          # mid-depth ResNet
        "down_blocks.1.attentions.0",       # first cross-attention (CrossAttnDownBlock2D)
        "down_blocks.2.resnets.0",          # deepest ResNet
        "down_blocks.2.attentions.0",       # deepest cross-attention (smallest spatial)
        # Bottleneck
        "mid_block.resnets.0",              # bottleneck ResNet
        "mid_block.attentions.0",           # bottleneck cross-attention
        # Up path
        "up_blocks.0.resnets.0",            # first upsampling ResNet
        "up_blocks.0.attentions.0",         # first upsampling cross-attention
        "up_blocks.2.resnets.0",            # final ResNet (UpBlock2D, no attention)
        # Time embedding (small, fast)
        "time_embedding",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} UNet layers ===")
    wrapper = TorchModelHookWrapper(unet)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Run inference
    print(f"\n=== Running SDXL inference ({args.num_steps} steps) ===")
    prompt = "A photorealistic landscape with mountains, a lake, and sunset sky"

    images = pipe(
        prompt=prompt,
        num_inference_steps=args.num_steps,
        height=args.image_size,
        width=args.image_size,
        num_images_per_prompt=args.num_images,
    ).images

    print(f"\nGenerated {len(images)} image(s) of size {images[0].size}")

    print(f"\n=== Tracing complete ===")
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
