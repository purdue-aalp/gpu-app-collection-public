#!/usr/bin/env python3
"""
ResNet-50 Inference Tracing via torchvision + torch_hook.

Traces early (layer1), middle (layer3), and late (layer4) residual blocks.

Usage:
  python3 resnet_inference_trace.py --batch_size 32 --image_size 224
"""
import argparse
import os
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size (224 for ImageNet)")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to run")
    args = parser.parse_args()

    # Import torch_hook after env vars are set
    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from torchvision import models

    print(f"=== ResNet-50 Trace Config ===")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Num batches: {args.num_batches}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    # Load pretrained ResNet-50
    print("Loading ResNet-50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = model.cuda().half().eval()

    # Print available layers for reference
    def print_layer_names(model):
        print("Available layers:")
        for name, module in model.named_modules():
            if name and '.' not in name:  # Top-level modules only
                print(f"  {name}: {module.__class__.__name__}")
            elif name.count('.') == 1:  # One level deep
                print(f"    {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # Hook early, middle, and late residual blocks
    # ResNet-50 structure:
    #   layer1: 3 Bottleneck blocks (output: 256 channels)
    #   layer2: 4 Bottleneck blocks (output: 512 channels)
    #   layer3: 6 Bottleneck blocks (output: 1024 channels)
    #   layer4: 3 Bottleneck blocks (output: 2048 channels)
    layers_to_trace = [
        # Early: first residual block
        "layer1.0",
        "layer1.2",
        # Middle: middle of layer3
        "layer3.0",
        "layer3.5",
        # Late: last residual blocks
        "layer4.0",
        "layer4.2",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} layers ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Create random input tensor (simulating ImageNet images)
    print(f"\n=== Running inference ===")
    dummy_input = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size,
        dtype=torch.float16, device='cuda'
    )

    # Warmup (without tracing) - already disabled by NVBIT_INSTRUMENTATION_ENABLED=0
    print("Warmup run...")
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()

    # Actual inference runs
    print(f"Running {args.num_batches} batch(es)...")
    with torch.no_grad():
        for i in range(args.num_batches):
            output = model(dummy_input)
            torch.cuda.synchronize()
            print(f"  Batch {i+1}/{args.num_batches}: output shape = {output.shape}")

    print(f"\n=== Tracing complete ===")
    print(f"  Total images processed: {args.batch_size * args.num_batches}")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
