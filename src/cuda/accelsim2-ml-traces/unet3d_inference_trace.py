#!/usr/bin/env python3
"""
3D-UNet Inference Tracing via MONAI + torch_hook.

Uses MONAI's pretrained 3D-UNet for medical image segmentation.
Traces encoder, bottleneck, and decoder blocks.

Usage:
  python3 unet3d_inference_trace.py --batch_size 1 --input_size 128
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (typically 1-2 for 3D volumes)")
    parser.add_argument("--input_size", type=int, default=128,
                        help="Input volume size (D=H=W)")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Number of input channels")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to run")
    args = parser.parse_args()

    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from monai.networks.nets import UNet

    print(f"=== 3D-UNet (MONAI) Inference Trace Config ===")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input size: {args.input_size}^3")
    print(f"  In channels: {args.in_channels}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    # MLPerf 3D-UNet config for medical image segmentation (BraTS/KiTS style)
    # 5 encoder levels, instance norm, LeakyReLU
    print("Building 3D-UNet model (MONAI, MLPerf-style config)...")
    model = UNet(
        spatial_dims=3,
        in_channels=args.in_channels,
        out_channels=3,  # 3 classes (e.g., background, tumor core, enhancing)
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
        act="leakyrelu",
    )
    model = model.cuda().half().eval()

    def print_layer_names(model):
        print("Available layers:")
        for name, module in model.named_modules():
            if name and name.count('.') <= 1:
                print(f"  {name}: {module.__class__.__name__}")
        print()
        print("Deeper structure (2 levels):")
        for name, module in model.named_modules():
            if name and name.count('.') <= 2:
                print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # MONAI UNet structure:
    #   model.0: initial down conv (encoder level 0)
    #   model.1.submodule.0: encoder level 1
    #   model.1.submodule.1.submodule.0: encoder level 2
    #   ... (nested structure)
    #   model.1.submodule.1.submodule.1.submodule.0: encoder level 3
    #   model.1.submodule.1.submodule.1.submodule.1.submodule: bottleneck
    #   model.1.submodule.1.submodule.1.submodule.1: decoder level 3 (up+cat+conv)
    #   model.1.submodule.1.submodule.1: decoder level 2
    #   model.1.submodule.1: decoder level 1
    #   model.1: decoder level 0
    #   model.2: final 1x1x1 conv
    layers_to_trace = [
        # Encoder (early) - first downsampling
        "model.0",
        # Encoder (middle) - deeper level
        "model.1.submodule.0",
        # Bottleneck
        "model.1.submodule.1.submodule.1.submodule.1.submodule",
        # Decoder (middle)
        "model.1.submodule.1",
        # Decoder (late) - final upsampling
        "model.1",
        # Final conv
        "model.2",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} layers ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Create dummy 3D volume input
    print(f"\n=== Running inference ===")
    s = args.input_size
    dummy_input = torch.randn(args.batch_size, args.in_channels, s, s, s,
                              dtype=torch.float16, device='cuda')

    print("Warmup run...")
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()

    print(f"Running {args.num_batches} batch(es)...")
    with torch.no_grad():
        for i in range(args.num_batches):
            output = model(dummy_input)
            torch.cuda.synchronize()
            print(f"  Batch {i+1}/{args.num_batches}: output shape = {output.shape}")

    print(f"\n=== Tracing complete ===")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
