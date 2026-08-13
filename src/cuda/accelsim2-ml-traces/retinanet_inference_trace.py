#!/usr/bin/env python3
"""
RetinaNet Inference Tracing via torchvision + torch_hook.

Traces backbone (ResNet-50 FPN), FPN neck, and classification/regression heads.

Usage:
  python3 retinanet_inference_trace.py --batch_size 4 --image_size 800
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--image_size", type=int, default=800,
                        help="Input image size")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to run")
    args = parser.parse_args()

    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

    print(f"=== RetinaNet Inference Trace Config ===")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    print("Loading RetinaNet (ResNet50-FPN v2)...")
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    model = model.cuda().half().eval()

    def print_layer_names(model):
        print("Available top-level layers:")
        for name, module in model.named_modules():
            if name and name.count('.') <= 1:
                print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # RetinaNet structure:
    #   backbone.body: ResNet-50 (layer1-4)
    #   backbone.fpn: Feature Pyramid Network
    #   head.classification_head: class prediction
    #   head.regression_head: box regression
    layers_to_trace = [
        # Backbone (early) - ResNet feature extraction
        "backbone.body.layer1",
        "backbone.body.layer4",
        # FPN (middle) - feature pyramid
        "backbone.fpn",
        # Detection heads (late)
        "head.classification_head",
        "head.regression_head",
        # Anchor generator
        "anchor_generator",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} layers ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Create dummy input (list of image tensors, as required by detection models)
    print(f"\n=== Running inference ===")
    dummy_images = [
        torch.randn(3, args.image_size, args.image_size, dtype=torch.float16, device='cuda')
        for _ in range(args.batch_size)
    ]

    print("Warmup run...")
    with torch.no_grad():
        _ = model(dummy_images)
    torch.cuda.synchronize()

    print(f"Running {args.num_batches} batch(es)...")
    with torch.no_grad():
        for i in range(args.num_batches):
            outputs = model(dummy_images)
            torch.cuda.synchronize()
            print(f"  Batch {i+1}/{args.num_batches}: {len(outputs)} detections")

    print(f"\n=== Tracing complete ===")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
