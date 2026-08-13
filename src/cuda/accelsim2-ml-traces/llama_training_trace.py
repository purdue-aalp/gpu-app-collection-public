#!/usr/bin/env python3
"""
Llama 3.1 8B Training (fine-tuning) Tracing via HuggingFace transformers + torch_hook.

Traces forward and backward passes through early, middle, and late decoder blocks.
Uses standard PyTorch training loop (not vLLM) since vLLM is inference-only.

Usage:
  python3 llama_training_trace.py --batch_size 1 --seq_len 256
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of training steps to trace")
    args = parser.parse_args()

    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer, hook_nvbit_to_layer_backward, hook_cuda_profiler_to_layer_backward
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== Llama 3.1 8B Training Trace Config ===")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Num training steps: {args.num_steps}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",  # No flash attention for NVBit compat
    )
    model = model.cuda().train()

    def print_layer_names(model):
        print("Available layers:")
        for name, module in model.named_modules():
            if any(k in name for k in ["self_attn", "mlp", "norm", "embed", "lm_head"]):
                if name.count('.') <= 3:
                    print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # Same layers as inference, but now we'll capture backward pass too
    layers_to_trace = [
        # Early block
        "model.layers.0.self_attn",
        "model.layers.0.mlp",
        # Middle block
        "model.layers.15.self_attn",
        "model.layers.15.mlp",
        # Late block
        "model.layers.31.self_attn",
        "model.layers.31.mlp",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} sub-layers (forward + backward) ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking (fwd+bwd): {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)
        hook_nvbit_to_layer_backward(wrapper, layer)
        hook_cuda_profiler_to_layer_backward(wrapper, layer)

    # Create dummy training data (random token IDs)
    print(f"\n=== Preparing training data ===")
    input_ids = torch.randint(0, 32000, (args.batch_size, args.seq_len), device='cuda')
    labels = input_ids.clone()
    print(f"  input_ids shape: {input_ids.shape}")

    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Warmup step (without tracing - hooks control this)
    print(f"\n=== Warmup step ===")
    outputs = model(input_ids=input_ids, labels=labels)
    outputs.loss.backward()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    # Actual training steps
    print(f"\n=== Running {args.num_steps} training step(s) ===")
    for step in range(args.num_steps):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print(f"  Step {step+1}/{args.num_steps}: loss = {loss.item():.4f}")

    print(f"\n=== Tracing complete ===")
    del model, optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
