#!/usr/bin/env python3
"""
BERT-large Inference Tracing via HuggingFace transformers + torch_hook.

Traces early (0), middle (11), and late (23) encoder layers.
Each layer: attention (self-attention) + output (MLP + residual)

Usage:
  python3 bert_inference_trace.py --batch_size 32 --seq_len 512
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased",
                        help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length (max 512 for BERT)")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to run")
    args = parser.parse_args()

    # Import after env vars are set
    from transformers import BertModel, BertConfig
    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer

    print(f"=== BERT Inference Trace Config ===")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Num batches: {args.num_batches}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    # Load pretrained BERT-large
    print(f"Loading {args.model}...")
    model = BertModel.from_pretrained(args.model)
    model = model.cuda().half().eval()

    # Print model structure
    def print_layer_names(model):
        print("Available layers (showing encoder structure):")
        for name, module in model.named_modules():
            # Show encoder layer structure
            if "encoder.layer.0" in name and name.count('.') <= 4:
                print(f"  {name}: {module.__class__.__name__}")
            elif name in ["embeddings", "encoder", "pooler"]:
                print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # BERT-base has 12 encoder layers (0-11)
    # Each layer: attention (self-attn) + intermediate (MLP expand) + output (MLP contract)
    # We trace attention + output for early/middle/late layers
    layers_to_trace = [
        # Early: layer 0
        "encoder.layer.0.attention",
        "encoder.layer.0.output",
        # Middle: layer 5 (middle of 12)
        "encoder.layer.5.attention",
        "encoder.layer.5.output",
        # Late: layer 11 (last layer)
        "encoder.layer.11.attention",
        "encoder.layer.11.output",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} layers ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Create dummy input (random token IDs, with attention mask)
    print(f"\n=== Preparing input ===")
    # Token IDs: random integers in BERT vocab range (0-30521 for bert-large-uncased)
    input_ids = torch.randint(0, 30522, (args.batch_size, args.seq_len), device='cuda')
    # Attention mask: all 1s (no padding)
    attention_mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.long, device='cuda')

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")

    # Warmup run
    print(f"\n=== Running inference ===")
    print("Warmup run...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()

    # Actual inference
    print(f"Running {args.num_batches} batch(es)...")
    with torch.no_grad():
        for i in range(args.num_batches):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            # outputs.last_hidden_state: [batch, seq_len, hidden_size]
            print(f"  Batch {i+1}/{args.num_batches}: output shape = {outputs.last_hidden_state.shape}")

    print(f"\n=== Tracing complete ===")
    print(f"  Total sequences processed: {args.batch_size * args.num_batches}")
    print(f"  Total tokens processed: {args.batch_size * args.seq_len * args.num_batches}")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
