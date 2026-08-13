#!/usr/bin/env python3
"""
Llama 3.1 8B Inference Tracing via vLLM + torch_hook.

Traces early (0), middle (15), and late (31) decoder blocks.
Captures both prefill and decode phase kernels.

Usage (via torch_hook/run.sh):
  ALLOW_REG_VAL_TRACING=1 ./run.sh python3 llama_inference_trace.py [--ctx_len 256]
"""
import argparse
import os
import sys
import torch

# Ensure sibling helpers in util/workloads are importable when this driver
# is launched from another directory (run_hw_trace.py does a chdir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--ctx_len", type=int, default=256,
                        help="Max context / generation length")
    parser.add_argument("--prompt_tokens", type=int, default=100,
                        help="Exact number of prompt tokens (including BOS)")
    parser.add_argument("--gen_tokens", type=int, default=50,
                        help="Exact number of tokens to generate (decode steps = gen_tokens - 1)")
    args = parser.parse_args()

    # Lazy imports so env vars are set before CUDA init
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from _llm_prompt_utils import build_exact_prompt_ids

    print(f"=== Llama Trace Config ===")
    print(f"  Model: {args.model}")
    print(f"  Context length: {args.ctx_len}")
    print(f"  Prompt tokens: ~{args.prompt_tokens}")
    print(f"  Generate tokens: {args.gen_tokens}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    # Create vLLM engine with eager mode (no CUDA graphs, needed for NVBit)
    llm = LLM(
        model=args.model,
        enforce_eager=True,      # Required: CUDA graphs break NVBit tracing
        max_model_len=args.ctx_len,
        gpu_memory_utilization=0.90,
        dtype="half",
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.gen_tokens,
        ignore_eos=True,
    )

    # Helper functions
    def print_layer_names(model):
        """Print all available layer names for reference."""
        print("Available layers:")
        for name, module in model.named_modules():
            if any(k in name for k in ["self_attn", "mlp", "norm", "embed", "lm_head"]):
                print(f"  {name}: {module.__class__.__name__}")

    def apply_hooks(model, layers):
        """Attach NVBit hooks to specific layers."""
        wrapper = TorchModelHookWrapper(model)
        hook_counter = {"count": 0}
        for layer in layers:
            print(f"  Hooking: {layer}")
            def make_pre_hook(name):
                def hook(module, input):
                    hook_counter["count"] += 1
                    print(f"[HOOK] PRE  {name} fired (call #{hook_counter['count']})", flush=True)
                return hook
            def make_post_hook(name):
                def hook(module, input, output):
                    print(f"[HOOK] POST {name} fired", flush=True)
                return hook
            wrapper.register_forward_pre_hook_by_name(layer, make_pre_hook(layer))
            wrapper.register_forward_hook_by_name(layer, make_post_hook(layer))
            hook_nvbit_to_layer(wrapper, layer)
            hook_cuda_profiler_to_layer(wrapper, layer)

    # Print available layers for reference
    llm.collective_rpc(lambda self: print_layer_names(self.model_runner.model))

    # Hook early (0), middle (15), and late (31) decoder blocks
    # Each block: self_attn captures QKV proj + RoPE + FlashAttn + O proj
    #             mlp captures gate_up proj + SiLU + down proj
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

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} sub-layers ===")
    llm.collective_rpc(lambda self: apply_hooks(self.model_runner.model, layers_to_trace))

    # Build a prompt with an EXACT token count (including BOS) using the
    # model's own tokenizer. Feed vLLM the raw IDs so no re-tokenization
    # or auto-BOS insertion drifts the count.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = build_exact_prompt_ids(tokenizer, args.prompt_tokens)
    print(f"\n=== Running inference (prompt = exactly {len(token_ids)} tokens) ===")
    outputs = llm.generate([{"prompt_token_ids": token_ids}], sampling_params)

    for output in outputs:
        gen_text = output.outputs[0].text
        print(f"\nGenerated {len(gen_text.split())} words")
        print(f"First 100 chars: {gen_text[:100]}...")

    print("\n=== Tracing complete ===")
    del llm
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
