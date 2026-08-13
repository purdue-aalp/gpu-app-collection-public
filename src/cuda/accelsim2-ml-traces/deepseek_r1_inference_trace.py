#!/usr/bin/env python3
"""
DeepSeek R1 Inference Tracing via vLLM + torch_hook.

Traces early, middle, and late decoder blocks.
DeepSeek R1 uses MoE (Mixture of Experts) with shared experts.

Usage:
  python3 deepseek_r1_inference_trace.py --ctx_len 256 --prompt_tokens 255 --gen_tokens 1
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--ctx_len", type=int, default=256,
                        help="Max context / generation length")
    parser.add_argument("--prompt_tokens", type=int, default=255,
                        help="Exact number of prompt tokens (including BOS)")
    parser.add_argument("--gen_tokens", type=int, default=1,
                        help="Exact number of tokens to generate (decode steps = gen_tokens - 1)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from _llm_prompt_utils import build_exact_prompt_ids

    print(f"=== DeepSeek R1 Trace Config ===")
    print(f"  Model: {args.model}")
    print(f"  Context length: {args.ctx_len}")
    print(f"  Prompt tokens: ~{args.prompt_tokens}")
    print(f"  Generate tokens: {args.gen_tokens}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.ctx_len,
        gpu_memory_utilization=0.90,
        dtype="half",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.gen_tokens,
        ignore_eos=True,
    )

    def print_layer_names(model):
        print("Available layers:")
        for name, module in model.named_modules():
            if any(k in name for k in ["self_attn", "mlp", "norm", "embed", "lm_head"]):
                if name.count('.') <= 3:
                    print(f"  {name}: {module.__class__.__name__}")

    def apply_hooks(model, layers):
        wrapper = TorchModelHookWrapper(model)
        for layer in layers:
            print(f"  Hooking: {layer}")
            hook_nvbit_to_layer(wrapper, layer)
            hook_cuda_profiler_to_layer(wrapper, layer)

    llm.collective_rpc(lambda self: print_layer_names(self.model_runner.model))

    # DeepSeek-R1-Distill-Llama-8B has 32 layers (Llama architecture)
    # The full DeepSeek R1 has MoE; the distilled version uses dense MLPs
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
