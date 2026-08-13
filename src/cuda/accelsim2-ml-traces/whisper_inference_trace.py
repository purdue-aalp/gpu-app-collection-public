#!/usr/bin/env python3
"""
Whisper Large V3 Inference Tracing via HuggingFace transformers + torch_hook.

Traces encoder and decoder layers.
Whisper is an encoder-decoder model: audio encoder + text decoder.

Usage:
  python3 whisper_inference_trace.py --audio_len 30
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3",
                        help="HuggingFace model ID")
    parser.add_argument("--audio_len", type=int, default=30,
                        help="Audio length in seconds (max 30 for Whisper)")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to run")
    args = parser.parse_args()

    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"=== Whisper Large V3 Inference Trace Config ===")
    print(f"  Model: {args.model}")
    print(f"  Audio length: {args.audio_len}s")
    print(f"  Num batches: {args.num_batches}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    print(f"Loading {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    model = model.cuda().eval()

    def print_layer_names(model):
        print("Available layers:")
        for name, module in model.named_modules():
            if name and name.count('.') <= 2:
                print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # Whisper Large V3: 32 encoder layers + 32 decoder layers
    # Encoder: model.encoder.layers.{0..31}
    # Decoder: model.decoder.layers.{0..31}
    layers_to_trace = [
        # Encoder (early)
        "model.encoder.layers.0",
        # Encoder (late)
        "model.encoder.layers.31",
        # Decoder (early)
        "model.decoder.layers.0",
        # Decoder (middle)
        "model.decoder.layers.15",
        # Decoder (late)
        "model.decoder.layers.31",
        # Projection head
        "proj_out",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} layers ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Create dummy mel spectrogram input (what Whisper expects)
    # Whisper processes 30s audio chunks -> 80-channel mel spectrogram, 3000 frames
    print(f"\n=== Running inference ===")
    # Generate fake audio at 16kHz
    import numpy as np
    fake_audio = np.random.randn(args.audio_len * 16000).astype(np.float32)
    input_features = processor(
        fake_audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.cuda().half()
    print(f"  Input features shape: {input_features.shape}")

    print("Warmup run...")
    with torch.no_grad():
        _ = model.generate(input_features, max_new_tokens=1)
    torch.cuda.synchronize()

    print(f"Running {args.num_batches} batch(es)...")
    with torch.no_grad():
        for i in range(args.num_batches):
            generated_ids = model.generate(
                input_features,
                max_new_tokens=50,
            )
            torch.cuda.synchronize()
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"  Batch {i+1}/{args.num_batches}: generated {generated_ids.shape[1]} tokens")
            print(f"  Transcription: {transcription[0][:100]}...")

    print(f"\n=== Tracing complete ===")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
