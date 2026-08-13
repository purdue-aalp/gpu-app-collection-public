#!/usr/bin/env python3
"""
DLRMv2 Inference Tracing via torchrec + torch_hook.

Uses the MLPerf Inference DLRMv2 configuration with torchrec's DLRM model.
Traces embedding, interaction, and MLP layers.

Usage:
  python3 dlrmv2_inference_trace.py --batch_size 256
"""
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to run")
    args = parser.parse_args()

    from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer, hook_cuda_profiler_to_layer
    from torchrec.models.dlrm import DLRM, DLRMTrain
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    print(f"=== DLRMv2 Inference Trace Config ===")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {args.num_batches}")
    print(f"  ALLOW_REG_VAL_TRACING: {os.environ.get('ALLOW_REG_VAL_TRACING', 'NOT SET')}")
    print(f"  SPINLOCK_HANDLING_MODE: {os.environ.get('SPINLOCK_HANDLING_MODE', 'NOT SET')}")
    print()

    # MLPerf DLRMv2 configuration:
    # 26 embedding tables, embedding_dim=128, dense features=13
    # Bottom MLP: 13 -> 512 -> 256 -> 128
    # Top MLP: (varies by interaction output) -> 1024 -> 1024 -> 512 -> 256 -> 1
    NUM_EMBEDDING_TABLES = 26
    EMBEDDING_DIM = 128
    NUM_DENSE_FEATURES = 13
    # MLPerf Criteo Terabyte table sizes (using smaller tables for tracing)
    TABLE_SIZES = [1000000] * NUM_EMBEDDING_TABLES

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{i}",
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=TABLE_SIZES[i],
            feature_names=[f"f_{i}"],
        )
        for i in range(NUM_EMBEDDING_TABLES)
    ]

    # Use EmbeddingBagCollection from torchrec
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    ebc = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("cuda"),
    )

    print("Building DLRMv2 model (MLPerf config)...")
    model = DLRM(
        embedding_bag_collection=ebc,
        dense_in_features=NUM_DENSE_FEATURES,
        dense_arch_layer_sizes=[512, 256, EMBEDDING_DIM],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
        dense_device=torch.device("cuda"),
    )
    model = model.cuda().eval()

    def print_layer_names(model):
        print("Available layers:")
        for name, module in model.named_modules():
            if name and name.count('.') <= 2:
                print(f"  {name}: {module.__class__.__name__}")

    print_layer_names(model)

    # Trace: dense MLP (bottom), sparse embeddings, interaction, and top MLP
    layers_to_trace = [
        # Dense arch (bottom MLP)
        "dense_arch",
        # Sparse arch (embedding lookups)
        "sparse_arch",
        # Inter arch (feature interaction)
        "inter_arch",
        # Over arch (top MLP)
        "over_arch",
    ]

    print(f"\n=== Attaching NVBit hooks to {len(layers_to_trace)} layers ===")
    wrapper = TorchModelHookWrapper(model)
    for layer in layers_to_trace:
        print(f"  Hooking: {layer}")
        hook_nvbit_to_layer(wrapper, layer)
        hook_cuda_profiler_to_layer(wrapper, layer)

    # Create dummy input
    print(f"\n=== Preparing input ===")
    dense_features = torch.randn(args.batch_size, NUM_DENSE_FEATURES,
                                 dtype=torch.float32, device='cuda')

    # Sparse features as KeyedJaggedTensor (torchrec format)
    # ~10 lookups per feature per sample
    num_lookups = args.batch_size * 10
    kjt = KeyedJaggedTensor(
        keys=[f"f_{i}" for i in range(NUM_EMBEDDING_TABLES)],
        values=torch.randint(0, 1000000, (num_lookups * NUM_EMBEDDING_TABLES,),
                            device='cuda'),
        lengths=torch.ones(args.batch_size * NUM_EMBEDDING_TABLES,
                          dtype=torch.int32, device='cuda') * 10,
    )

    print("Warmup run...")
    with torch.no_grad():
        _ = model(dense_features, kjt)
    torch.cuda.synchronize()

    print(f"Running {args.num_batches} batch(es)...")
    with torch.no_grad():
        for i in range(args.num_batches):
            output = model(dense_features, kjt)
            torch.cuda.synchronize()
            print(f"  Batch {i+1}/{args.num_batches}: output shape = {output.shape}")

    print(f"\n=== Tracing complete ===")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
