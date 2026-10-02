// Standalone GPGPU-Sim Programmatic Dependent Launch (PDL) regression tests:
//   explicit_wait:   early secondary execution and a wait that blocks reads.
//   all_cta_trigger: every CTA must signal; duplicate triggers do not count.
//   exit_arrival:    a CTA that exits without triggering counts as an arrival.
//   no_attribute:    ordinary same-stream serialization remains intact.
//
// The first three cases require observed overlap, using two small primary CTAs
// and a secondary CTA. Use the prepared Hopper timing configuration with spare
// SMs. CUDA hardware may legally serialize PDL grids, so these strict overlap
// checks are simulator regressions, not portable hardware conformance tests.
// All primary delays are bounded; no primary waits for the secondary to run.
//
// Run all cases, or select one:
//   ./griddepcontrol_semantics
//   ./griddepcontrol_semantics --case explicit_wait
// Negative control (must fail explicit_wait/exit_arrival under the same config):
//   ./griddepcontrol_semantics --case explicit_wait --skip-wait
// The runtime skip preserves a wait instruction in the kernel's PTX so that
// GPGPU-Sim still recognizes it as eligible for an early dependent launch.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kPrimaryBlocks = 2;
constexpr int kPrimaryThreads = 32;
constexpr int kSecondaryThreads = 128;
constexpr int kDelayIterations = 4000;
constexpr int kTriggerDelayIterations = 1000;
constexpr int kLateMagic = 0x5a170000;

struct Witness {
  int late[kPrimaryBlocks];
  int done[kPrimaryBlocks];
  int secondary_started;
  unsigned stale_mask;
  int errors;
  int delayed_cta_ready;
  int premature_entries;
};

enum class PrimaryKind { Explicit, Uneven, MixedExit };

struct TestCase {
  const char *name;
  PrimaryKind primary;
  bool pdl_attribute;
  // In addition to any stale value, require these specific CTA bits to be stale.
  unsigned required_stale_mask;
};

const TestCase kCases[] = {
    {"explicit_wait", PrimaryKind::Explicit, true, 0},
    {"all_cta_trigger", PrimaryKind::Uneven, true, 1u << 1},
    {"exit_arrival", PrimaryKind::MixedExit, true, 1u << 0},
    {"no_attribute", PrimaryKind::Explicit, false, 0},
};

__device__ __forceinline__ int late_value(int cta) {
  return kLateMagic ^ cta;
}

__device__ __forceinline__ void burn_cycles(int iterations) {
  volatile unsigned value = threadIdx.x + blockIdx.x;
  for (int i = 0; i < iterations; ++i) {
    value = value * 1664525u + 1013904223u;
  }
}

__device__ __forceinline__ void publish_results(Witness *witness) {
  if (threadIdx.x == 0) {
    witness->late[blockIdx.x] = late_value(blockIdx.x);
    // This fence cannot protect an early secondary read: the write itself is
    // delayed. It preserves the original targeted publication witness; done
    // is written afterward and must also be visible when the dependency ends.
    __threadfence();
    witness->done[blockIdx.x] = 1;
  }
}

// Cases 1 and 4: every CTA signals, delays, and then publishes its result.
__global__ void primary_trigger_then_publish(Witness *witness) {
  if (threadIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }

  burn_cycles(kDelayIterations);

  publish_results(witness);
}

// Case 2: CTA 0 signals twice before CTA 1 is ready. The two signals must
// contribute only one logical CTA arrival. CTA 1 observes secondary entry
// before its trigger; the secondary also checks CTA 1's readiness before wait.
__global__ void primary_uneven_triggers(Witness *witness) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
      cudaTriggerProgrammaticLaunchCompletion();
    }
  } else {
    burn_cycles(kTriggerDelayIterations);
    if (threadIdx.x == 0) {
      if (atomicAdd(&witness->secondary_started, 0) != 0) {
        atomicAdd(&witness->premature_entries, 1);
      }
      // Atomics make this scheduling witness visible independently of PDL's
      // memory guarantee. Publication immediately precedes the delayed signal.
      atomicExch(&witness->delayed_cta_ready, 1);
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }

  burn_cycles(kDelayIterations);
  publish_results(witness);
}

// Case 3: CTA 1 never signals and returns uniformly. CTA 0 signals but stays
// alive, so failure to count CTA 1's exit cannot hide behind kernel completion.
// On the prepared configuration CTAs occupy different SMs; both hardware CTA
// slots can be zero, making logical grid CTA identity essential for counting.
__global__ void primary_mixed_trigger_and_exit(Witness *witness) {
  if (blockIdx.x == 1) {
    if (threadIdx.x == 0) {
      witness->late[blockIdx.x] = late_value(blockIdx.x);
      witness->done[blockIdx.x] = 1;
    }
    return;
  }

  if (threadIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
  burn_cycles(kDelayIterations);
  publish_results(witness);
}

__global__ void secondary_snapshot_then_wait(Witness *witness, bool skip_wait,
                                             bool check_delayed_cta) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    atomicExch(&witness->secondary_started, 1);
    if (check_delayed_cta &&
        atomicAdd(&witness->delayed_cta_ready, 0) != 1) {
      atomicAdd(&witness->premature_entries, 1);
    }
  }

  // Volatile ensures actual, separate loads on each side of the wait. These
  // intentionally unsynchronized pre-wait snapshots are only diagnostics;
  // primary results are consumed for correctness after the dependency wait.
  const volatile int *late = witness->late;
  const volatile int *done = witness->done;
  if (tid < kPrimaryBlocks && late[tid] != late_value(tid)) {
    atomicOr(&witness->stale_mask, 1u << tid);
  }

  if (!skip_wait) {
    cudaGridDependencySynchronize();
  }

  // A real wait releases only after the primary completes and its global
  // writes are visible.  All late values and completion witnesses are then
  // required to be present.
  if (tid < kPrimaryBlocks &&
      (late[tid] != late_value(tid) || done[tid] != 1)) {
    atomicAdd(&witness->errors, 1);
  }
}

bool run_case(const TestCase &test, bool skip_wait) {
  Witness *d_witness = nullptr;
  cudaStream_t stream = nullptr;

  CHECK_CUDA(cudaMalloc(&d_witness, sizeof(*d_witness)));
  CHECK_CUDA(cudaMemset(d_witness, 0, sizeof(*d_witness)));
  CHECK_CUDA(cudaStreamCreate(&stream));

  switch (test.primary) {
    case PrimaryKind::Explicit:
      primary_trigger_then_publish<<<kPrimaryBlocks, kPrimaryThreads, 0,
                                     stream>>>(d_witness);
      break;
    case PrimaryKind::Uneven:
      primary_uneven_triggers<<<kPrimaryBlocks, kPrimaryThreads, 0, stream>>>(
          d_witness);
      break;
    case PrimaryKind::MixedExit:
      primary_mixed_trigger_and_exit<<<kPrimaryBlocks, kPrimaryThreads, 0,
                                      stream>>>(d_witness);
      break;
  }
  CHECK_CUDA(cudaGetLastError());

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(kSecondaryThreads);
  config.stream = stream;
  config.attrs = test.pdl_attribute ? &attribute : nullptr;
  config.numAttrs = test.pdl_attribute ? 1 : 0;

  CHECK_CUDA(cudaLaunchKernelEx(&config, secondary_snapshot_then_wait, d_witness,
                               skip_wait, test.primary == PrimaryKind::Uneven));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  Witness result{};
  CHECK_CUDA(cudaMemcpy(&result, d_witness, sizeof(result),
                       cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_witness));

  int stale_before_wait = 0;
  for (int cta = 0; cta < kPrimaryBlocks; ++cta) {
    if (result.stale_mask & (1u << cta)) {
      ++stale_before_wait;
    }
  }
  std::printf("[%s] pdl_attr=%d, skip_wait=%d, secondary_started=%d, "
              "stale_before_wait=%d, stale_mask=0x%x, "
              "premature_entries=%d, errors=%d\n",
              test.name, test.pdl_attribute, skip_wait,
              result.secondary_started, stale_before_wait, result.stale_mask,
              result.premature_entries, result.errors);

  bool passed = true;
  if (result.secondary_started != 1 || result.errors != 0) {
    std::fprintf(stderr, "[%s] FAILED: secondary must run and see every late "
                         "value and completion flag after the wait\n",
                 test.name);
    passed = false;
  }
  if (test.pdl_attribute &&
      (result.stale_mask == 0 ||
       (result.stale_mask & test.required_stale_mask) !=
           test.required_stale_mask)) {
    std::fprintf(stderr, "[%s] FAILED: required pre-wait overlap was not "
                         "observed (use a simulator config with spare SMs)\n",
                 test.name);
    passed = false;
  }
  if (!test.pdl_attribute && result.stale_mask != 0) {
    std::fprintf(stderr, "[%s] FAILED: secondary read unfinished primary data "
                         "without the PDL launch attribute\n", test.name);
    passed = false;
  }
  if (test.primary == PrimaryKind::Uneven &&
      (result.premature_entries != 0 || result.delayed_cta_ready != 1)) {
    std::fprintf(stderr, "[%s] FAILED: secondary entered before the delayed "
                         "CTA was ready to trigger\n", test.name);
    passed = false;
  }
  return passed;
}

void usage(const char *program) {
  std::fprintf(stderr, "Usage: %s [--case NAME] [--skip-wait]\n"
                       "Cases: explicit_wait, all_cta_trigger, exit_arrival, "
                       "no_attribute\n"
                       "--skip-wait is a negative control, expected to fail "
                       "the overlap cases.\n", program);
}

}  // namespace

int main(int argc, char **argv) {
  const char *selected_case = nullptr;
  bool skip_wait = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--skip-wait") == 0) {
      skip_wait = true;
    } else if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) {
      selected_case = argv[++i];
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  bool found = selected_case == nullptr;
  for (const TestCase &test : kCases) {
    if (selected_case && std::strcmp(selected_case, test.name) == 0) {
      found = true;
    }
  }
  if (!found) {
    usage(argv[0]);
    return 1;
  }

  bool passed = true;
  for (const TestCase &test : kCases) {
    if (!selected_case || std::strcmp(selected_case, test.name) == 0) {
      if (!run_case(test, skip_wait)) {
        passed = false;
      }
    }
  }
  if (!passed) {
    return 2;
  }

  std::puts("PASS");
  return 0;
}
