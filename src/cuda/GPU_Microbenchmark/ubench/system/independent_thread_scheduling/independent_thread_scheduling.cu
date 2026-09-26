#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 2 * kWarpSize;
constexpr int kUnset = -1;
// Shared-memory atomics used to keep one path busy long enough (thousands of
// cycles) for a sibling path to run ahead if nothing holds it back.
constexpr int kDelayAtomics = 400;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Atomic accesses make synchronization explicit to real hardware and the
// simulator, without relying on volatile shared-memory visibility.
__device__ __forceinline__ int poll(int *address) {
  return atomicAdd(address, 0);
}

// Lanes 1..31 can reach the wait before lane 0 reaches the publishing path.
// A non-preemptive PDOM stack can pin those waiters above lane 0 forever.
__global__ void forward_publish_kernel(int *output) {
  __shared__ int ready;
  __shared__ int payload;
  const int lane = threadIdx.x;

  if (lane == 0) {
    ready = 0;
    payload = 0;
  }
  __syncthreads();

  if (lane != 0) {
    while (poll(&ready) == 0) {
    }
    output[lane] = payload + lane;
  } else {
    payload = 1000;
    atomicExch(&ready, 1);
    output[0] = 1000;
  }
}

// Mirror the predicate and producer lane.  This catches implementations that
// accidentally make progress only for one branch direction or lane ordering.
__global__ void reverse_publish_kernel(int *output) {
  __shared__ int ready;
  __shared__ int payload;
  const int lane = threadIdx.x;

  if (lane == 0) {
    ready = 0;
    payload = 0;
  }
  __syncthreads();

  if (lane != kWarpSize - 1) {
    while (poll(&ready) == 0) {
    }
    output[lane] = payload - lane;
  } else {
    payload = 2000;
    atomicExch(&ready, 1);
    output[lane] = 2000 - lane;
  }
}

// The active owner changes 32 times. Each lane must leave the polling path,
// publish its trace slot, and wake the next lane. This exercises many dynamic
// split masks rather than a single producer/consumer split.
__global__ void token_ring_kernel(int *output) {
  __shared__ int token;
  const int lane = threadIdx.x;

  if (lane == 0)
    atomicExch(&token, 0);
  __syncthreads();

  while (poll(&token) != lane) {
  }
  output[lane] = 3000 + lane;
  atomicExch(&token, lane + 1);
}

// Reuse the same dynamic split/reconvergence point several times. This catches
// stale pending masks and split entries that survive one round of scheduling.
__global__ void multi_round_ring_kernel(int *output) {
  constexpr int kRounds = 4;
  __shared__ int token;
  const int lane = threadIdx.x;

  output[lane] = 0;
  if (lane == 0)
    atomicExch(&token, 0);
  __syncthreads();

  for (int round = 0; round < kRounds; ++round) {
    const int turn = round * kWarpSize + lane;
    while (poll(&token) != turn) {
    }
    output[lane] += (round + 1) * (lane + 1);
    atomicExch(&token, turn + 1);
  }
}

// Progress must work in both lane directions in one kernel. The second phase
// depends on lane 31 becoming runnable after lane 0 releases the first phase.
__global__ void bidirectional_handshake_kernel(int *output) {
  __shared__ int stage;
  const int lane = threadIdx.x;

  if (lane == 0)
    atomicExch(&stage, 0);
  __syncthreads();

  if (lane == 0) {
    atomicExch(&stage, 1);
  } else {
    while (poll(&stage) < 1) {
    }
  }

  if (lane == kWarpSize - 1) {
    atomicExch(&stage, 2);
  } else {
    while (poll(&stage) < 2) {
    }
  }
  output[lane] = 5000 + lane;
}

// barrier.sync without .aligned: lanes of one warp may reach the barrier at
// different instructions (sm_70+). __syncthreads() emits the .aligned form,
// which requires every lane of a warp to execute the same barrier instruction.
// The asm text differs per call site so the compiler cannot merge two call
// sites into one barrier.
#define BARRIER_SYNC_UNALIGNED(site)                                           \
  asm volatile("barrier.sync 0; // " site ::: "memory")

__device__ void delay(int *counter, int iterations) {
  for (int i = 0; i < iterations; ++i)
    atomicAdd(counter, 1);
}

// Warp 1 publishes only after a long delay. Warp 0's lanes 0-15 reach the
// barrier first while lanes 16-31 are still diverging (a loop whose trip count
// differs per lane) and reconverging. No lane of warp 0 may leave the barrier
// before warp 1 has published, whatever order its splits are scheduled in.
// Each arm reads the published value right after its own barrier, before the
// arms rejoin; the two arms store with different instructions so the compiler
// cannot sink the read into the join.
__global__ void barrier_divergence_kernel(int *output) {
  __shared__ int published[kWarpSize];
  __shared__ int counter;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;

  if (tid < kWarpSize)
    published[tid] = 0;
  if (tid == 0)
    counter = 0;
  __syncthreads();

  if (tid >= kWarpSize) {
    delay(&counter, kDelayAtomics);
    published[lane] = 6000 + lane;
    BARRIER_SYNC_UNALIGNED("publisher");
    output[tid] = 6000 + lane;
  } else if (lane < kWarpSize / 2) {
    BARRIER_SYNC_UNALIGNED("early half");
    output[tid] = published[lane];
  } else {
    delay(&counter, (lane & 3) + 1);
    BARRIER_SYNC_UNALIGNED("diverged half");
    atomicExch(&output[tid], published[lane]);
  }
}

// Lanes 0-15 of warp 0 execute the barrier as the last instruction before the
// branch rejoins; lanes 16-31 reach their barrier much later. The early lanes
// must stay at the barrier until warp 1 has published, even if the simulator
// would otherwise let lanes waiting at the join run on without their siblings.
__global__ void barrier_before_join_kernel(int *output) {
  __shared__ int published[kWarpSize];
  __shared__ int counter;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;

  if (tid < kWarpSize)
    published[tid] = 0;
  if (tid == 0)
    counter = 0;
  __syncthreads();

  if (tid >= kWarpSize) {
    delay(&counter, kDelayAtomics);
    published[lane] = 7000 + lane;
    BARRIER_SYNC_UNALIGNED("publisher");
  } else {
    // Written late-arm first: nvcc lays the else arm out so that its barrier
    // falls straight through into the join, which is the case under test.
    if (lane >= kWarpSize / 2) {
      delay(&counter, kDelayAtomics);
      BARRIER_SYNC_UNALIGNED("late half");
    } else {
      BARRIER_SYNC_UNALIGNED("early half");
    }
  }
  output[tid] = published[lane];
}

// The callees below are __noinline__ so they run as real PTX call/ret, where
// divergence inside the callee has to reconverge and return correctly.

// Lanes split inside the callee: lane 0 publishes, the rest wait for it, and
// both paths rejoin before returning.
__device__ __noinline__ int handshake_callee(int *flag, int lane) {
  int value;
  if (lane == 0) {
    value = 8000;
    atomicExch(flag, 1);
  } else {
    while (poll(flag) == 0) {
    }
    value = 8000 + lane;
  }
  return value;
}

__global__ void call_divergence_kernel(int *output) {
  __shared__ int flag;
  const int lane = threadIdx.x;

  if (lane == 0)
    atomicExch(&flag, 0);
  __syncthreads();

  output[lane] = handshake_callee(&flag, lane);
}

// Odd lanes return from the callee immediately; the even lanes stay behind for
// a handshake and return later.
__device__ __noinline__ int early_return_callee(int *flag, int lane) {
  if (lane & 1)
    return 9000 + lane;
  if (lane == 0) {
    atomicExch(flag, 1);
  } else {
    while (poll(flag) == 0) {
    }
  }
  return 9000 + lane;
}

__global__ void call_early_return_kernel(int *output) {
  __shared__ int flag;
  const int lane = threadIdx.x;

  if (lane == 0)
    atomicExch(&flag, 0);
  __syncthreads();

  output[lane] = early_return_callee(&flag, lane);
}

// Only half the warp calls the inner function, from inside a divergent branch
// of the outer one, so the inner call's frame must return to a split that is
// itself inside a call.
__device__ __noinline__ int nested_inner(int *flag, int lane) {
  if (lane == 0) {
    atomicExch(flag, 1);
  } else {
    while (poll(flag) == 0) {
    }
  }
  return 10000 + lane;
}

__device__ __noinline__ int nested_outer(int *flag, int lane) {
  int value;
  if (lane < kWarpSize / 2)
    value = nested_inner(flag, lane);
  else
    value = 11000 + lane;
  return value + 1;
}

__global__ void call_nested_kernel(int *output) {
  __shared__ int flag;
  const int lane = threadIdx.x;

  if (lane == 0)
    atomicExch(&flag, 0);
  __syncthreads();

  output[lane] = nested_outer(&flag, lane);
}

enum TestId {
  kRing,
  kMultiRoundRing,
  kHandshake,
  kForward,
  kReverse,
  kBarrierDivergence,
  kBarrierBeforeJoin,
  kCallDivergence,
  kCallEarlyReturn,
  kCallNested,
  kTestCount
};

const char *const kTestNames[kTestCount] = {
    "token-ring", "multi-round-ring", "bidirectional-handshake",
    "forward-publish", "reverse-publish", "barrier-divergence",
    "barrier-before-join", "call-divergence", "call-early-return",
    "call-nested"};

int thread_count(TestId test) {
  switch (test) {
  case kBarrierDivergence:
  case kBarrierBeforeJoin:
    return kMaxThreads;
  default:
    return kWarpSize;
  }
}

int expected_value(TestId test, int tid) {
  const int lane = tid % kWarpSize;
  switch (test) {
  case kForward:
    return 1000 + lane;
  case kReverse:
    return 2000 - lane;
  case kRing:
    return 3000 + lane;
  case kMultiRoundRing:
    return 10 * (lane + 1);
  case kHandshake:
    return 5000 + lane;
  case kBarrierDivergence:
    return 6000 + lane;
  case kBarrierBeforeJoin:
    return 7000 + lane;
  case kCallDivergence:
    return 8000 + lane;
  case kCallEarlyReturn:
    return 9000 + lane;
  case kCallNested:
    return lane < kWarpSize / 2 ? 10001 + lane : 11001 + lane;
  default:
    return kUnset;
  }
}

void launch(TestId test, int *device_output) {
  switch (test) {
  case kForward:
    forward_publish_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kReverse:
    reverse_publish_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kRing:
    token_ring_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kMultiRoundRing:
    multi_round_ring_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kHandshake:
    bidirectional_handshake_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kBarrierDivergence:
    barrier_divergence_kernel<<<1, kMaxThreads>>>(device_output);
    break;
  case kBarrierBeforeJoin:
    barrier_before_join_kernel<<<1, kMaxThreads>>>(device_output);
    break;
  case kCallDivergence:
    call_divergence_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kCallEarlyReturn:
    call_early_return_kernel<<<1, kWarpSize>>>(device_output);
    break;
  case kCallNested:
    call_nested_kernel<<<1, kWarpSize>>>(device_output);
    break;
  default:
    std::abort();
  }
}

bool run_test(TestId test) {
  int *device_output = nullptr;
  int host_output[kMaxThreads];
  const int threads = thread_count(test);
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(host_output)));
  CUDA_CHECK(cudaMemset(device_output, 0xff, sizeof(host_output)));

  launch(test, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(host_output, device_output, sizeof(host_output),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));

  int failures = 0;
  for (int tid = 0; tid < threads; ++tid) {
    const int expected = expected_value(test, tid);
    if (host_output[tid] != expected) {
      std::printf("  thread %2d: got %d, expected %d\n", tid, host_output[tid],
                  expected);
      ++failures;
    }
  }
  std::printf("[%s] %s\n", failures == 0 ? "PASS" : "FAIL", kTestNames[test]);
  return failures == 0;
}

int find_test(const char *name) {
  for (int i = 0; i < kTestCount; ++i) {
    if (std::strcmp(name, kTestNames[i]) == 0)
      return i;
  }
  return -1;
}

} // namespace

int main(int argc, char **argv) {
  int first = 0;
  int last = kTestCount;
  if (argc == 2) {
    first = find_test(argv[1]);
    if (first < 0) {
      std::fprintf(stderr, "Unknown test '%s'. Available tests:\n", argv[1]);
      for (const char *name : kTestNames)
        std::fprintf(stderr, "  %s\n", name);
      return EXIT_FAILURE;
    }
    last = first + 1;
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [test-name]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int failures = 0;
  for (int test = first; test < last; ++test) {
    if (!run_test(static_cast<TestId>(test)))
      ++failures;
  }

  std::printf("RESULT: %s (%d/%d tests passed)\n",
              failures == 0 ? "PASSED" : "FAILED", last - first - failures,
              last - first);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
