#include "tensor_bw_half_sm100.h"

int main(int argc, char *argv[])
{
  intilizeDeviceProp(0, argc, argv);

  std::cout << "deviceProp: " << deviceProp.major << " " << deviceProp.minor << std::endl;

  // tcgen05.mma requires Blackwell (SM100) or later
  if ((deviceProp.major != 10) || (deviceProp.major == 10 && deviceProp.minor > 1)) {
    std::cout << "This benchmark requires NVIDIA Blackwell (SM100a) for tcgen05.mma support" << std::endl;
    std::cout << "Detected compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << "FP16 operand, FP32 accumulate (using CUTLASS CuTE tcgen05.mma):\n";
  using T = cute::half_t;
  using R = float;
  // Row = 64, Col = 8:8:256
  tensor_bw<T, R, 64, 8>();
  tensor_bw<T, R, 64, 16>();
  tensor_bw<T, R, 64, 24>();
  tensor_bw<T, R, 64, 32>();
  tensor_bw<T, R, 64, 40>();
  tensor_bw<T, R, 64, 48>();
  tensor_bw<T, R, 64, 56>();
  tensor_bw<T, R, 64, 64>();
  tensor_bw<T, R, 64, 72>();
  tensor_bw<T, R, 64, 80>();
  tensor_bw<T, R, 64, 88>();
  tensor_bw<T, R, 64, 96>();
  tensor_bw<T, R, 64, 104>();
  tensor_bw<T, R, 64, 112>();
  tensor_bw<T, R, 64, 120>();
  tensor_bw<T, R, 64, 128>();
  tensor_bw<T, R, 64, 136>();
  tensor_bw<T, R, 64, 144>();
  tensor_bw<T, R, 64, 152>();
  tensor_bw<T, R, 64, 160>();
  tensor_bw<T, R, 64, 168>();
  tensor_bw<T, R, 64, 176>();
  tensor_bw<T, R, 64, 184>();
  tensor_bw<T, R, 64, 192>();
  tensor_bw<T, R, 64, 200>();
  tensor_bw<T, R, 64, 208>();
  tensor_bw<T, R, 64, 216>();
  tensor_bw<T, R, 64, 224>();
  tensor_bw<T, R, 64, 232>();
  tensor_bw<T, R, 64, 240>();
  tensor_bw<T, R, 64, 248>();
  tensor_bw<T, R, 64, 256>();

  // Row = 128, Col = 8:8:256
  tensor_bw<T, R, 128, 16>();
  tensor_bw<T, R, 128, 32>();
  tensor_bw<T, R, 128, 48>();
  tensor_bw<T, R, 128, 64>();
  tensor_bw<T, R, 128, 80>();
  tensor_bw<T, R, 128, 96>();
  tensor_bw<T, R, 128, 112>();
  tensor_bw<T, R, 128, 128>();
  tensor_bw<T, R, 128, 144>();
  tensor_bw<T, R, 128, 160>();
  tensor_bw<T, R, 128, 176>();
  tensor_bw<T, R, 128, 192>();
  tensor_bw<T, R, 128, 208>();
  tensor_bw<T, R, 128, 224>();
  tensor_bw<T, R, 128, 240>();
  tensor_bw<T, R, 128, 256>();
  // tensor_bw<cute::half_t, float, 128, 192>();

  std::cout << "\nFP16 operand, FP16 accumulate (using CUTLASS CuTE tcgen05.mma):\n";
  tensor_bw<cute::half_t, cute::half_t>();

  // std::cout << "\ne4m3 operand, FP32 accumulate (using CUTLASS CuTE tcgen05.mma):\n";
  // tensor_bw<cute::float_e4m3_t, float>();
#else
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined. Please compile with SM100+ support." << std::endl;
#endif

  return 0;
}
