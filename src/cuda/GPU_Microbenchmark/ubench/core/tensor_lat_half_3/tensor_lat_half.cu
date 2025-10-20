#include "tensor_lat_half.h"

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
  tensor_lat<cute::half_t, float>();

  std::cout << "\nFP16 operand, FP16 accumulate (using CUTLASS CuTE tcgen05.mma):\n";
  tensor_lat<cute::half_t, cute::half_t>();

  // std::cout << "\ne4m3 operand, FP32 accumulate (using CUTLASS CuTE tcgen05.mma):\n";
  // tensor_lat<cute::float_e4m3_t, float>();
#else
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined. Please compile with SM100+ support." << std::endl;
#endif

  return 0;
}
