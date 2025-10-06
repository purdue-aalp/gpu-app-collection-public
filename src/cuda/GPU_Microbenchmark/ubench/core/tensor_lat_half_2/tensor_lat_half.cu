#include "tensor_lat_half.h"

int main(int argc, char *argv[])
{
  intilizeDeviceProp(0, argc, argv);

  std::cout << "deviceProp: " << deviceProp.major << " " << deviceProp.minor << std::endl;

  // WGMMA requires Hopper (SM90) or later
  if (deviceProp.major < 9) {
    std::cout << "This benchmark requires NVIDIA Hopper (SM90+) or later for WGMMA support" << std::endl;
    std::cout << "Detected compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  std::cout << "FP16 operand, FP32 accumulate (using CUTLASS CuTE WGMMA):\n";
  tensor_lat<cute::half_t, float>();

  std::cout << "\nFP16 operand, FP16 accumulate (using CUTLASS CuTE WGMMA):\n";
  tensor_lat<cute::half_t, cute::half_t>();
#else
  std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED is not defined. Please compile with SM90+ support." << std::endl;
#endif

  return 0;
}
