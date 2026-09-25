#include "shfl_scan.h"

int main(int argc, char *argv[])
{
  initializeDeviceProp(0, argc, argv);

  bool passed = run_shuffle_test();
  passed = run_scan_test() && passed;
  printf("Overall: %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
