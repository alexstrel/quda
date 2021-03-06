//
// Auto-tuned blas CUDA parameters, generated by blas_test
//

static int blas_threads[22][3] = {
  {  64,  256,   64},  // Kernel  0: copyCuda
  {  64,   64,   64},  // Kernel  1: axpbyCuda
  {  64,   64,   64},  // Kernel  2: xpyCuda
  {  64,   64,   64},  // Kernel  3: axpyCuda
  {  64,   64,   64},  // Kernel  4: xpayCuda
  {  64,   64,   64},  // Kernel  5: mxpyCuda
  {  64,   64,   64},  // Kernel  6: axCuda
  {  64,   64,   64},  // Kernel  7: caxpyCuda
  {  64,   64,   64},  // Kernel  8: caxpbyCuda
  {  64,   64,   64},  // Kernel  9: cxpaypbzCuda
  {  64,  128,  128},  // Kernel 10: axpyZpbxCuda
  {  64,  128,  128},  // Kernel 11: caxpbypzYmbwCuda
  {  64,  128,   64},  // Kernel 12: sumCuda
  {  64,  128,   64},  // Kernel 13: normCuda
  {  64,   64,   64},  // Kernel 14: reDotProductCuda
  {  64,  128,   64},  // Kernel 15: axpyNormCuda
  {  64,  128,   64},  // Kernel 16: xmyNormCuda
  {  64,  128,   64},  // Kernel 17: cDotProductCuda
  {  64,  128,   64},  // Kernel 18: xpaycDotzyCuda
  {  64,   64,   64},  // Kernel 19: cDotProductNormACuda
  {  64,   64,   64},  // Kernel 20: cDotProductNormBCuda
  {  64,  128,   64}   // Kernel 21: caxpbypzYmbwcDotProductWYNormYQuda
};

static int blas_blocks[22][3] = {
  {16384,  256, 4096},  // Kernel  0: copyCuda
  {2048,  128,  128},  // Kernel  1: axpbyCuda
  {2048,  128,  128},  // Kernel  2: xpyCuda
  {  64,  128,  128},  // Kernel  3: axpyCuda
  {  64,  128,  128},  // Kernel  4: xpayCuda
  {2048,  128,  128},  // Kernel  5: mxpyCuda
  {2048,  128,  128},  // Kernel  6: axCuda
  {2048,  128, 16384},  // Kernel  7: caxpyCuda
  {2048,  128,  128},  // Kernel  8: caxpbyCuda
  {2048,  128, 8192},  // Kernel  9: cxpaypbzCuda
  { 256,  256,  128},  // Kernel 10: axpyZpbxCuda
  {2048,  128, 16384},  // Kernel 11: caxpbypzYmbwCuda
  { 128,  128,  128},  // Kernel 12: sumCuda
  { 128,  128,  128},  // Kernel 13: normCuda
  {  64,  128,  128},  // Kernel 14: reDotProductCuda
  {  64,  128,  128},  // Kernel 15: axpyNormCuda
  {  64,  128,  128},  // Kernel 16: xmyNormCuda
  {  64,  128,  128},  // Kernel 17: cDotProductCuda
  {  64,  128,  128},  // Kernel 18: xpaycDotzyCuda
  {  64, 1024,  128},  // Kernel 19: cDotProductNormACuda
  {  64, 1024,  128},  // Kernel 20: cDotProductNormBCuda
  { 256,  256,  128}   // Kernel 21: caxpbypzYmbwcDotProductWYNormYQuda
};
