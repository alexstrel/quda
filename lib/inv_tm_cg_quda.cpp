#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <iostream>

#define Nf 2

void invertTMCgCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &xFlv1, cudaColorSpinorField &xFlv2,
		    cudaColorSpinorField &bFlv1, cudaColorSpinorField &bFlv2, QudaInvertParam *invert_param)
{
  int k=0;
  int rUpdate = 0;
    
  cudaColorSpinorField *rFlv1 = new cudaColorSpinorField(bFlv1);
  cudaColorSpinorField *rFlv2 = new cudaColorSpinorField(bFlv2);
  
  ColorSpinorParam param;
  param.create = QUDA_ZERO_FIELD_CREATE;
  
  cudaColorSpinorField *yFlv1 = new cudaColorSpinorField(bFlv1, param); 
  cudaColorSpinorField *yFlv2 = new cudaColorSpinorField(bFlv2, param);   
  
  mat(*rFlv1, *rFlv2, xFlv1, xFlv2, *yFlv1, *yFlv2);

  zeroCuda(*yFlv1);
  zeroCuda(*yFlv2);

  double r2 = xmyNormCuda(bFlv1, *rFlv1) + xmyNormCuda(bFlv2, *rFlv2);
  rUpdate ++;
  
  param.precision = invert_param->cuda_prec_sloppy;

  cudaColorSpinorField *ApFlv1 = new cudaColorSpinorField(xFlv1, param); 
  cudaColorSpinorField *ApFlv2 = new cudaColorSpinorField(xFlv2, param);  

  cudaColorSpinorField *tmpFlv1 = new cudaColorSpinorField(xFlv1, param); 
  cudaColorSpinorField *tmpFlv2 = new cudaColorSpinorField(xFlv2, param);  

  cudaColorSpinorField *tmp2Flv1 = new cudaColorSpinorField(xFlv1, param); 
  cudaColorSpinorField *tmp2Flv2 = new cudaColorSpinorField(xFlv2, param);  

  
  cudaColorSpinorField *x_sloppyFlv1, *x_sloppyFlv2;
  cudaColorSpinorField *r_sloppyFlv1, *r_sloppyFlv2;
  
  if (invert_param->cuda_prec_sloppy == xFlv1.Precision()) {
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    x_sloppyFlv1 = &xFlv1;
    r_sloppyFlv1 = rFlv1;
    //
    x_sloppyFlv2 = &xFlv2;
    r_sloppyFlv2 = rFlv2;
  } else {
    param.create = QUDA_COPY_FIELD_CREATE;
    x_sloppyFlv1 = new cudaColorSpinorField(xFlv1, param);
    r_sloppyFlv1 = new cudaColorSpinorField(*rFlv1, param);
    //
    x_sloppyFlv2 = new cudaColorSpinorField(xFlv2, param);
    r_sloppyFlv2 = new cudaColorSpinorField(*rFlv2, param);
  }

  cudaColorSpinorField &xFlv1Sloppy = *x_sloppyFlv1;
  cudaColorSpinorField &rFlv1Sloppy = *r_sloppyFlv1;

  cudaColorSpinorField &xFlv2Sloppy = *x_sloppyFlv2;
  cudaColorSpinorField &rFlv2Sloppy = *r_sloppyFlv2;
  
  
  cudaColorSpinorField *pFlv1 = new cudaColorSpinorField(rFlv1Sloppy);
  cudaColorSpinorField *pFlv2 = new cudaColorSpinorField(rFlv2Sloppy);
  
  double r2_old;
  double src_normFlv1 = norm2(bFlv1);
  double src_normFlv2 = norm2(bFlv2);  
  double src_norm  = src_normFlv1 + src_normFlv2;
  double stop = src_norm*invert_param->tol*invert_param->tol; // stopping condition of solver

  double alpha, beta;
  double pAp;

  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = invert_param->reliable_delta;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CG: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();
  while (r2 > stop && k<invert_param->maxiter) {

    matSloppy(*ApFlv1, *ApFlv2, *pFlv1, *pFlv2, *tmpFlv1, *tmpFlv2); 
    
    pAp = reDotProductCuda(*pFlv1, *ApFlv1) + reDotProductCuda(*pFlv2, *ApFlv2);
    alpha = r2 / pAp;        
    r2_old = r2;
    r2 = axpyNormCuda(-alpha, *ApFlv1, rFlv1Sloppy) + axpyNormCuda(-alpha, *ApFlv2, rFlv2Sloppy);

    // reliable update conditions
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    
    if (!(updateR || updateX)) {
      beta = r2 / r2_old;
      axpyZpbxCuda(alpha, *pFlv1, xFlv1Sloppy, rFlv1Sloppy, beta);
      axpyZpbxCuda(alpha, *pFlv2, xFlv2Sloppy, rFlv2Sloppy, beta);      
    } else {
      axpyCuda(alpha, *pFlv1, xFlv1Sloppy), axpyCuda(alpha, *pFlv2, xFlv2Sloppy);      
      if (xFlv1.Precision() != xFlv1Sloppy.Precision()) copyCuda(xFlv1, xFlv1Sloppy), copyCuda(xFlv2, xFlv2Sloppy);
      
      xpyCuda(xFlv1, *yFlv1), xpyCuda(xFlv2, *yFlv2); // swap these around?
      mat(*rFlv1, *rFlv2, *yFlv1, *yFlv2, xFlv1, xFlv2); // here we can use x as tmp
      r2 = xmyNormCuda(bFlv1, *rFlv1) + xmyNormCuda(bFlv2, *rFlv2);
      if (xFlv1.Precision() != rFlv1Sloppy.Precision()) copyCuda(rFlv1Sloppy, *rFlv1), copyCuda(rFlv2Sloppy, *rFlv2);            
      zeroCuda(xFlv1Sloppy), zeroCuda(xFlv2Sloppy);

      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;

      beta = r2 / r2_old;
      xpayCuda(rFlv1Sloppy, beta, *pFlv1);
      xpayCuda(rFlv2Sloppy, beta, *pFlv2);      
    }

    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE)
      printfQuda("CG: %d iterations, r2 = %e\n", k, r2);
  }

  if (xFlv1.Precision() != xFlv1Sloppy.Precision()) copyCuda(xFlv1, xFlv1Sloppy), copyCuda(xFlv2, xFlv2Sloppy);
  xpyCuda(*yFlv1, xFlv1), xpyCuda(*yFlv2, xFlv2);

  invert_param->secs = stopwatchReadSeconds();

  
  if (k==invert_param->maxiter) 
    warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

  float gflops = (blas_quda_flops + mat.flops() + matSloppy.flops())*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops = gflops;
  invert_param->iter = k;

  blas_quda_flops = 0;

  //#if 0
  // Calculate the true residual
  mat(*rFlv1, *rFlv2, xFlv1, xFlv2, *yFlv1, *yFlv2);
  
  //double true_res = xmyNormCuda(bFlv1, *rFlv1) + xmyNormCuda(bFlv2, *rFlv2);
  
  double true_resFlv1 = xmyNormCuda(bFlv1, *rFlv1); 
  double true_resFlv2 = xmyNormCuda(bFlv2, *rFlv2);
  
  if (invert_param->verbosity >= QUDA_SUMMARIZE){
    printfQuda("Converged after %d iterations, r2 = %e, relative true_r2 flavor1 = %e, relative true_r2 flavor2 = %e\n", 
	       k, r2, true_resFlv1 / src_normFlv1, true_resFlv2 / src_normFlv2);
  }
  //#endif

  if (invert_param->cuda_prec_sloppy != xFlv1.Precision()) {
    delete r_sloppyFlv1;
    delete r_sloppyFlv2;    
    delete x_sloppyFlv1;
    delete x_sloppyFlv2;    
  }

  return;
}
