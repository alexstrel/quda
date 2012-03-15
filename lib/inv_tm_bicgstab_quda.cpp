#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include <color_spinor_field.h>

__attribute__((always_inline)) double3 operator+(const double3 a, const double3 b)
{
  double3 c;
  c.x = a.x + b.x; 
  c.y = a.y + b.y;
  c.z = a.z + b.z;
  return c;
}

void invertTMBiCGstabCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &xFlv1, cudaColorSpinorField &xFlv2,
			 cudaColorSpinorField &bFlv1, cudaColorSpinorField &bFlv2, QudaInvertParam *invert_param)
{
  typedef std::complex<double> Complex;

  ColorSpinorParam param;
  param.create = QUDA_ZERO_FIELD_CREATE;
  
  cudaColorSpinorField *yFlv1 = new cudaColorSpinorField(xFlv1, param); 
  cudaColorSpinorField *yFlv2 = new cudaColorSpinorField(xFlv2, param); 
  //
  cudaColorSpinorField *rFlv1 = new cudaColorSpinorField(xFlv1, param);   
  cudaColorSpinorField *rFlv2 = new cudaColorSpinorField(xFlv2, param);   

  param.precision = invert_param->cuda_prec_sloppy;    

  cudaColorSpinorField *pFlv1 = new cudaColorSpinorField(xFlv1, param);     
  cudaColorSpinorField *pFlv2 = new cudaColorSpinorField(xFlv2, param);     
  //
  cudaColorSpinorField *vFlv1 = new cudaColorSpinorField(xFlv1, param); 
  cudaColorSpinorField *vFlv2 = new cudaColorSpinorField(xFlv2, param); 
  //
  cudaColorSpinorField *tmpFlv1 = new cudaColorSpinorField(xFlv1, param);     
  cudaColorSpinorField *tmpFlv2 = new cudaColorSpinorField(xFlv2, param);     
  //
  cudaColorSpinorField *tFlv1 = new cudaColorSpinorField(xFlv1, param);     
  cudaColorSpinorField *tFlv2 = new cudaColorSpinorField(xFlv2, param);     


  

  cudaColorSpinorField *x_sloppyFlv1, *x_sloppyFlv2;
  cudaColorSpinorField *r_sloppyFlv1, *r_sloppyFlv2;  
  cudaColorSpinorField *r_0Flv1, *r_0Flv2;  
  
  if (invert_param->cuda_prec_sloppy == xFlv1.Precision()) {
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    x_sloppyFlv1 = &xFlv1,  x_sloppyFlv2 = &xFlv2;
    
    r_sloppyFlv1 = rFlv1, r_sloppyFlv2 = rFlv2;    
    r_0Flv1 = &bFlv1,   r_0Flv2 = &bFlv2;    
    zeroCuda(*x_sloppyFlv1), zeroCuda(*x_sloppyFlv2);
    copyCuda(*r_sloppyFlv1, bFlv1), copyCuda(*r_sloppyFlv2, bFlv2);
  } else {
    x_sloppyFlv1 = new cudaColorSpinorField(xFlv1, param);
    x_sloppyFlv2 = new cudaColorSpinorField(xFlv2, param);
    param.create = QUDA_COPY_FIELD_CREATE;
    r_sloppyFlv1 = new cudaColorSpinorField(bFlv1, param);
    r_0Flv1      = new cudaColorSpinorField(bFlv1, param);
    //
    r_sloppyFlv2 = new cudaColorSpinorField(bFlv2, param);
    r_0Flv2      = new cudaColorSpinorField(bFlv2, param);
    
  }

  // Syntatic sugar
  cudaColorSpinorField &rFlv1Sloppy = *r_sloppyFlv1;
  cudaColorSpinorField &rFlv2Sloppy = *r_sloppyFlv2;  
  //
  cudaColorSpinorField &xFlv1Sloppy = *x_sloppyFlv1;
  cudaColorSpinorField &xFlv2Sloppy = *x_sloppyFlv2;  
  //
  cudaColorSpinorField &rFlv1_0 = *r_0Flv1;
  cudaColorSpinorField &rFlv2_0 = *r_0Flv2;
  
  
  double src_normFlv1 = normCuda(bFlv1);
  double src_normFlv2 = normCuda(bFlv2);   
  double b2 = src_normFlv1 + src_normFlv2;

  double r2 = b2;
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver
  double delta = invert_param->reliable_delta;

  int k = 0;
  int rUpdate = 0;
  
  Complex rho(1.0, 0.0);
  Complex rho0 = rho;
  Complex alpha(1.0, 0.0);
  Complex omega(1.0, 0.0);
  Complex beta;

  double3 rho_r2;
  double3 omega_t2;
  
  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrr = rNorm;
  double maxrx = rNorm;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();

  while (r2 > stop && k<invert_param->maxiter) {
    
    if (k==0) {
      rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
      copyCuda(*pFlv1, rFlv1Sloppy), copyCuda(*pFlv2, rFlv2Sloppy);      
    } else {
      if (abs(rho*alpha) == 0.0) beta = 0.0;
      else beta = (rho/rho0) * (alpha/omega);

      cxpaypbzCuda(rFlv1Sloppy, -beta*omega, *vFlv1, beta, *pFlv1);
      cxpaypbzCuda(rFlv2Sloppy, -beta*omega, *vFlv2, beta, *pFlv2);
    }
    
    matSloppy(*vFlv1, *vFlv2, *pFlv1, *pFlv2, *tmpFlv1, *tmpFlv2);

    if (abs(rho) == 0.0) alpha = 0.0;
    else alpha = rho / (cDotProductCuda(rFlv1_0, *vFlv1) + cDotProductCuda(rFlv2_0, *vFlv2));

    // r -= alpha*v
    caxpyCuda(-alpha, *vFlv1, rFlv1Sloppy);
    caxpyCuda(-alpha, *vFlv2, rFlv2Sloppy);
    
    matSloppy(*tFlv1, *tFlv2, rFlv1Sloppy, rFlv2Sloppy, *tmpFlv1, *tmpFlv2);
    
    // omega = (t, r) / (t, t)
    omega_t2 = cDotProductNormACuda(*tFlv1, rFlv1Sloppy) + cDotProductNormACuda(*tFlv2, rFlv2Sloppy);
    omega = Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);

    //x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
    rho_r2 = caxpbypzYmbwcDotProductWYNormYCuda(alpha, *pFlv1, omega, rFlv1Sloppy, xFlv1Sloppy, *tFlv1, rFlv1_0) + caxpbypzYmbwcDotProductWYNormYCuda(alpha, *pFlv2, omega, rFlv2Sloppy, xFlv2Sloppy, *tFlv2, rFlv2_0);
    
    rho0 = rho;
    rho = Complex(rho_r2.x, rho_r2.y);
    r2 = rho_r2.z;

    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
      if (xFlv1.Precision() != xFlv1Sloppy.Precision()) copyCuda(xFlv1, xFlv1Sloppy), copyCuda(xFlv2, xFlv2Sloppy);
      
      xpyCuda(xFlv1, *yFlv1);
      xpyCuda(xFlv2, *yFlv2); 
      mat(*rFlv1, *rFlv2, *yFlv1, *yFlv2, xFlv1, xFlv2);
      r2 = xmyNormCuda(bFlv1, *rFlv1) + xmyNormCuda(bFlv2, *rFlv2);

      if (xFlv1.Precision() != rFlv2Sloppy.Precision()) copyCuda(rFlv1Sloppy, *rFlv1), copyCuda(rFlv2Sloppy, *rFlv2);            
      zeroCuda(xFlv1Sloppy);
      zeroCuda(xFlv2Sloppy);
      
      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;
    }
    
    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("BiCGstab: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (xFlv1.Precision() != xFlv1Sloppy.Precision()) copyCuda(xFlv1, xFlv1Sloppy), copyCuda(xFlv2, xFlv2Sloppy);
  xpyCuda(*yFlv1, xFlv1);
  xpyCuda(*yFlv2, xFlv2);
  
  if (k==invert_param->maxiter) warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
  invert_param->secs += stopwatchReadSeconds();
  
  float gflops = (blas_quda_flops + mat.flops() + matSloppy.flops())*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;
  
  //#if 0
  // Calculate the true residual
  mat(*rFlv1, *rFlv2, xFlv1, xFlv2, 0);//temporal hack
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
