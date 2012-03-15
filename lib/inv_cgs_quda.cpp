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

void invertCgsCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x, 
			cudaColorSpinorField &b, QudaInvertParam *invert_param)
{
  typedef std::complex<double> Complex;

  ColorSpinorParam param;
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField y(x, param);
  cudaColorSpinorField r(x, param); 

  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField p(x, param);
  cudaColorSpinorField q(x, param);  
  cudaColorSpinorField v(x, param);
  cudaColorSpinorField t(x, param);
  cudaColorSpinorField tmp(x, param);  

  cudaColorSpinorField *x_sloppy, *r_sloppy, *r_0;
  if (invert_param->cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    x_sloppy = &x;
    r_sloppy = &r;
    r_0 = &b;
    zeroCuda(*x_sloppy);
    copyCuda(*r_sloppy, b);
  } else {
    x_sloppy = new cudaColorSpinorField(x, param);
    param.create = QUDA_COPY_FIELD_CREATE;
    r_sloppy = new cudaColorSpinorField(b, param);
    r_0 = new cudaColorSpinorField(b, param);
  }

  // Syntatic sugar
  cudaColorSpinorField &rSloppy = *r_sloppy;
  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &r0 = *r_0;

  double b2 = normCuda(b);

  double r2 = b2;
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver
  double delta = invert_param->reliable_delta;

  int k = 0;
  int rUpdate = 0;
  
  double alpha, beta;

  double rho0 = 1.0;  
  double rho  = reDotProductCuda(r0, rSloppy);
  
  zeroCuda(p);
  zeroCuda(q);
  
  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrr = rNorm;
  double maxrx = rNorm;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CGS: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();

  while (r2 > stop && k<invert_param->maxiter) {

     if(rho0 != 0) beta = rho / rho0;//CHECK 
     else beta = rho; 
     //
     xpayZpbypcwCuda(v, rSloppy, beta, q, beta, beta*beta, p);//(z, x, a, y, b, c, w) : z[i] =  x[i] + a*y[i]; w[i] = z[i] + b*y[i]+c*w[i]
     
     matSloppy(t, p, tmp);     
    
     if (fabs(rho) == 0.0) alpha = 0.0;
     else alpha = (rho / reDotProductCuda(r0, t));
     
     //t = v-alpha*t
     //v += t
     //x += alpha*v
     //and then q = t (temporal hack)
     
     xpaYpxZpbxCuda(v, -alpha, t, alpha, xSloppy);//y[i] = x[i] + a*y[i], x[i] = x[i] + y[i]; z[i]=z[i]+b*x[i] 
     copyCuda(q, t);
	  
//!
     matSloppy(t, v, tmp);     

          // r -= alpha*t
     axpyCuda(-alpha, t, rSloppy);

     rho0 = rho;
     rho  = reDotProductCuda(r0, rSloppy);

     r2 = normCuda(rSloppy);
//printf("\n\nAlpha = %le, Beta = %le, Rho = %le\n\n", alpha, beta, rho);
    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      xpyCuda(x, y); // swap these around?
      mat(r, y, x);
      r2 = xmyNormCuda(b, r);

      if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
      zeroCuda(xSloppy);
      ///
      rho0 = 1.0;//? 
      rho  = reDotProductCuda(r0, rSloppy);
      zeroCuda(p);//?
      zeroCuda(q);//?
      ///
      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;
    }

    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("CGS: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);
    
  if (k==invert_param->maxiter) warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CGS: Reliable updates = %d\n", rUpdate);
  
  invert_param->secs += stopwatchReadSeconds();
  
  float gflops = (blas_quda_flops + mat.flops() + matSloppy.flops())*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;
  
  //#if 0
  // Calculate the true residual
  mat(r, x);
  double true_res = xmyNormCuda(b, r);
    
  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printfQuda("CGS: Converged after %d iterations, r2 = %e, true_r2 = %e\n", k, sqrt(r2/b2), sqrt(true_res / b2));    
  //#endif

  if (invert_param->cuda_prec_sloppy != x.Precision()) {
    delete r_0;
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}

/*
void invertCgsCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x, 
			cudaColorSpinorField &b, QudaInvertParam *invert_param)
{
  typedef std::complex<double> Complex;

  ColorSpinorParam param;
  param.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField y(x, param);
  cudaColorSpinorField r(x, param); 

  param.precision = invert_param->cuda_prec_sloppy;
  cudaColorSpinorField p(x, param);
  cudaColorSpinorField v(x, param);
  cudaColorSpinorField q(x, param);//new
  cudaColorSpinorField tmp(x, param);
  cudaColorSpinorField tmp2(x, param);//TM!  
  cudaColorSpinorField t(x, param);

  cudaColorSpinorField *x_sloppy, *r_sloppy, *r_0;
  if (invert_param->cuda_prec_sloppy == x.Precision()) {
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    x_sloppy = &x;
    r_sloppy = &r;
    r_0 = &b;
    zeroCuda(*x_sloppy);
    copyCuda(*r_sloppy, b);
  } else {
    x_sloppy = new cudaColorSpinorField(x, param);
    param.create = QUDA_COPY_FIELD_CREATE;
    r_sloppy = new cudaColorSpinorField(b, param);
    r_0 = new cudaColorSpinorField(b, param);
  }

  // Syntatic sugar
  cudaColorSpinorField &rSloppy = *r_sloppy;
  cudaColorSpinorField &xSloppy = *x_sloppy;
  cudaColorSpinorField &r0 = *r_0;

  double b2 = normCuda(b);

  double r2 = b2;
  double stop = b2*invert_param->tol*invert_param->tol; // stopping condition of solver
  double delta = invert_param->reliable_delta;

  int k = 0;
  int rUpdate = 0;
  
  double rho = 1.0;
  double rho0 = 1.0;
  double alpha = 1.0;
  double beta;
  
  copyCuda(p, r0);
  copyCuda(v, r0);
  
  double rNorm = sqrt(r2);
  double r0Norm = rNorm;
  double maxrr = rNorm;
  double maxrx = rNorm;

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CGS: %d iterations, r2 = %e\n", k, r2);

  blas_quda_flops = 0;

  stopwatchStart();

  while (r2 > stop && k<invert_param->maxiter) {

     if (k == 0) rho = reDotProductCuda(r0, rSloppy);
     //matSloppy(q, p, tmp, tmp2);
     matSloppy(q, p, tmp);     
	  
    
     if (fabs(rho) == 0.0) alpha = 0.0;
     else alpha = (rho / reDotProductCuda(r0, q));
	  
    //input: v , q,  x_SLoppy;  output (update): q, v , x_SLoppy
	  //q = v - alpha * q
	  //v += q
	  //x_sloppy += alpha * v
     xpaYpxZpbxCuda(v, -alpha, q, alpha, xSloppy);
	  //!
     //matSloppy(t, v, tmp, tmp2);
     matSloppy(t, v, tmp);     

          // r -= alpha*t
     axpyCuda(-alpha, t, rSloppy);

     rho0 = reDotProductCuda(r0, rSloppy);

     beta = rho0 / rho;
     rho = rho0;
	  //
     xpayZpbypcwCuda(v, rSloppy, beta, q, beta, beta*beta, p);//z[i] =  x[i] + a*y[i]; u[i] = z[i] + b*y[i]+c*u[i]	  
     r2 = normCuda(rSloppy);
//printf("\n\nAlpha = %le, Beta = %le, Rho = %le\n\n", alpha, beta, rho);
    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;

    if (updateR) {
printf("\n\n\nDo update\n\n\n"); 
//exit(-1);     
      if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      xpyCuda(x, y); // swap these around?
      mat(r, y, x);
      r2 = xmyNormCuda(b, r);

      if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
      zeroCuda(xSloppy);
      ///
      rho = reDotProductCuda(r0, rSloppy);
//beta = rho / rho0;
//xpayZpaypcuCuda(v, rSloppy, beta, q, beta, beta*beta, p);
      ///
      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
      r0Norm = rNorm;      
      rUpdate++;
    }

    k++;
    if (invert_param->verbosity >= QUDA_VERBOSE) 
      printfQuda("CGS: %d iterations, r2 = %e\n", k, r2);
  }
  
  if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
  xpyCuda(y, x);
    
  if (k==invert_param->maxiter) warningQuda("Exceeded maximum iterations %d", invert_param->maxiter);

  if (invert_param->verbosity >= QUDA_VERBOSE) printfQuda("CGS: Reliable updates = %d\n", rUpdate);
  
  invert_param->secs += stopwatchReadSeconds();
  
  float gflops = (blas_quda_flops + mat.flops() + matSloppy.flops())*1e-9;
  //  printfQuda("%f gflops\n", gflops / stopwatchReadSeconds());
  invert_param->gflops += gflops;
  invert_param->iter += k;
  
  //#if 0
  // Calculate the true residual
  mat(r, x);
  double true_res = xmyNormCuda(b, r);
    
  if (invert_param->verbosity >= QUDA_SUMMARIZE)
    printfQuda("CGS: Converged after %d iterations, r2 = %e, true_r2 = %e\n", k, sqrt(r2/b2), sqrt(true_res / b2));    
  //#endif

  if (invert_param->cuda_prec_sloppy != x.Precision()) {
    delete r_0;
    delete r_sloppy;
    delete x_sloppy;
  }

  return;
}
*/

