#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <blas_reference.h>
#include <twisted_mass_dslash_reference.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define CONF_FILE_PATH "../data/conf.1000"

int main(int argc, char **argv)
{
  // set QUDA parameters

  int device = 0; // CUDA device number

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = QUDA_HALF_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param  = newQudaInvertParam();
 
  gauge_param.X[0] = 24; 
  gauge_param.X[1] = 24;
  gauge_param.X[2] = 24;
  gauge_param.X[3] = 48;

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = QUDA_TWISTED_MASS_DSLASH;

  //inv_param.inv_type = QUDA_CGS_INVERTER;
  //inv_param.inv_type = QUDA_BICGSTAB_INVERTER; 
  inv_param.inv_type = QUDA_CG_INVERTER;  

  double mass = -0.9;
  inv_param.kappa = 1.0 / (2.0*(1 + 3/gauge_param.anisotropy + mass));
  inv_param.mu = 0.25;

  inv_param.tol = 5e-16;
  inv_param.maxiter = 10000;
  inv_param.reliable_delta = 1e-2;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  //inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  inv_param.solve_type = QUDA_NORMEQ_PC_SOLVE;
  //inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;  
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  gauge_param.ga_pad = 16*16*16; // 24*24*24;
  inv_param.sp_pad = 16*16*16;   // 24*24*24;
  inv_param.cl_pad = 0;   // 24*24*24;

  inv_param.verbosity = QUDA_VERBOSE;

  // Everything between here and the call to initQuda() is application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  setDims(gauge_param.X);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  //construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  readILDGconfig(gauge, CONF_FILE_PATH, &gauge_param);   

  void *spinorInFlv1      = malloc(V*spinorSiteSize*sSize);
  void *spinorInFlv2      = malloc(V*spinorSiteSize*sSize);  
  
  void *spinorOutFlv1     = malloc(V*spinorSiteSize*sSize);
  void *spinorOutFlv2     = malloc(V*spinorSiteSize*sSize);  
  
  void *spinorCheckFlv1   = malloc(V*spinorSiteSize*sSize);
  void *spinorCheckFlv2   = malloc(V*spinorSiteSize*sSize);  

  // create a point source at 0
  //if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) *((float*)spinorIn) = 1.0;
  //else *((double*)spinorIn) = 1.0;
  
  construct_spinor_field(spinorInFlv1, 1, 0, 0, 0, inv_param.cpu_prec);
  construct_spinor_field(spinorInFlv2, 1, 0, 0, 0, inv_param.cpu_prec);
  
  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // perform the inversion for the first flavor:
  inv_param.twist_flavor = QUDA_TWIST_MINUS;  
  invertQuda(spinorOutFlv1, spinorInFlv1, &inv_param);
  
  printf("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  printf("\nDone for the first flavor: %i iter / %g secs = %g Gflops\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs);
  
  // perform the inversion for the first flavor:
  inv_param.twist_flavor = QUDA_TWIST_PLUS;  
  invertQuda(spinorOutFlv2, spinorInFlv2, &inv_param);
  
  printf("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  printf("\nDone for the second flavor: %i iter / %g secs = %g Gflops\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs);

	 
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printf("\nTotal time = %g secs\n",  time0);
  
  inv_param.twist_flavor = QUDA_TWIST_MINUS;
  if (inv_param.solution_type == QUDA_MAT_SOLUTION) { 
    mat(spinorCheckFlv1, gauge, spinorOutFlv1, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
	0, inv_param.cpu_prec, gauge_param.cpu_prec); 
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.5/inv_param.kappa, spinorCheckFlv1, V*spinorSiteSize, inv_param.cpu_prec);
  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {   
    matpc(spinorCheckFlv1, gauge, spinorOutFlv1, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
	  inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param.cpu_prec);
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheckFlv1, V*spinorSiteSize, inv_param.cpu_prec);
  }

  mxpy(spinorInFlv1, spinorCheckFlv1, V*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheckFlv1, V*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorInFlv1, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual for the first flavor: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  inv_param.twist_flavor = QUDA_TWIST_PLUS;
  if (inv_param.solution_type == QUDA_MAT_SOLUTION) { 
    mat(spinorCheckFlv2, gauge, spinorOutFlv2, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
	0, inv_param.cpu_prec, gauge_param.cpu_prec); 
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.5/inv_param.kappa, spinorCheckFlv2, V*spinorSiteSize, inv_param.cpu_prec);
  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {   
    matpc(spinorCheckFlv2, gauge, spinorOutFlv2, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
	  inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param.cpu_prec);
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheckFlv2, V*spinorSiteSize, inv_param.cpu_prec);
  }

  mxpy(spinorInFlv2, spinorCheckFlv2, V*spinorSiteSize, inv_param.cpu_prec);
  nrm2 = norm_2(spinorCheckFlv2, V*spinorSiteSize, inv_param.cpu_prec);
  src2 = norm_2(spinorInFlv2, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual for the first flavor: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));
  
  // finalize the QUDA library
  endQuda();

  return 0;
}
