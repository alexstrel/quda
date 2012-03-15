#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

int main(int argc, char **argv)
{
  // set QUDA parameters

  int device = 0; // CUDA device number

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = QUDA_SINGLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = QUDA_HALF_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
 
  gauge_param.X[0] = 16;
  gauge_param.X[1] = 16;
  gauge_param.X[2] = 16;
  gauge_param.X[3] = 24;

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

  int clover_yes = 0; // 0 for plain Wilson, 1 for clover
  
  if (clover_yes) {
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  } else {
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
  }
  inv_param.inv_type = QUDA_BICGSTAB_INVERTER;

  double mass = -0.9;
  inv_param.kappa = 1.0 / (2.0*(1 + 3/gauge_param.anisotropy + mass));
  inv_param.tol = 5e-8;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 0.1;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  gauge_param.ga_pad = 0; // 24*24*24;
  inv_param.sp_pad = 0;   // 24*24*24;
  inv_param.cl_pad = 0;   // 24*24*24;

  if (clover_yes) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }
  inv_param.verbosity = QUDA_VERBOSE;

  // Everything between here and the call to initQuda() is application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  setDims(gauge_param.X);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv, *clover;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);

  if (clover_yes) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);

    // The uninverted clover term is only needed when solving the unpreconditioned
    // system or when using "asymmetric" even/odd preconditioning.
    int preconditioned = (inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ||
			  inv_param.solve_type == QUDA_NORMEQ_PC_SOLVE);
    int asymmetric = preconditioned &&
                         (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
                          inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
    if (!preconditioned) {
      clover = clover_inv;
      clover_inv = NULL;
    } else if (asymmetric) { // fake it by using the same random matrix
      clover = clover_inv;   // for both clover and clover_inv
    } else {
      clover = NULL;
    }
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize);
  void *spinorOut = malloc(V*spinorSiteSize*sSize);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize);

  // create a point source at 0
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) *((float*)spinorIn) = 1.0;
  else *((double*)spinorIn) = 1.0;

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // load the clover term, if desired
  if (clover_yes) loadCloverQuda(clover, clover_inv, &inv_param);
  
  // perform the inversion
  invertQuda(spinorOut, spinorIn, &inv_param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printf("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  if (clover_yes) printf("   Clover: %f GiB\n", inv_param.cloverGiB);
  printf("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  if (inv_param.solution_type == QUDA_MAT_SOLUTION) { 
    mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, 
	inv_param.cpu_prec, gauge_param.cpu_prec); 
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {   
    matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, 
	  inv_param.cpu_prec, gauge_param.cpu_prec);
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
      ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  }


  mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  // finalize the QUDA library
  endQuda();

  return 0;
}
