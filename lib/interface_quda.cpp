#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <quda_internal.h>
#include <blas_quda.h>
#include <gauge_quda.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <clover_quda.h>
#include <invert_quda.h>

#include <color_spinor_field.h>

#define spinorSiteSize 24 // real numbers per spinor

FullGauge cudaGaugePrecise;      // Wilson links
FullGauge cudaGaugeSloppy;

FullGauge cudaFatLinkPrecise;    // asqtad fat links
FullGauge cudaFatLinkSloppy;

FullGauge cudaLongLinkPrecise;   // asqtad long links
FullGauge cudaLongLinkSloppy;

FullClover cudaCloverPrecise;    // clover term
FullClover cudaCloverSloppy;

FullClover cudaCloverInvPrecise; // inverted clover term
FullClover cudaCloverInvSloppy;

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM


void initQuda(int dev)
{
  static int initialized = 0;
  if (initialized) {
    return;
  }
  initialized = 1;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    errorQuda("No devices supporting CUDA");
  }

  for(int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    fprintf(stderr, "QUDA: Found device %d: %s\n", i, deviceProp.name);
  }

  if (dev < 0) {
    dev = deviceCount - 1;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.major < 1) {
    errorQuda("Device %d does not support CUDA", dev);
  }

  fprintf(stderr, "QUDA: Using device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  cudaGaugePrecise.even = NULL;
  cudaGaugePrecise.odd = NULL;
  cudaGaugeSloppy.even = NULL;
  cudaGaugeSloppy.odd = NULL;

  cudaFatLinkPrecise.even = NULL;
  cudaFatLinkPrecise.odd = NULL;
  cudaFatLinkSloppy.even = NULL;
  cudaFatLinkSloppy.odd = NULL;

  cudaLongLinkPrecise.even = NULL;
  cudaLongLinkPrecise.odd = NULL;
  cudaLongLinkSloppy.even = NULL;
  cudaLongLinkSloppy.odd = NULL;

  cudaCloverPrecise.even.clover = NULL;
  cudaCloverPrecise.odd.clover = NULL;
  cudaCloverSloppy.even.clover = NULL;
  cudaCloverSloppy.odd.clover = NULL;

  cudaCloverInvPrecise.even.clover = NULL;
  cudaCloverInvPrecise.odd.clover = NULL;
  cudaCloverInvSloppy.even.clover = NULL;
  cudaCloverInvSloppy.odd.clover = NULL;

  initBlas();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);  
}


void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  int packed_size;
  double anisotropy;
  FullGauge *precise, *sloppy;

  checkGaugeParam(param);

  switch (param->reconstruct) {
  case QUDA_RECONSTRUCT_8:
    packed_size = 8;
    break;
  case QUDA_RECONSTRUCT_12:
    packed_size = 12;
    break;
  case QUDA_RECONSTRUCT_NO:
    packed_size = 18;
    break;
  default:
    errorQuda("Invalid reconstruct type");
  }
  param->packed_size = packed_size;

  switch (param->type) {
  case QUDA_WILSON_LINKS:
    precise = &cudaGaugePrecise;
    sloppy = &cudaGaugeSloppy;
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    precise = &cudaFatLinkPrecise;
    sloppy = &cudaFatLinkSloppy;
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    precise = &cudaLongLinkPrecise;
    sloppy = &cudaLongLinkSloppy;
    break;
  default:
    errorQuda("Invalid gauge type");   
  }

  if (param->type == QUDA_WILSON_LINKS) {
    anisotropy = param->anisotropy;
  } else {
    anisotropy = 1.0;
  }

  if (param->type != QUDA_WILSON_LINKS &&
      param->gauge_fix == QUDA_GAUGE_FIXED_YES) {
    errorQuda("Temporal gauge fixing not supported for staggered");
  }

  if ((param->cuda_prec == QUDA_HALF_PRECISION && param->reconstruct == QUDA_RECONSTRUCT_NO) ||
      (param->cuda_prec_sloppy == QUDA_HALF_PRECISION && param->reconstruct_sloppy == QUDA_RECONSTRUCT_NO)) {
    warningQuda("Loading gauge field in half precision may give wrong results "
		"unless all elements have magnitude bounded by 1");
  }

  createGaugeField(precise, h_gauge, param->cuda_prec, param->cpu_prec, param->gauge_order, param->reconstruct, param->gauge_fix,
		   param->t_boundary, param->X, anisotropy, param->tadpole_coeff, param->ga_pad);
  param->gaugeGiB += 2.0 * precise->bytes / (1 << 30);

  if (param->cuda_prec_sloppy != param->cuda_prec ||
      param->reconstruct_sloppy != param->reconstruct) {
    createGaugeField(sloppy, h_gauge, param->cuda_prec_sloppy, param->cpu_prec, param->gauge_order,
		     param->reconstruct_sloppy, param->gauge_fix, param->t_boundary,
		     param->X, anisotropy, param->tadpole_coeff, param->ga_pad);
    param->gaugeGiB += 2.0 * sloppy->bytes / (1 << 30);
  } else {
    *sloppy = *precise;
  }
}


void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  FullGauge *gauge;

  switch (param->type) {
  case QUDA_WILSON_LINKS:
    gauge = &cudaGaugePrecise;
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    gauge = &cudaFatLinkPrecise;
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    gauge = &cudaLongLinkPrecise;
    break;
  default:
    errorQuda("Invalid gauge type");   
  }

  restoreGaugeField(h_gauge, gauge, param->cpu_prec, param->gauge_order);
}


void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{

  if (!h_clover && !h_clovinv) {
    errorQuda("loadCloverQuda() called with neither clover term nor inverse");
  }
  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }
  if (cudaGaugePrecise.even == NULL) {
    errorQuda("Gauge field must be loaded before clover");
  }
  if (inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
    errorQuda("Wrong dslash_type in loadCloverQuda()");
  }

  // determines whether operator is preconditioned when calling invertQuda()
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE ||
		   inv_param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  // determines whether operator is preconditioned when calling MatQuda() or MatDagMatQuda()
  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
		      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool asymmetric = (inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
		     inv_param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);

  // We issue a warning only when it seems likely that the user is screwing up:

  // inverted clover term is required when applying preconditioned operator
  if (!h_clovinv && pc_solve && pc_solution) {
    warningQuda("Inverted clover term not loaded");
  }

  // uninverted clover term is required when applying unpreconditioned operator,
  // but note that dslashQuda() is always preconditioned
  if (!h_clover && !pc_solve && !pc_solution) {
    //warningQuda("Uninverted clover term not loaded");
  }

  // uninverted clover term is also required for "asymmetric" preconditioning
  if (!h_clover && pc_solve && pc_solution && asymmetric) {
    warningQuda("Uninverted clover term not loaded");
  }

  int X[4];
  for (int i=0; i<4; i++) {
    X[i] = cudaGaugePrecise.X[i];
  }

  inv_param->cloverGiB = 0;

  if (h_clover) {
    allocateCloverField(&cudaCloverPrecise, X, inv_param->cl_pad, inv_param->clover_cuda_prec);
    loadCloverField(cudaCloverPrecise, h_clover, inv_param->clover_cpu_prec, inv_param->clover_order);
    inv_param->cloverGiB += 2.0*cudaCloverPrecise.even.bytes / (1<<30);

    if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
      allocateCloverField(&cudaCloverSloppy, X, inv_param->cl_pad, inv_param->clover_cuda_prec_sloppy);
      loadCloverField(cudaCloverSloppy, h_clover, inv_param->clover_cpu_prec, inv_param->clover_order);
      inv_param->cloverGiB += 2.0*cudaCloverInvSloppy.even.bytes / (1<<30);
    } else {
      cudaCloverSloppy = cudaCloverPrecise;
    }
  }

  if (h_clovinv) {
    allocateCloverField(&cudaCloverInvPrecise, X, inv_param->cl_pad, inv_param->clover_cuda_prec);
    loadCloverField(cudaCloverInvPrecise, h_clovinv, inv_param->clover_cpu_prec, inv_param->clover_order);
    inv_param->cloverGiB += 2.0*cudaCloverInvPrecise.even.bytes / (1<<30);
    
    if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
      allocateCloverField(&cudaCloverInvSloppy, X, inv_param->cl_pad, inv_param->clover_cuda_prec_sloppy);
      loadCloverField(cudaCloverInvSloppy, h_clovinv, inv_param->clover_cpu_prec, inv_param->clover_order);
      inv_param->cloverGiB += 2.0*cudaCloverInvSloppy.even.bytes / (1<<30);
    } else {
      cudaCloverInvSloppy = cudaCloverInvPrecise;
    }
  }

}


#if 0
// discard clover term but keep the inverse
void discardCloverQuda(QudaInvertParam *inv_param)
{
  inv_param->cloverGiB -= 2.0*cudaCloverPrecise.even.bytes / (1<<30);
  freeCloverField(&cudaCloverPrecise);
  if (cudaCloverSloppy.even.clover) {
    inv_param->cloverGiB -= 2.0*cudaCloverSloppy.even.bytes / (1<<30);
    freeCloverField(&cudaCloverSloppy);
  }
}
#endif


void endQuda(void)
{
  cudaColorSpinorField::freeBuffer();
  freeGaugeField(&cudaGaugePrecise);
  freeGaugeField(&cudaGaugeSloppy);
  freeGaugeField(&cudaFatLinkPrecise);
  freeGaugeField(&cudaFatLinkSloppy);
  freeGaugeField(&cudaLongLinkPrecise);
  freeGaugeField(&cudaLongLinkSloppy);
  if (cudaCloverPrecise.even.clover) freeCloverField(&cudaCloverPrecise);
  if (cudaCloverSloppy.even.clover) freeCloverField(&cudaCloverSloppy);
  if (cudaCloverInvPrecise.even.clover) freeCloverField(&cudaCloverInvPrecise);
  if (cudaCloverInvSloppy.even.clover) freeCloverField(&cudaCloverInvSloppy);
  endBlas();
}


void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc)
{
  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    kappa *= cudaGaugePrecise.anisotropy;
  }

  switch (inv_param->dslash_type) {
  case QUDA_WILSON_DSLASH:
    diracParam.type = pc ? QUDA_WILSONPC_DIRAC : QUDA_WILSON_DIRAC;
    break;
  case QUDA_CLOVER_WILSON_DSLASH:
    diracParam.type = pc ? QUDA_CLOVERPC_DIRAC : QUDA_CLOVER_DIRAC;
    break;
  case QUDA_DOMAIN_WALL_DSLASH:
    diracParam.type = pc ? QUDA_DOMAIN_WALLPC_DIRAC : QUDA_DOMAIN_WALL_DIRAC;
    break;
  case QUDA_ASQTAD_DSLASH:
    diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
    break;
  case QUDA_TWISTED_MASS_DSLASH://Do we need to specify non-deg case?
    diracParam.type = pc ? QUDA_TWISTED_MASSPC_DIRAC : QUDA_TWISTED_MASS_DIRAC;
    break;
  default:
    errorQuda("Unsupported dslash_type");
  }

  diracParam.matpcType = inv_param->matpc_type;
  diracParam.dagger = inv_param->dagger;
  diracParam.gauge = &cudaGaugePrecise;
  diracParam.clover = &cudaCloverPrecise;
  diracParam.cloverInv = &cudaCloverInvPrecise;
  diracParam.kappa = kappa;
  diracParam.mass = inv_param->mass;
  diracParam.m5 = inv_param->m5;
  diracParam.mu = inv_param->mu;
  diracParam.epsilon = inv_param->epsilon;//NEW!
  diracParam.verbose = inv_param->verbosity;
}


void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc)
{
  setDiracParam(diracParam, inv_param, pc);

  diracParam.gauge = &cudaGaugeSloppy;
  diracParam.clover = &cudaCloverSloppy;
  diracParam.cloverInv = &cudaCloverInvSloppy;
}


static void massRescale(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type, 
			QudaMassNormalization mass_normalization, cudaColorSpinorField &b)
{    
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (mass_normalization != QUDA_MASS_NORMALIZATION) {
      errorQuda("Staggered code only supports QUDA_MASS_NORMALIZATION");
    }
    return;
  }

  // multiply the source to compensate for normalization of the Dirac operator, if necessary
  switch (solution_type) {
  case QUDA_MAT_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(2.0*kappa, b);
    }
    break;
  case QUDA_MATDAG_MAT_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(4.0*kappa*kappa, b);
    }
    break;
  case QUDA_MATPC_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(2.0*kappa, b);
    }
    break;
  case QUDA_MATPCDAG_MATPC_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	axCuda(16.0*pow(kappa,4), b);
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
    }
    break;
  default:
    errorQuda("Solution type %d not supported", solution_type);
  }
}


void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)
{
  ColorSpinorParam cpuParam(h_in, *inv_param, cudaGaugePrecise.X);
  cpuParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuColorSpinorField hIn(cpuParam);

  cudaColorSpinorField in(hIn, cudaParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    axCuda(cudaGaugePrecise.anisotropy, in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->Dslash(out, in, parity); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;
  cpuColorSpinorField hOut(cpuParam);
  out.saveCPUSpinorField(hOut); // since this is a reference, this won't work: hOut = out;
}


void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
	     inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  if (!pc) cudaGaugePrecise.X[0] *= 2;
  ColorSpinorParam cpuParam(h_in, *inv_param, cudaGaugePrecise.X);
  if (!pc) {
    cudaGaugePrecise.X[0] /= 2;
    cpuParam.siteSubset = QUDA_FULL_SITE_SUBSET;;
  } else {
    cpuParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuColorSpinorField hIn(cpuParam);
  cudaColorSpinorField in(hIn, cudaParam);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->M(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  }

  cpuParam.v = h_out;
  cpuColorSpinorField hOut(cpuParam);
  out.saveCPUSpinorField(hOut); // since this is a reference, this won't work: hOut = out;
}


void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
	     inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  if (!pc) cudaGaugePrecise.X[0] *= 2;
  ColorSpinorParam cpuParam(h_in, *inv_param, cudaGaugePrecise.X);
  if (!pc) {
    cudaGaugePrecise.X[0] /= 2;
    cpuParam.siteSubset = QUDA_FULL_SITE_SUBSET;;
  } else {
    cpuParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  ColorSpinorParam cudaParam(cpuParam, *inv_param);

  cpuColorSpinorField hIn(cpuParam);
  cudaColorSpinorField in(hIn, cudaParam);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  //  double kappa = inv_param->kappa;
  //  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= cudaGaugePrecise.anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->MdagM(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(1.0/pow(2.0*kappa,4), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  }

  cpuParam.v = h_out;
  cpuColorSpinorField hOut(cpuParam);
  out.saveCPUSpinorField(hOut); // since this is a reference, this won't work: hOut = out;
}


void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  checkInvertParam(param);

  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
		   param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
		      param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  param->spinorGiB = cudaGaugePrecise.volume * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *dirac;
  Dirac *diracSloppy;

  // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param, pc_solve);

  cpuColorSpinorField* h_b;
  cpuColorSpinorField* h_x;
  cudaColorSpinorField* b;
  cudaColorSpinorField* x;
  cudaColorSpinorField *in, *out;

  if (param->dslash_type == QUDA_ASQTAD_DSLASH) {

    if(param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION  
	&& param->solution_type != QUDA_MATDAG_MAT_SOLUTION){
	errorQuda("Your solution type not supported for staggered. "
	"Only QUDA_MATPCDAG_MATPC_SOLUTION and QUDA_MATDAG_MAT_SOLUTION is supported.");
     }

    ColorSpinorParam csParam;
    csParam.precision = param->cpu_prec;
    csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
    csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;  
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.nColor=3;
    csParam.nSpin=1;
    csParam.nDim=4;
    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    
    if (param->solve_type == QUDA_NORMEQ_SOLVE) {
      csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      csParam.x[0] = 2*cudaFatLinkPrecise.X[0];
    } else if (param->solve_type == QUDA_NORMEQ_PC_SOLVE) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] = cudaFatLinkPrecise.X[0];
    } else {
      errorQuda("Direct solve_type not supported for staggered");
    }
    csParam.x[1] = cudaFatLinkPrecise.X[1];
    csParam.x[2] = cudaFatLinkPrecise.X[2];
    csParam.x[3] = cudaFatLinkPrecise.X[3];
    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
    csParam.v = hp_b;  
    h_b = new cpuColorSpinorField(csParam);
    
    csParam.v = hp_x;
    h_x = new cpuColorSpinorField(csParam);
    
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    
    csParam.pad = param->sp_pad;
    csParam.precision = param->cuda_prec;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    
    b = new cudaColorSpinorField(csParam);
    
    diracParam.fatGauge = &cudaFatLinkPrecise;
    diracParam.longGauge = &cudaLongLinkPrecise;
    
    dirac = Dirac::create(diracParam); // create the Dirac operator
    
    diracParam.fatGauge = &cudaFatLinkSloppy;
    diracParam.longGauge = &cudaLongLinkSloppy;
    
    setDiracSloppyParam(diracParam, param, pc_solve);
    diracSloppy = Dirac::create(diracParam);
    
    *b = *h_b; // send data from CPU to GPU
    
    csParam.create = QUDA_COPY_FIELD_CREATE;  
    x = new cudaColorSpinorField(*h_x, csParam); // solution  
    csParam.create = QUDA_ZERO_FIELD_CREATE;
        
  } else if (param->dslash_type == QUDA_WILSON_DSLASH ||
	     param->dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
	     param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
	     param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {

    // temporary hack
    if (!pc_solution) cudaGaugePrecise.X[0] *= 2;
    ColorSpinorParam cpuParam(hp_b, *param, cudaGaugePrecise.X);
    if (!pc_solution) {
      cudaGaugePrecise.X[0] /= 2;
      cpuParam.siteSubset = QUDA_FULL_SITE_SUBSET;;
    } else {
      cpuParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    }
    
    ColorSpinorParam cudaParam(cpuParam, *param);
    
    h_b = new cpuColorSpinorField(cpuParam);
    cpuParam.v = hp_x;
    h_x = new cpuColorSpinorField(cpuParam);
    
    b = new cudaColorSpinorField(*h_b, cudaParam); // download source
    
    if (param->verbosity >= QUDA_VERBOSE) {
      printfQuda("Source: CPU = %f, CUDA copy = %f\n", norm2(*h_b),norm2(*b));
    }
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    x = new cudaColorSpinorField(cudaParam); // solution
    
    // if using preconditioning but solving the full system
    if (pc_solve && !pc_solution) {
      cudaParam.x[0] /= 2;
      cudaParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    }
    
    dirac = Dirac::create(diracParam); // create the Dirac operator
    
    setDiracSloppyParam(diracParam, param, pc_solve);
    diracSloppy = Dirac::create(diracParam);
    
  } else {
    errorQuda("Invalid dslash_type");
  }
  
  dirac->prepare(in, out, *x, *b, param->solution_type);
  if (param->verbosity >= QUDA_VERBOSE) printfQuda("Prepared source = %f\n", norm2(*in));   

  massRescale(param->dslash_type, diracParam.kappa, param->solution_type, param->mass_normalization, *in);
  if (param->verbosity >= QUDA_VERBOSE) printfQuda("Mass rescale done\n");   
  
  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATDAG_MAT_SOLUTION && param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda(*out, *in);
      dirac->Mdag(*in, *out);
    }
    invertCgCuda(DiracMdagM(dirac), DiracMdagM(diracSloppy), *out, *in, param);
    break;
    
  case QUDA_CGS_INVERTER:

    if (param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
       invertCgsCuda(DiracMdag(dirac), DiracMdag(diracSloppy), *out, *in, param);
       copyCuda(*in, *out);
    }

     invertCgsCuda(DiracM(dirac), DiracM(diracSloppy), *out, *in, param);
     break;
    
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(DiracMdag(dirac), DiracMdag(diracSloppy), *out, *in, param);
      copyCuda(*in, *out);
    }
    invertBiCGstabCuda(DiracM(dirac), DiracM(diracSloppy), *out, *in, param);
    break;
  default:
    errorQuda("Inverter type %d not implemented", param->inv_type);
  }
  
  if (param->verbosity >= QUDA_VERBOSE){
    printfQuda("Solution = %f\n",norm2(*x));
  }
  dirac->reconstruct(*x, *b, param->solution_type);
  
  x->saveCPUSpinorField(*h_x); // since this is a reference, this won't work: h_x = x;
  
  if (param->verbosity >= QUDA_VERBOSE){
    printfQuda("Reconstructed: CUDA solution = %f, CPU copy = %f\n", norm2(*x), norm2(*h_x));
  }
  
  delete diracSloppy;
  delete dirac;
  
  delete h_b;
  delete h_x;
  delete b;
  delete x;
  
  return;
}


void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			  double* offsets, int num_offsets, double* residue_sq)
{
  checkInvertParam(param);

  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
    errorQuda("Multi-shift solver only supports staggered");
  }
  if (param->solve_type != QUDA_NORMEQ_SOLVE &&
      param->solve_type != QUDA_NORMEQ_PC_SOLVE) {
    errorQuda("Direct solve_type not supported for staggered");
  }
  if (num_offsets <= 0){
    warningQuda("invertMultiShiftQuda() called with no offsets");
    return;
  }

  if(param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION
      && param->solution_type != QUDA_MATDAG_MAT_SOLUTION){
      errorQuda("Your solution type not supported for staggered. "
      "Only QUDA_MATPCDAG_MATPC_SOLUTION and QUDA_MATDAG_MAT_SOLUTION is supported.");
  }



  bool pc_solve = (param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  double low_offset = offsets[0];
  int low_index = 0;
  for (int i=1;i < num_offsets;i++){
    if (offsets[i] < low_offset){
      low_offset = offsets[i];
      low_index = i;
    }
  }
  
  void* hp_x[num_offsets];
  void* hp_b = _hp_b;
  for(int i=0;i < num_offsets;i++){
    hp_x[i] = _hp_x[i];
  }
  
  if (low_index != 0){
    void* tmp = hp_x[0];
    hp_x[0] = hp_x[low_index] ;
    hp_x[low_index] = tmp;
    
    double tmp1 = offsets[0];
    offsets[0]= offsets[low_index];
    offsets[low_index] =tmp1;
  }
    
  ColorSpinorParam csParam;
  csParam.precision = param->cpu_prec;
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;  
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if (param->solve_type == QUDA_NORMEQ_SOLVE) {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    csParam.x[0] = 2*cudaFatLinkPrecise.X[0];
  } else if (param->solve_type == QUDA_NORMEQ_PC_SOLVE) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] = cudaFatLinkPrecise.X[0];
  } else {
    errorQuda("Direct solve_type not supported for staggered");
  }
  csParam.x[1] = cudaFatLinkPrecise.X[1];
  csParam.x[2] = cudaFatLinkPrecise.X[2];
  csParam.x[3] = cudaFatLinkPrecise.X[3];
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.v = hp_b;  
  cpuColorSpinorField h_b(csParam);
  
  cpuColorSpinorField* h_x[num_offsets];
  
  for (int i=0; i<num_offsets; i++) {
    csParam.v = hp_x[i];
    h_x[i] = new cpuColorSpinorField(csParam);
  }
  
  csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
  csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  csParam.pad = param->sp_pad;
  csParam.precision = param->cuda_prec;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  
  cudaColorSpinorField b(csParam);
  
  // set the mass in the invert_param
  param->mass = sqrt(offsets[0]/4);
  
  // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param, pc_solve);
  diracParam.fatGauge = &cudaFatLinkPrecise;
  diracParam.longGauge = &cudaLongLinkPrecise;
  
  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator

  diracParam.fatGauge = &cudaFatLinkSloppy;
  diracParam.longGauge = &cudaLongLinkSloppy;
  
  setDiracSloppyParam(diracParam, param, pc_solve);
  Dirac *diracSloppy = Dirac::create(diracParam);
  
  b = h_b; //send data from CPU to GPU
  
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField* x[num_offsets]; // solution  
  for (int i=0; i < num_offsets; i++) {
    x[i] = new cudaColorSpinorField(csParam);
  }
  cudaColorSpinorField tmp(csParam); // temporary
  invertMultiShiftCgCuda(DiracMdagM(dirac), DiracMdagM(diracSloppy), x, b, param, offsets, num_offsets, residue_sq);    
  
  for (int i=0; i < num_offsets; i++) {
    x[i]->saveCPUSpinorField(*h_x[i]);
  }
  
  for(int i=0; i<num_offsets; i++) {
    delete h_x[i];
    delete x[i];
  }
  delete diracSloppy;
  delete dirac;
  
  return;
}

#define NF 2

void invertMultiFlavorQuda(void *hp_xf1, void *hp_xf2, void *hp_bf1, void *hp_bf2, QudaInvertParam *param)
{
  checkInvertParam(param);

  bool pc_solve    = (param->solve_type == QUDA_DIRECT_PC_SOLVE || param->solve_type == QUDA_NORMEQ_PC_SOLVE);
  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||  param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  param->spinorGiB = cudaGaugePrecise.volume * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) 
  {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } 
  else 
  {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *dirac;
  Dirac *diracSloppy;

  // set the Dirac operator parameters
  DiracParam diracParam;
  setDiracParam(diracParam, param, pc_solve);

  cpuColorSpinorField  *h_bf1, *h_bf2;
  cpuColorSpinorField  *h_xf1, *h_xf2;
  cudaColorSpinorField *bf1, *bf2;
  cudaColorSpinorField *xf1, *xf2;
  cudaColorSpinorField *inf1, *inf2;
  cudaColorSpinorField *outf1, *outf2;  

  if (param->dslash_type != QUDA_TWISTED_MASS_DSLASH) errorQuda("Invalid dslash_type");
  
  // temporary hack
  if (!pc_solution) cudaGaugePrecise.X[0] *= 2;
  
  ColorSpinorParam cpuParam1(hp_bf1, *param, cudaGaugePrecise.X);
  ColorSpinorParam cpuParam2(hp_bf2, *param, cudaGaugePrecise.X);
  
  if (!pc_solution) 
  {
      cudaGaugePrecise.X[0] /= 2;
      cpuParam1.siteSubset = QUDA_FULL_SITE_SUBSET;
      cpuParam2.siteSubset = QUDA_FULL_SITE_SUBSET;      
  } 
  else 
  {
      cpuParam1.siteSubset = QUDA_PARITY_SITE_SUBSET;
      cpuParam2.siteSubset = QUDA_PARITY_SITE_SUBSET;      
  }
    
  ColorSpinorParam cudaParam(cpuParam1, *param);
    
  h_bf1 = new cpuColorSpinorField(cpuParam1);
  h_bf2 = new cpuColorSpinorField(cpuParam2);
  
  cpuParam1.v = hp_xf1;
  cpuParam2.v = hp_xf2;  
    
  h_xf1 = new cpuColorSpinorField(cpuParam1);
  h_xf2 = new cpuColorSpinorField(cpuParam2);  
  bf1 = new cudaColorSpinorField(*h_bf1, cudaParam); // download source
  bf2 = new cudaColorSpinorField(*h_bf2, cudaParam); // download source
    
  if (param->verbosity >= QUDA_VERBOSE) 
  {
      printfQuda("Source: CPU = %f, CUDA copy = %f\n", norm2(*h_bf1) + norm2(*h_bf2), norm2(*bf1) + norm2(*bf2));
  }
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  
  xf1 = new cudaColorSpinorField(cudaParam); // solution
  xf2 = new cudaColorSpinorField(cudaParam); // solution
    
    // if using preconditioning but solving the full system
  if (pc_solve && !pc_solution) 
  {
      cudaParam.x[0] /= 2;
      cudaParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
    
  dirac = Dirac::create(diracParam); // create the (TM)Dirac operator
    
  setDiracSloppyParam(diracParam, param, pc_solve);
  diracSloppy = Dirac::create(diracParam);
   
  dirac->prepare(inf1, inf2, outf1, outf2, *xf1, *xf2, *bf1, *bf2, param->solution_type);
  if (param->verbosity >= QUDA_VERBOSE) printfQuda("Prepared source = %f\n", norm2(*inf1) + norm2(*inf2));   

  massRescale(param->dslash_type, diracParam.kappa, param->solution_type, param->mass_normalization, *inf1);
  massRescale(param->dslash_type, diracParam.kappa, param->solution_type, param->mass_normalization, *inf2);
  
  if (param->verbosity >= QUDA_VERBOSE) printfQuda("Mass rescale done\n");   
  
  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATDAG_MAT_SOLUTION && param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda(*outf1, *inf1), copyCuda(*outf2, *inf2);
      dirac->Mdag(*inf1, *inf2, *outf1, *outf2);
    }
    invertTMCgCuda(DiracMdagM(dirac), DiracMdagM(diracSloppy), *outf1, *outf2, *inf1, *inf2, param);
    break;
    
  case QUDA_CGS_INVERTER:

    //if (param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
       //invertCgsCuda(DiracMdag(dirac), DiracMdag(diracSloppy), *out[0], *out[1], *in[0], *in[1], param);
       //copyCuda(*in[0], *out[0]), copyCuda(*in[1], *out[1]);
    //}

     //invertCgsCuda(DiracM(dirac), DiracM(diracSloppy), *out[0], *out[1], *in[0], *in[1], param);
     break;
    
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertTMBiCGstabCuda(DiracMdag(dirac), DiracMdag(diracSloppy), *outf1, *outf2, *inf1, *inf2, param);
      copyCuda(*inf1, *outf1), copyCuda(*inf2, *outf2);
    }
    invertTMBiCGstabCuda(DiracM(dirac), DiracM(diracSloppy), *outf1, *outf2, *inf1, *inf2, param);
    break;
  default:
    errorQuda("Inverter type %d not implemented", param->inv_type);
  }
  
  if (param->verbosity >= QUDA_VERBOSE){
    printfQuda("Solution = %f\n", (norm2(*xf1) + norm2(*xf2)));
  }
  dirac->reconstruct(*xf1, *xf2, *bf1, *bf2, param->solution_type);
  
 
  xf1->saveCPUSpinorField(*h_xf1); // since this is a reference, this won't work: h_x = x;
  xf2->saveCPUSpinorField(*h_xf2); // since this is a reference, this won't work: h_x = x;  
  
  if (param->verbosity >= QUDA_VERBOSE){
    printfQuda("Reconstructed: CUDA solution = %f, CPU copy = %f\n", (norm2(*xf1)+norm2(*xf2)), (norm2(*h_xf1)+norm2(*h_xf2)));
  }
  
  delete diracSloppy;
  delete dirac;

  delete h_bf1;
  delete h_bf2;  
  delete h_xf1;
  delete h_xf2;  
  delete bf1;
  delete bf2;  
  delete xf1;
  delete xf2;  

  return;
}

