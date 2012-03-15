#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <test_util.h>
#include <twisted_mass_dslash_reference.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
const int test_type = 0;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const QudaDagType dagger = QUDA_DAG_NO;     // apply Dslash or Dslash dagger?
const int transfer = 0; // include transfer time in the benchmark?

const int loops = 10;

QudaPrecision cpu_prec  = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec = QUDA_HALF_PRECISION;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

FullGauge gauge;

cpuColorSpinorField  *spinor1, *spinor2,   *spinorOut1, *spinorOut2, *spinorRef1, *spinorRef2;
cudaColorSpinorField *cudaSpinor1, *cudaSpinor2, *cudaSpinorOut1, *cudaSpinorOut2, *tmp1=0, *tmp2=0;

void *hostGauge[4];

Dirac *dirac;

void init() {

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = 16;
  gauge_param.X[1] = 16;
  gauge_param.X[2] = 16;
  gauge_param.X[3] = 32;
  setDims(gauge_param.X);

  gauge_param.anisotropy = 2.3;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_8;
  gauge_param.reconstruct_sloppy = gauge_param.reconstruct;
  gauge_param.cuda_prec_sloppy = gauge_param.cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  inv_param.kappa = 0.1;
  inv_param.mu = 0.1;
  inv_param.epsilon = 0.0;
  inv_param.twist_flavor = QUDA_TWIST_DUPLET;

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = dagger;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;

  gauge_param.ga_pad = 0;
  inv_param.sp_pad   = 0;
  inv_param.cl_pad   = 0;

  //gauge_param.ga_pad = 24*24*24;
  //inv_param.sp_pad = 24*24*24;
  //inv_param.cl_pad = 24*24*24;

  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (test_type == 2) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
  }

  inv_param.dslash_type = QUDA_TWISTED_MASS_DSLASH;

  inv_param.verbosity = QUDA_VERBOSE;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.twistFlavor = inv_param.twist_flavor;
  csParam.nDim = 4;
  
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  
  if (test_type < 2) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }    
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  
  spinor1    = new cpuColorSpinorField(csParam);
  spinorOut1 = new cpuColorSpinorField(csParam);
  spinorRef1 = new cpuColorSpinorField(csParam);
  
  spinor2    = new cpuColorSpinorField(csParam);
  spinorOut2 = new cpuColorSpinorField(csParam);
  spinorRef2 = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  
  spinor1->Source(QUDA_RANDOM_SOURCE);
  spinor2->Source(QUDA_RANDOM_SOURCE);
  
  printfQuda("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);

  printfQuda("Sending gauge field to GPU\n");

  loadGaugeQuda(hostGauge, &gauge_param);
  gauge = cudaGaugePrecise;

  if (!transfer) {
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if (test_type < 2) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor1    = new cudaColorSpinorField(csParam);
    cudaSpinor2    = new cudaColorSpinorField(csParam);    
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut1 = new cudaColorSpinorField(csParam);
    cudaSpinorOut2 = new cudaColorSpinorField(csParam);    

    if (test_type == 2) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp1 = new cudaColorSpinorField(csParam);
    tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor1 = *spinor1;
    *cudaSpinor2 = *spinor2;
    
    std::cout << "Flavor1 " << "Source: CPU = " << norm2(*spinor1) << ", CUDA = " << 
      norm2(*cudaSpinor1) << std::endl;

    std::cout << "Flavor2 " << "Source: CPU = " << norm2(*spinor2) << ", CUDA = " << 
      norm2(*cudaSpinor2) << std::endl;

      
    bool pc = (test_type != 2);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.verbose = QUDA_VERBOSE;
    diracParam.tmp1 = tmp1;
    diracParam.tmp2 = tmp2;
    
    dirac = Dirac::create(diracParam);
  } else {
    std::cout << "Flavor1 " << "Source: CPU = " << norm2(*spinor1) << std::endl;
    std::cout << "Flavor1 " << "Source: CPU = " << norm2(*spinor2) << std::endl;    
  }
    
}

void end() {
  if (!transfer) {
    delete dirac;
    delete cudaSpinor1;
    delete cudaSpinor2;    
    delete cudaSpinorOut1;
    delete cudaSpinorOut2;    
    delete tmp1;
    delete tmp2;
  }

  // release memory
  delete spinor1;
  delete spinor2;  
  delete spinorOut1;
  delete spinorOut2;  
  delete spinorRef1;
  delete spinorRef2;
  
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  endQuda();
}

// execute kernel
double ndegDslashCUDA() {

  printfQuda("Executing %d kernel loops...\n", loops);
  fflush(stdout);
  stopwatchStart();
  for (int i = 0; i < loops; i++) {
    switch (test_type) {
    case 0:
      if (transfer) {
	//ndegDslashQuda(spinorOut1->v, spinorOut2->v, spinor1->v, spinor2->v, &inv_param, parity);
      } else if(inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH){
	dirac->Dslash(*cudaSpinorOut1, *cudaSpinorOut2, *cudaSpinor1, *cudaSpinor2, parity);
      }
      break;
    case 1:
    {
        printfQuda("\ncompute Mdag\n");//See CPU version!
        dirac->Mdag(*cudaSpinorOut1, *cudaSpinorOut2, *cudaSpinor1, *cudaSpinor2);
    }
      break;
    case 2:
      if (transfer) {
	//MatQuda(spinorOut->v, spinor->v, &inv_param);
      } else {
	dirac->M(*cudaSpinorOut1, *cudaSpinorOut2, *cudaSpinor1, *cudaSpinor2);
      }
      break;
    }
  }
    
  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  cudaThreadSynchronize();
  double secs = stopwatchReadSeconds();
  printf("done.\n\n");

  return secs;
}

void ndegDslashRef() {

  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:
    ndeg_dslash(spinorRef1->v, spinorRef2->v, hostGauge, spinor1->v, spinor2->v, inv_param.kappa, inv_param.mu, inv_param.epsilon,
	   parity, dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  case 1: //NOTE !dagger   
    ndeg_matpc(spinorRef1->v, spinorRef2->v, hostGauge, spinor1->v, spinor2->v, inv_param.kappa, inv_param.mu, inv_param.epsilon,
	  inv_param.matpc_type, !dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  case 2:
    //mat(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	//dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  default:
    printf("Test type not defined\n");
    exit(-1);
  }

  printf("done.\n");
    
}

int main(int argc, char **argv)
{
  init();

  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  float sharedKB = 0;//(float)dslashCudaSharedBytes(inv_param.cuda_prec) / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);
  
  int attempts = 1;
  ndegDslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = ndegDslashCUDA();

    if (!transfer) *spinorOut1 = *cudaSpinorOut1, *spinorOut2 = *cudaSpinorOut2;

    // print timing information
    printf("%fms per loop\n", 1000*secs);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int floats = test_type ? 2*(7*24+8*gauge_param.packed_size+24)+24 : 7*24+8*gauge_param.packed_size+24;

    printf("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printf("GiB/s = %f\n\n", Vh*floats*sizeof(float)/((secs/loops)*(1<<30)));
    
    if (!transfer) {
      std::cout << "Flavor1 " << "Results: CPU = " << norm2(*spinorRef1) << ", CUDA = " << norm2(*cudaSpinorOut1) << 
	", CPU-CUDA = " << norm2(*spinorOut1) << std::endl;
      std::cout << "Flavor2 " << "Results: CPU = " << norm2(*spinorRef2) << ", CUDA = " << norm2(*cudaSpinorOut2) << 
	", CPU-CUDA = " << norm2(*spinorOut2) << std::endl;	
    } else {
      std::cout << "Flavor1 " << "Result: CPU = " << norm2(*spinorRef1) << ", CPU-CUDA = " << norm2(*spinorOut1) << std::endl;
      std::cout << "Flavor2 " << "Result: CPU = " << norm2(*spinorRef2) << ", CPU-CUDA = " << norm2(*spinorOut2) << std::endl;      
    }
    
    cpuColorSpinorField::Compare(*spinorRef1, *spinorOut1);
    cpuColorSpinorField::Compare(*spinorRef2, *spinorOut2);    
  }    
  end();
}
