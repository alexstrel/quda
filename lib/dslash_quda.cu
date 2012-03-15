#include <stdlib.h>
#include <stdio.h>

//these are access control for staggered action
#if (__CUDA_ARCH__ >= 200)
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
#else
#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
#endif

#include <quda_internal.h>
#include <dslash_quda.h>

#define BLOCK_DIM 64

#include <dslash_textures.h>
#include <dslash_constants.h>

#include <staggered_dslash_def.h> // staggered Dslash kernels
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#include <dw_dslash_def.h>        // Domain Wall kernels
#include <tm_dslash_def.h>        // Twisted Mass kernels
#include <tm_ndeg_dslash_def.h>        // Twisted Mass kernels
#include <dslash_core/tm_core.h>  // solo twisted mass kernel
#include <clover_def.h>           // kernels for applying the clover term alone

#ifndef SHARED_FLOATS_PER_THREAD
#define SHARED_FLOATS_PER_THREAD 0
#endif

//BEGIN NEW
#ifndef SHARED_TMNDEG_FLOATS_PER_THREAD
#define SHARED_TMNDEG_FLOATS_PER_THREAD 0
#endif
#ifndef SHARED_TMDEG_FLOATS_PER_THREAD
#define SHARED_TMDEG_FLOATS_PER_THREAD 0
#endif
//END NEW

#include <blas_quda.h>

int dslashCudaSharedBytes(QudaPrecision precision) {
  return BLOCK_DIM*SHARED_FLOATS_PER_THREAD*precision;
}

template <int spinorN, typename spinorFloat, typename gaugeFloat>
void dslashCuda(spinorFloat *out, float *outNorm, const gaugeFloat *gauge0, const gaugeFloat *gauge1, 
		const QudaReconstructType reconstruct, const spinorFloat *in, const float *inNorm,
		const int parity, const int dagger, const spinorFloat *x, const float *xNorm, 
		const double &a, const int volume, const int length) {

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<spinorN>(length, in, inNorm, x, xNorm);

  if (x==0) { // not doing xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	dslash18Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity);
      } else {
	dslash18DaggerKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	dslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity);
      } else {
	dslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity);
      }
    } else {
      if (!dagger) {
	dslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity);
      } else {
	dslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity);
      }
    }
  } else { // doing xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	dslash18XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, x, xNorm, a);
      } else {
	dslash18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, x, xNorm, a);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	dslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, x, xNorm, a);
      } else {
	dslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, x, xNorm, a);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {
      if (!dagger) {
	dslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, x, xNorm, a);
      } else {
	dslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, x, xNorm, a);
      }
    }
  }
 
  unbindSpinorTex<spinorN>(in, inNorm, x, xNorm);
 
}

// Wilson wrappers
void dslashCuda(void *out, void *outNorm, const FullGauge gauge, const void *in, const void *inNorm, 
		const int parity, const int dagger, const void *x, const void *xNorm, 
		const double k, const int volume, const int length, const QudaPrecision precision) {

#ifdef GPU_WILSON_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    dslashCuda<2>((double2*)out, (float*)outNorm, (double2*)gauge0, (double2*)gauge1, 
		  gauge.reconstruct, (double2*)in, (float*)inNorm, parity, dagger, 
		  (double2*)x, (float*)xNorm, k, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    dslashCuda<4>((float4*)out, (float*)outNorm, (float4*)gauge0, (float4*)gauge1,
    		  gauge.reconstruct, (float4*)in, (float*)inNorm, parity, dagger, 
		  (float4*)x, (float*)xNorm, k, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    dslashCuda<4>((short4*)out, (float*)outNorm, (short4*)gauge0, (short4*)gauge1,
		  gauge.reconstruct, (short4*)in, (float*)inNorm, parity, dagger, 
		  (short4*)x, (float*)xNorm, k, volume, length);
  }
  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

}


template <int N, typename spinorFloat, typename cloverFloat>
void cloverCuda(spinorFloat *out, float *outNorm, const cloverFloat *clover,
		const float *cloverNorm, const spinorFloat *in, const float *inNorm, 
		const int parity, const int volume, const int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm);
  cloverKernel<<<gridDim, blockDim, shared_bytes>>> 
    (out, outNorm, clover, cloverNorm, in, inNorm, parity);
  unbindSpinorTex<N>(in, inNorm);
}

void cloverCuda(void *out, void *outNorm, const FullGauge gauge, const FullClover clover, 
		const void *in, const void *inNorm, const int parity, const int volume,
		const int length, const QudaPrecision precision) {

#ifdef GPU_WILSON_DIRAC
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

  if (precision != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    cloverCuda<2>((double2*)out, (float*)outNorm, (double2*)cloverP, 
		  (float*)cloverNormP, (double2*)in, 
		  (float*)inNorm, parity, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    cloverCuda<4>((float4*)out, (float*)outNorm, (float4*)cloverP, 
		  (float*)cloverNormP, (float4*)in, 
		  (float*)inNorm, parity, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    cloverCuda<4>((short4*)out, (float*)outNorm, (short4*)cloverP, 
		  (float*)cloverNormP, (short4*)in,
		  (float*)inNorm, parity, volume, length);
  }
  unbindCloverTex(clover);

  checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif

}

// Clover wrappers
template <int N, typename spinorFloat, typename cloverFloat, typename gaugeFloat>
void cloverDslashCuda(spinorFloat *out, float *outNorm, const gaugeFloat gauge0, 
		      const gaugeFloat gauge1, const QudaReconstructType reconstruct, 
		      const cloverFloat *clover, const float *cloverNorm, const spinorFloat *in, 
		      const float* inNorm, const int parity, const int dagger, const spinorFloat *x, 
		      const float* xNorm, const double &a, const int volume, const int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm, x, xNorm);

  if (x==0) { // not xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	cloverDslash18Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
      } else {
	cloverDslash18DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	cloverDslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
      } else {
	cloverDslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
      }
    } else {
      if (!dagger) {
	cloverDslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
      } else {
	cloverDslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity);
      }
    }
  } else { // doing xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	cloverDslash18XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, x, xNorm, a);
      } else {
	cloverDslash18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, x, xNorm, a);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	cloverDslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, x, xNorm, a);
      } else {
	cloverDslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, x, xNorm, a);
      }
    } else {
      if (!dagger) {
	cloverDslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, x, xNorm, a);
      } else {
	cloverDslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, clover, cloverNorm, in, inNorm, parity, x, xNorm, a);
      }
    }
  }

  unbindSpinorTex<N>(in, inNorm, x, xNorm);
}

void cloverDslashCuda(void *out, void *outNorm, const FullGauge gauge, const FullClover cloverInv,
		      const void *in, const void *inNorm, const int parity, const int dagger, 
		      const void *x, const void *xNorm, const double a, const int volume, 
		      const int length, const QudaPrecision precision) {

#ifdef GPU_WILSON_DIRAC
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

  void *gauge0, *gauge1;

  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  if (precision != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    cloverDslashCuda<2>((double2*)out, (float*)outNorm, (double2*)gauge0, (double2*)gauge1, 
			gauge.reconstruct, (double2*)cloverP, (float*)cloverNormP, (double2*)in, 
			(float*)inNorm, parity, dagger, (double2*)x, (float*)xNorm, a, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    cloverDslashCuda<4>((float4*)out, (float*)outNorm, (float4*)gauge0, (float4*)gauge1, 
			gauge.reconstruct, (float4*)cloverP, (float*)cloverNormP, (float4*)in, 
			(float*)inNorm, parity, dagger, (float4*)x, (float*)xNorm, a, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    cloverDslashCuda<4>((short4*)out, (float*)outNorm, (short4*)gauge0, (short4*)gauge1, 
			gauge.reconstruct, (short4*)cloverP, (float*)cloverNormP, (short4*)in,
			(float*)inNorm, parity, dagger, (short4*)x, (float*)xNorm, a, volume, length);
  }

  unbindGaugeTex(gauge);
  unbindCloverTex(cloverInv);

  checkCudaError();
#else
  errorQuda("Clover dslash has not been built");
#endif


}

// Domain wall wrappers
template <int N, typename spinorFloat, typename gaugeFloat>
void domainWallDslashCuda(spinorFloat *out, float *outNorm, const gaugeFloat gauge0, 
			  const gaugeFloat gauge1, const QudaReconstructType reconstruct, 
			  const spinorFloat *in, const float* inNorm, const int parity, const int dagger, const spinorFloat *x, 
			  const float* xNorm, const double &m_f, const double &k2, const int volume_5d, const int length)
{

  dim3 gridDim(volume_5d/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm, x, xNorm);

  if (x==0) { // not xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	domainWallDslash18Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f);
      } else {
	domainWallDslash18DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	domainWallDslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f);
      } else {
	domainWallDslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f);
      }
    } else {
      if (!dagger) {
	domainWallDslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f);
      } else {
	domainWallDslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f);
      }
    }
  } else { // doing xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	domainWallDslash18XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f, x, xNorm, k2);
      } else {
	domainWallDslash18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f, x, xNorm, k2);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	domainWallDslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f, x, xNorm, k2);
      } else {
	domainWallDslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f, x, xNorm, k2);
      }
    } else {
      if (!dagger) {
	domainWallDslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f, x, xNorm, k2);
      } else {
	domainWallDslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, m_f, x, xNorm, k2);
      }
    }
  }

  unbindSpinorTex<N>(in, inNorm, x, xNorm);
}

void domainWallDslashCuda(void *out, void *outNorm, const FullGauge gauge, 
			  const void *in, const void *inNorm, const int parity, const int dagger, 
			  const void *x, const void *xNorm, const double m_f, const double k2, const int volume5d, 
			  const int length, const QudaPrecision precision) {

#ifdef GPU_DOMAIN_WALL_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    domainWallDslashCuda<2>((double2*)out, (float*)outNorm, (double2*)gauge0, (double2*)gauge1, 
			    gauge.reconstruct, (double2*)in, (float*)inNorm, parity, dagger, 
			    (double2*)x, (float*)xNorm, m_f, k2, volume5d, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    domainWallDslashCuda<4>((float4*)out, (float*)outNorm, (float4*)gauge0, (float4*)gauge1, 
			    gauge.reconstruct, (float4*)in, (float*)inNorm, parity, dagger, 
			    (float4*)x, (float*)xNorm, m_f, k2, volume5d, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    domainWallDslashCuda<4>((short4*)out, (float*)outNorm, (short4*)gauge0, (short4*)gauge1, 
			    gauge.reconstruct, (short4*)in, (float*)inNorm, parity, dagger, 
			    (short4*)x, (float*)xNorm, m_f, k2, volume5d, length);
  }

  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Domain wall dslash has not been built");
#endif

}

template <int spinorN, typename spinorFloat, typename fatGaugeFloat, typename longGaugeFloat>
  void staggeredDslashCuda(spinorFloat *out, float *outNorm, const fatGaugeFloat *fatGauge0, const fatGaugeFloat *fatGauge1, 
			   const longGaugeFloat* longGauge0, const longGaugeFloat* longGauge1, 
			   const QudaReconstructType reconstruct, const spinorFloat *in, const float *inNorm,
			   const int parity, const int dagger, const spinorFloat *x, const float *xNorm, 
			   const double &a, const int volume, const int length, const QudaPrecision precision) {
    
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  if (precision == QUDA_HALF_PRECISION && (volume % 128 == 0)) {
    blockDim.x = 128;
    gridDim.x = volume/blockDim.x;
  }
  
  int shared_bytes = blockDim.x*6*bindSpinorTex<spinorN>(length, in, inNorm, x, xNorm);
  
  if (x==0) { // not doing xpay
    if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	staggeredDslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity);
      } else {
	staggeredDslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_8){
	  
      if (!dagger) {
	staggeredDslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity);
      } else {
	staggeredDslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity);
      }
    }else{
      errorQuda("Invalid reconstruct value(%d) in function %s\n", reconstruct, __FUNCTION__);
    }
  } else { // doing xpay
    
    if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	staggeredDslash12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity, x, xNorm, a);
      } else {
	staggeredDslash12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity, x, xNorm, a);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {
      if (!dagger) {
	staggeredDslash8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity, x, xNorm, a);
      } else {
	staggeredDslash8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity, x, xNorm, a);
      }
    }else{
      errorQuda("Invalid reconstruct value in function %s\n", __FUNCTION__);	  
    }    
  }
  
  cudaThreadSynchronize();
  
  unbindSpinorTex<spinorN>(in, inNorm, x, xNorm);
}


template <int spinorN, typename spinorFloat, typename fatGaugeFloat, typename longGaugeFloat>
  void staggeredDslashNoReconCuda(spinorFloat *out, float *outNorm, const fatGaugeFloat *fatGauge0, const fatGaugeFloat *fatGauge1, 
				  const longGaugeFloat* longGauge0, const longGaugeFloat* longGauge1, 
				  const QudaReconstructType reconstruct, const spinorFloat *in, const float *inNorm,
				  const int parity, const int dagger, const spinorFloat *x, const float *xNorm, 
				  const double &a, const int volume, const int length, const QudaPrecision precision) 
{  
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  if (precision == QUDA_HALF_PRECISION) {
    blockDim.x = 128;
    gridDim.x = volume/blockDim.x;
  }
  int shared_bytes = blockDim.x*6*bindSpinorTex<spinorN>(length, in, inNorm, x, xNorm);
  
  if (x==0) { // not doing xpay
    if (!dagger) {
      staggeredDslash18Kernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity);
    } else {
      staggeredDslash18DaggerKernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity);
    }    
  } else { // doing xpay
    
    if (!dagger) {
      staggeredDslash18AxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	(out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity, x, xNorm, a);
    } else {
      staggeredDslash18DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	(out, outNorm, fatGauge0, fatGauge1, longGauge0, longGauge1, in, inNorm, parity, x, xNorm, a);
    }          
  }
  
  cudaThreadSynchronize();

  unbindSpinorTex<spinorN>(in, inNorm, x, xNorm);
}


void staggeredDslashCuda(void *out, void *outNorm, const FullGauge fatGauge, const FullGauge longGauge, 
			 const void *in, const void *inNorm, 
			 const int parity, const int dagger, const void *x, const void *xNorm, 
			 const double k, const int volume, const int length, const QudaPrecision precision) 
{

#ifdef GPU_STAGGERED_DIRAC
  void *fatGauge0, *fatGauge1;
  void* longGauge0, *longGauge1;
  bindFatGaugeTex(fatGauge, parity, &fatGauge0, &fatGauge1);
  bindLongGaugeTex(longGauge, parity, &longGauge0, &longGauge1);
    
  if (precision != fatGauge.precision || precision != longGauge.precision){
    errorQuda("Mixing gauge and spinor precision not supported");
  }
    
  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    if (longGauge.reconstruct == QUDA_RECONSTRUCT_NO){
      staggeredDslashNoReconCuda<2>((double2*)out, (float*)outNorm, (double2*)fatGauge0, (double2*)fatGauge1, 			       
				    (double2*)longGauge0, (double2*)longGauge1,
				    longGauge.reconstruct, (double2*)in, (float*)inNorm, parity, dagger, 
				    (double2*)x, (float*)xNorm, k, volume, length, precision);
    }else{
      staggeredDslashCuda<2>((double2*)out, (float*)outNorm, (double2*)fatGauge0, (double2*)fatGauge1, 			       
			     (double2*)longGauge0, (double2*)longGauge1,
			     longGauge.reconstruct, (double2*)in, (float*)inNorm, parity, dagger, 
			     (double2*)x, (float*)xNorm, k, volume, length, precision);
    }
    
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if (longGauge.reconstruct == QUDA_RECONSTRUCT_NO){
      staggeredDslashNoReconCuda<2>((float2*)out, (float*)outNorm, (float2*)fatGauge0, (float2*)fatGauge1,
				    (float2*)longGauge0, (float2*)longGauge1,
				    longGauge.reconstruct, (float2*)in, (float*)inNorm, parity, dagger, 
				    (float2*)x, (float*)xNorm, k, volume, length, precision);
    }else{
      staggeredDslashCuda<2>((float2*)out, (float*)outNorm, (float2*)fatGauge0, (float2*)fatGauge1,
			     (float4*)longGauge0, (float4*)longGauge1,
			     longGauge.reconstruct, (float2*)in, (float*)inNorm, parity, dagger, 
			     (float2*)x, (float*)xNorm, k, volume, length, precision);
    }
  } else if (precision == QUDA_HALF_PRECISION) {	
    if (longGauge.reconstruct == QUDA_RECONSTRUCT_NO){
      staggeredDslashNoReconCuda<2>((short2*)out, (float*)outNorm, (short2*)fatGauge0, (short2*)fatGauge1,
				    (short2*)longGauge0, (short2*)longGauge1,
				    longGauge.reconstruct, (short2*)in, (float*)inNorm, parity, dagger, 
				    (short2*)x, (float*)xNorm, k, volume, length, precision);
    }else{
      staggeredDslashCuda<2>((short2*)out, (float*)outNorm, (short2*)fatGauge0, (short2*)fatGauge1,
			     (short4*)longGauge0, (short4*)longGauge1,
			     longGauge.reconstruct, (short2*)in, (float*)inNorm, parity, dagger, 
			     (short2*)x, (float*)xNorm, k, volume, length, precision);
    }
  }

  unbindLongGaugeTex(longGauge);
  unbindFatGaugeTex(fatGauge);

  checkCudaError();
#else
  errorQuda("Staggered dslash has not been built");
#endif  

}

void setTwistParam(double &a, double &b, const double &kappa, const double &mu, 
		   const int dagger, const QudaTwistGamma5Type twist) {
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
    a = 2.0 * kappa * mu;
    b = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
    a = -2.0 * kappa * mu;
    b = 1.0 / (1.0 + a*a);
  } else {
    errorQuda("Twist type %d not defined\n", twist);
  }
  if (dagger) a *= -1.0;

}

template <int N, typename spinorFloat>
void twistGamma5Cuda(spinorFloat *out, float *outNorm, const spinorFloat *in, 
		     const float *inNorm, const int dagger, const double &kappa, 
		     const double &mu, const int volume, const int length, 
		     const QudaTwistGamma5Type twist)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  double a=0.0, b=0.0;
  setTwistParam(a, b, kappa, mu, dagger, twist);

  bindSpinorTex<N>(length, in, inNorm);
  twistGamma5Kernel<<<gridDim, blockDim, 0>>> (out, outNorm, a, b);
  unbindSpinorTex<N>(in, inNorm);
}

void twistGamma5Cuda(void *out, void *outNorm, const void *in, const void *inNorm,
		     const int dagger, const double kappa, const double mu, const int volume, 
		     const int length, const QudaPrecision precision, 
		     const QudaTwistGamma5Type twist) {

#ifdef GPU_TWISTED_MASS_DIRAC
  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    twistGamma5Cuda<2>((double2*)out, (float*)outNorm, (double2*)in, 
		       (float*)inNorm, dagger, kappa, mu, volume, length, twist);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    twistGamma5Cuda<4>((float4*)out, (float*)outNorm, (float4*)in, 
    		       (float*)inNorm, dagger, kappa, mu, volume, length, twist);
  } else if (precision == QUDA_HALF_PRECISION) {
    twistGamma5Cuda<4>((short4*)out, (float*)outNorm, (short4*)in,
		       (float*)inNorm, dagger, kappa, mu, volume, length, twist);
  }
  checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC

}

// Twisted mass wrappers
template <int N, typename spinorFloat, typename gaugeFloat>
void twistedMassDslashCuda(spinorFloat *out, float *outNorm, const gaugeFloat gauge0, 
			   const gaugeFloat gauge1, const QudaReconstructType reconstruct, 
			   const spinorFloat *in, const float* inNorm, const int parity, 
			   const int dagger, const spinorFloat *x, const float* xNorm, 
			   const double &kappa, const double &mu, const double &k, 
			   const int volume, const int length)
{

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  int shared_bytes = blockDim.x*SHARED_TMDEG_FLOATS_PER_THREAD*bindSpinorTex<N>(length, in, inNorm, x, xNorm);

  double a=0.0, b=0.0;
  setTwistParam(a, b, kappa, mu, dagger, QUDA_TWIST_GAMMA5_INVERSE);

  if (x==0) { // not xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	twistedMassDslash18Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b);
      } else {
	twistedMassDslash18DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	twistedMassDslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b);
      } else {
	twistedMassDslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b);
      }
    } else {
      if (!dagger) {
	twistedMassDslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b);
      } else {
	twistedMassDslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b);
      }
    }
  } else { // doing xpay
    b *= k;
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	twistedMassDslash18XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b, x, xNorm);
      } else {
	twistedMassDslash18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b, x, xNorm);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	twistedMassDslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b, x, xNorm);
      } else {
	twistedMassDslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b, x, xNorm);
      }
    } else {
      if (!dagger) {
	twistedMassDslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b, x, xNorm);
      } else {
	twistedMassDslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out, outNorm, gauge0, gauge1, in, inNorm, parity, a, b, x, xNorm);
      }
    }
  }
  
  unbindSpinorTex<N>(in, inNorm, x, xNorm);
}

void twistedMassDslashCuda(void *out, void *outNorm, const FullGauge gauge, 
			   const void *in, const void *inNorm, const int parity, const int dagger, 
			   const void *x, const void *xNorm, const double kappa, const double mu, 
			   const double a, const int volume, const int length, 
			   const QudaPrecision precision) {

#ifdef GPU_TWISTED_MASS_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    twistedMassDslashCuda<2>((double2*)out, (float*)outNorm, (double2*)gauge0, (double2*)gauge1, 
			     gauge.reconstruct, (double2*)in, (float*)inNorm, parity, dagger, 
			     (double2*)x, (float*)xNorm, kappa, mu, a, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    twistedMassDslashCuda<4>((float4*)out, (float*)outNorm, (float4*)gauge0, (float4*)gauge1, 
			     gauge.reconstruct, (float4*)in, (float*)inNorm, parity, dagger, 
			     (float4*)x, (float*)xNorm, kappa, mu, a, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    twistedMassDslashCuda<4>((short4*)out, (float*)outNorm, (short4*)gauge0, (short4*)gauge1, 
			     gauge.reconstruct, (short4*)in, (float*)inNorm, parity, dagger, 
			     (short4*)x, (float*)xNorm, kappa, mu, a, volume, length);
  }

  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif

}

//BEGIN NEW


template <int N, typename spinorFloatN>
void twistNDGamma5Cuda(spinorFloatN *out1, float *outNorm1, spinorFloatN *out2, float *outNorm2, 
                     const spinorFloatN *in1,  const float *inNorm1, const spinorFloatN *in2,  const float *inNorm2,  
                     const int dagger,  double &a, double &b, double &c, const int volume, const int length)
{
  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindFlavorSpinorTex<N>(length, in1, inNorm1, in2, inNorm2);
  twistGamma5Kernel<<<gridDim, blockDim, 0>>> (out1, outNorm1, out2, outNorm2, a, b, c); 
  unbindFlavorSpinorTex<N>(in1, inNorm1, in2, inNorm2);
}

void twistNDGamma5Cuda(void *out1, void *outNorm1, void *out2, void *outNorm2, 
                     const void *in1, const void *inNorm1, const void *in2, const void *inNorm2,
		     const int dagger, double a, double b, double c, 
		     const int volume, const int length, const QudaPrecision precision) {

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    twistNDGamma5Cuda<2>((double2*)out1, (float*)outNorm1, (double2*)out2, (float*)outNorm2, 
                       (double2*)in1,  (float*)inNorm1, (double2*)in2,  (float*)inNorm2, 
                       dagger, a, b, c, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    twistNDGamma5Cuda<4>((float4*)out1, (float*)outNorm1, (float4*)out2, (float*)outNorm2, 
                       (float4*)in1, (float*)inNorm1, (float4*)in2, (float*)inNorm2, 
                       dagger, a, b, c, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    twistNDGamma5Cuda<4>((short4*)out1, (float*)outNorm1, (short4*)out2, (float*)outNorm2, 
                       (short4*)in1, (float*)inNorm1, (short4*)in2, (float*)inNorm2, 
                       dagger, a, b, c, volume, length);
  }
  checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_NDEG_TWISTED_MASS_DIRAC

}

// Twisted mass wrappers
template <int N, typename spinorFloat, typename gaugeFloat>
void twistedNDMassDslashCuda(spinorFloat *out1, float *outNorm1, spinorFloat *out2, float *outNorm2, const gaugeFloat gauge0, 
			     const gaugeFloat gauge1, const QudaReconstructType reconstruct, 
			     const spinorFloat *in1, const float* inNorm1, const spinorFloat *in2, const float* inNorm2, const int parity, 
			     const int dagger, const spinorFloat *x1, const float* xNorm1, const spinorFloat *x2, const float* xNorm2,  
			     const double &a, const double &b, const double &c, 
			     const int volume, const int length)
{

  dim3 gridDim(volume/BLOCK_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  
  int shared_bytes = blockDim.x*SHARED_TMNDEG_FLOATS_PER_THREAD*bindFlavorSpinorTex<N>(length, in1, inNorm1, in2, inNorm2, x1, xNorm1, x2, xNorm2);

  if (x1 == 0) { // not xpay
    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	twistedNDMassDslash18Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c);
      } else {
	twistedNDMassDslash18DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	twistedNDMassDslash12Kernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c);
      } else {
	twistedNDMassDslash12DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c);
      }
    } else {
      if (!dagger) {
	twistedNDMassDslash8Kernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c);
      } else {
	twistedNDMassDslash8DaggerKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c);
      }
    }
  } else { // doing xpay

    if (reconstruct == QUDA_RECONSTRUCT_NO) {
      if (!dagger) {
	twistedNDMassDslash18XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c, x1, xNorm1, x2, xNorm2);
      } else {
	twistedNDMassDslash18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c, x1, xNorm1, x2, xNorm2);
      }
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (!dagger) {
	twistedNDMassDslash12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c, x1, xNorm1, x2, xNorm2);
      } else {
	twistedNDMassDslash12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c, x1, xNorm1, x2, xNorm2);
      }
    } else {
      if (!dagger) {
	twistedNDMassDslash8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 	
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c, x1, xNorm1, x2, xNorm2);
      } else {
	twistedNDMassDslash8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  (out1, outNorm1, out2, outNorm2, gauge0, gauge1, in1, inNorm1, in2, inNorm2, parity, a, b, c, x1, xNorm1, x2, xNorm2);
      }
    }
  }
  
  unbindFlavorSpinorTex<N>(in1, inNorm1, in2, inNorm2, x1, xNorm1, x2, xNorm2);
}

void twistedNDMassDslashCuda(void *out1, void *outNorm1, void *out2, void *outNorm2, const FullGauge gauge, 
			     const void *in1, const void *inNorm1, const void *in2, const void *inNorm2, 
			     const int parity, const int dagger, const void *x1, const void *xNorm1, const void *x2, const void *xNorm2, 
			     const double a, const double b, const double c, 
			     const int volume, const int length, const QudaPrecision precision) {

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
  void *gauge0, *gauge1;
  bindGaugeTex(gauge, parity, &gauge0, &gauge1);

  if (precision != gauge.precision)
    errorQuda("Mixing gauge and spinor precision not supported");

  
  if (precision == QUDA_DOUBLE_PRECISION) {
#if (__CUDA_ARCH__ >= 130)
    twistedNDMassDslashCuda<2>((double2*)out1, (float*)outNorm1, (double2*)out2, (float*)outNorm2, (double2*)gauge0, (double2*)gauge1, 
			       gauge.reconstruct, (double2*)in1, (float*)inNorm1, (double2*)in2, (float*)inNorm2, parity, dagger, 
			       (double2*)x1, (float*)xNorm1, (double2*)x2, (float*)xNorm2, a, b, c, volume, length);
#else
    errorQuda("Double precision not supported on this GPU");
#endif
  } else if (precision == QUDA_SINGLE_PRECISION) {
    twistedNDMassDslashCuda<4>((float4*)out1, (float*)outNorm1, (float4*)out2, (float*)outNorm2, (float4*)gauge0, (float4*)gauge1, 
			      gauge.reconstruct, (float4*)in1, (float*)inNorm1, (float4*)in2, (float*)inNorm2, parity, dagger, 
			      (float4*)x1, (float*)xNorm1, (float4*)x2, (float*)xNorm2, a, b, c, volume, length);
  } else if (precision == QUDA_HALF_PRECISION) {
    twistedNDMassDslashCuda<4>((short4*)out1, (float*)outNorm1, (short4*)out2, (float*)outNorm2, (short4*)gauge0, (short4*)gauge1, 
			      gauge.reconstruct, (short4*)in1, (float*)inNorm1, (short4*)in2, (float*)inNorm2, parity, dagger, 
			      (short4*)x1, (float*)xNorm1, (short4*)x2, (float*)xNorm2, a, b, c, volume, length);
  }

  unbindGaugeTex(gauge);

  checkCudaError();
#else
  errorQuda("Twisted mass dslash has not been built");
#endif

}

//END NEW

#if defined(GPU_FATLINK)||defined(GPU_GAUGE_FORCE)|| defined(GPU_FERMION_FORCE)
#include <force_common.h>
#include "force_kernel_common.cu"
#endif

#ifdef GPU_FATLINK
#include "llfat_quda.cu"
#endif

#ifdef GPU_GAUGE_FORCE
#include "gauge_force_quda.cu"
#endif

#ifdef GPU_FERMION_FORCE
#include "fermion_force_quda.cu"
#endif
