
#include <stdio.h>
#include <quda_internal.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <read_gauge.h>
#include <gauge_quda.h>
#include <force_common.h>
#include "llfat_quda.h"
#include <face_quda.h>


#define BLOCK_DIM 64

void
llfat_cuda(FullGauge cudaFatLink, FullGauge cudaSiteLink, 
	   FullStaple cudaStaple, FullStaple cudaStaple1,
	   QudaGaugeParam* param, double* act_path_coeff)
{
  int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
  int Vh = volume/2;
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 halfGridDim(Vh/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM , 1, 1);
  
  QudaPrecision prec = cudaSiteLink.precision;
  QudaReconstructType recon = cudaSiteLink.reconstruct;
  
  if( ((param->X[0] % 2 != 0)
       ||(param->X[1] % 2 != 0)
       ||(param->X[2] % 2 != 0)
       ||(param->X[3] % 2 != 0))
      && (recon  == QUDA_RECONSTRUCT_12)){
    errorQuda("12 reconstruct and odd dimensionsize is not supported by link fattening code (yet)\n");
    
  }
      
  int nStream=9;
  cudaStream_t stream[nStream];
  for(int i = 0;i < nStream; i++){
    cudaStreamCreate(&stream[i]);
  }

 
  llfatOneLinkKernel(cudaFatLink, cudaSiteLink,cudaStaple, cudaStaple1,
		     param, act_path_coeff); CUERR;



  llfat_kernel_param_t kparam;
  for(int i=0;i < 4;i++){
     kparam.ghostDim[i] = commDimPartitioned(i);
  }
  int ktype[8] = {
		LLFAT_EXTERIOR_KERNEL_BACK_X, 
		LLFAT_EXTERIOR_KERNEL_FWD_X, 
		LLFAT_EXTERIOR_KERNEL_BACK_Y, 
		LLFAT_EXTERIOR_KERNEL_FWD_Y, 
		LLFAT_EXTERIOR_KERNEL_BACK_Z, 
		LLFAT_EXTERIOR_KERNEL_FWD_Z, 
		LLFAT_EXTERIOR_KERNEL_BACK_T, 
		LLFAT_EXTERIOR_KERNEL_FWD_T, 
  };


  for(int dir = 0;dir < 4; dir++){
    for(int nu = 0; nu < 4; nu++){
      if (nu != dir){

	//start of one call
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  
	  kparam.kernel_type = ktype[2*k];
	  siteComputeGenStapleParityKernel((void*)cudaStaple.even, (void*)cudaStaple.odd,
					   (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					   (void*)cudaFatLink.even, (void*)cudaFatLink.odd,
					   dir, nu,
					   act_path_coeff[2],
					   recon, prec, halfGridDim,
					   kparam, &stream[2*k]); CUERR;	  
	  
	  exchange_gpu_staple_start(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]);  CUERR;
	  
	  kparam.kernel_type = ktype[2*k+1];
	  siteComputeGenStapleParityKernel((void*)cudaStaple.even, (void*)cudaStaple.odd,
					   (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					   (void*)cudaFatLink.even, (void*)cudaFatLink.odd,
					   dir, nu,
					   act_path_coeff[2],
					   recon, prec, halfGridDim,
					   kparam, &stream[2*k+1]); CUERR;
	  exchange_gpu_staple_start(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]);  CUERR;
	}
        kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	siteComputeGenStapleParityKernel((void*)cudaStaple.even, (void*)cudaStaple.odd,
					 (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					 (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					 dir, nu,
					 act_path_coeff[2],
					 recon, prec, halfGridDim, 
					 kparam, &stream[nStream-1]); CUERR;
	
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  exchange_gpu_staple_comms(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]); CUERR;
	  exchange_gpu_staple_comms(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	}	
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  exchange_gpu_staple_wait(param->X, &cudaStaple, k, (int)QUDA_BACKWARDS, &stream[2*k]); CUERR;
	  exchange_gpu_staple_wait(param->X, &cudaStaple, k, (int)QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	}
 	for(int k=3; k >= 0 ;k--){
	  if(!commDimPartitioned(k)) continue;
	  cudaStreamSynchronize(stream[2*k]);
	  cudaStreamSynchronize(stream[2*k+1]);
	}	
	//end

	//start of one call
        kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL,
					  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					  (void*)cudaStaple.even, (void*)cudaStaple.odd,
					  dir, nu, 0,
					  act_path_coeff[5],
					  recon, prec,  halfGridDim, kparam, &stream[nStream-1]); CUERR;

	//end


	for(int rho = 0; rho < 4; rho++){
	  if (rho != dir && rho != nu){

	    //start of one call
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      kparam.kernel_type = ktype[2*k];	    
	      computeGenStapleFieldParityKernel((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						(void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						(void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						(void*)cudaStaple.even, (void*)cudaStaple.odd,
						dir, rho, 1,
						act_path_coeff[3],
						recon, prec, halfGridDim, kparam, &stream[2*k]); CUERR;	      
	      exchange_gpu_staple_start(param->X, &cudaStaple1, k, (int)QUDA_BACKWARDS, &stream[2*k]);  CUERR;
	      kparam.kernel_type = ktype[2*k+1];	    
	      computeGenStapleFieldParityKernel((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						(void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						(void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						(void*)cudaStaple.even, (void*)cudaStaple.odd,
						dir, rho, 1,
						act_path_coeff[3],
						recon, prec, halfGridDim, kparam, &stream[2*k+1]); CUERR;
	      exchange_gpu_staple_start(param->X, &cudaStaple1, k, (int)QUDA_FORWARDS, &stream[2*k+1]);  CUERR;
	    }	    

	    kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
	    computeGenStapleFieldParityKernel((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
					      (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					      (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					      (void*)cudaStaple.even, (void*)cudaStaple.odd,
					      dir, rho, 1,
					      act_path_coeff[3],
					      recon, prec, halfGridDim, kparam, &stream[nStream-1]); CUERR;

#ifdef MULTI_GPU
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      exchange_gpu_staple_comms(param->X, &cudaStaple1, k, (int)QUDA_BACKWARDS, &stream[2*k]); CUERR;
	      exchange_gpu_staple_comms(param->X, &cudaStaple1, k, (int)QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	    }
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      exchange_gpu_staple_wait(param->X, &cudaStaple1, k, QUDA_BACKWARDS, &stream[2*k]); CUERR;
	      exchange_gpu_staple_wait(param->X, &cudaStaple1, k, QUDA_FORWARDS, &stream[2*k+1]); CUERR;
	    }
	    for(int k=3; k >= 0 ;k--){
	      if(!commDimPartitioned(k)) continue;
	      cudaStreamSynchronize(stream[2*k]);
	      cudaStreamSynchronize(stream[2*k+1]);
	    }	
#endif	    
	    //end
	    return ;
	    
	    for(int sig = 0; sig < 4; sig++){
	      if (sig != dir && sig != nu && sig != rho){						
		
		//start of one call
		kparam.kernel_type = LLFAT_INTERIOR_KERNEL;
		computeGenStapleFieldParityKernel((void*)NULL, (void*)NULL, 
						  (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						  (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						  (void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						  dir, sig, 0,
						  act_path_coeff[4],
						  recon, prec, halfGridDim, kparam, &stream[nStream-1]);	 CUERR;

		//end
		
	      }			    
	    }//sig
	  }
	}//rho	
      }
    }//nu
  }//dir
  
  
  cudaThreadSynchronize(); 
  checkCudaError();
  
  for(int i=0;i < nStream; i++){
    cudaStreamDestroy(stream[i]);
  }

  return;
}


void
llfat_cuda_ex(FullGauge cudaFatLink, FullGauge cudaSiteLink, 
	      FullStaple cudaStaple, FullStaple cudaStaple1,
	      QudaGaugeParam* param, double* act_path_coeff)
{
  int volume_ex = (param->X[0]+4)*(param->X[1]+4)*(param->X[2]+4)*(param->X[3]+4);
  int Vh_ex = volume_ex/2;
  dim3 halfGridDim(Vh_ex/BLOCK_DIM,1,1);
  
  QudaPrecision prec = cudaSiteLink.precision;
  QudaReconstructType recon = cudaSiteLink.reconstruct;
  
  if( ((param->X[0] % 2 != 0)
       ||(param->X[1] % 2 != 0)
       ||(param->X[2] % 2 != 0)
       ||(param->X[3] % 2 != 0))
      && (recon  == QUDA_RECONSTRUCT_12)){
    errorQuda("12 reconstruct and odd dimensionsize is not supported by link fattening code (yet)\n");
    
  }
      
  llfatOneLinkKernel_ex(cudaFatLink, cudaSiteLink,cudaStaple, cudaStaple1,
			param, act_path_coeff); CUERR;
  
  
  for(int dir = 0;dir < 4; dir++){
    for(int nu = 0; nu < 4; nu++){
      if (nu != dir){
	
	siteComputeGenStapleParityKernel_ex((void*)cudaStaple.even, (void*)cudaStaple.odd,
					    (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					    (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					    dir, nu,
					    act_path_coeff[2],
					    recon, prec, halfGridDim); 
	
	computeGenStapleFieldParityKernel_ex((void*)NULL, (void*)NULL,
					     (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
					     (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
					     (void*)cudaStaple.even, (void*)cudaStaple.odd,
					     dir, nu, 0,
					     act_path_coeff[5],
					     recon, prec,  halfGridDim);
	
	
	for(int rho = 0; rho < 4; rho++){
	  if (rho != dir && rho != nu){
	    
	    computeGenStapleFieldParityKernel_ex((void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						 (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						 (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						 (void*)cudaStaple.even, (void*)cudaStaple.odd,
						 dir, rho, 1,
						 act_path_coeff[3],
						 recon, prec, halfGridDim);
	    
	    
	    for(int sig = 0; sig < 4; sig++){
	      if (sig != dir && sig != nu && sig != rho){						
		
		computeGenStapleFieldParityKernel_ex((void*)NULL, (void*)NULL, 
						     (void*)cudaSiteLink.even, (void*)cudaSiteLink.odd,
						     (void*)cudaFatLink.even, (void*)cudaFatLink.odd, 
						     (void*)cudaStaple1.even, (void*)cudaStaple1.odd,
						     dir, sig, 0,
						     act_path_coeff[4],
						     recon, prec, halfGridDim);
		
	      }			    
	    }//sig
	  }
	}//rho	
      }
    }//nu
  }//dir
  
  
  cudaThreadSynchronize(); 
  checkCudaError();
  
  return;
}

