#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

#ifdef __cplusplus
extern "C" {
#endif

  // defined in interface_quda.cpp

  extern FullGauge cudaGaugePrecise;
  extern FullGauge cudaGaugeSloppy;

  extern FullGauge cudaFatLinkPrecise;
  extern FullGauge cudaFatLinkSloppy;

  extern FullGauge cudaLongLinkPrecise;
  extern FullGauge cudaLongLinkSloppy;

  extern FullClover cudaCloverPrecise;
  extern FullClover cudaCloverSloppy;

  extern FullClover cudaCloverInvPrecise;
  extern FullClover cudaCloverInvSloppy;

  // defined in inv_cg_cuda.cpp

  void invertCgCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x,
		    cudaColorSpinorField &b, QudaInvertParam *param);

  // defined in inv_multi_cg_quda.cpp

  int invertMultiShiftCgCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField **x, 
			     cudaColorSpinorField b, QudaInvertParam *param, double *offsets, 
			     int num_offsets, double *residue_sq);
  
  // defined in inv_bicgstab_cuda.cpp

  void invertBiCGstabCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x,
			  cudaColorSpinorField &b, QudaInvertParam *param);

//BEGIN NEW
  // defined in inv_cgs_cuda.cpp
	
  void invertCgsCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &x,
							cudaColorSpinorField &b, QudaInvertParam *param);

//for TM stuff only							
  void invertTMCgCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &xf0, cudaColorSpinorField &xf1,
		    cudaColorSpinorField &bf0, cudaColorSpinorField &bf1, QudaInvertParam *invert_param);
		    
  void invertTMBiCGstabCuda(const DiracMatrix &mat, const DiracMatrix &matSloppy, cudaColorSpinorField &xf0, cudaColorSpinorField &xf1,
			 cudaColorSpinorField &bf0, cudaColorSpinorField &bf1, QudaInvertParam *invert_param);		    
//END NEW
#ifdef __cplusplus
}
#endif

#endif // _INVERT_QUDA_H
