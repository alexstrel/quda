#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>
#include <tune_quda.h>

DiracClover::DiracClover(const DiracParam &param)
  : DiracWilson(param), blockClover(64, 1, 1), gridClover((param.gauge->volumeCB + blockClover.x-1)/blockClover.x, 1, 1),
    clover(*(param.clover))
{

}

DiracClover::DiracClover(const DiracClover &dirac) 
  : DiracWilson(dirac), blockClover(64, 1, 1), gridClover(dirac.gridClover), clover(dirac.clover)
{

}

DiracClover::~DiracClover()
{

}

DiracClover& DiracClover::operator=(const DiracClover &dirac)
{

  if (&dirac != this) {
    DiracWilson::operator=(dirac);
    blockClover = dirac.blockClover;
    gridClover = dirac.gridClover;
    clover = dirac.clover;
  }

  return *this;
}

// Find the best block size parameters for the Dslash and DslashXpay kernels
void DiracClover::Tune(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		       const cudaColorSpinorField &x) {

  DiracWilson::Tune(out, in, x);

  setDslashTuning(QUDA_TUNE_YES);

  { // Tune clover application
    TuneDiracClover cloverTune(*this, out, in);
    cloverTune.Benchmark(blockClover, gridClover);
  }

  setDslashTuning(QUDA_TUNE_NO);
}

void DiracClover::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in,
				    const FullClover &clover) const
{
  Dirac::checkParitySpinor(out, in);

  if (out.Volume() != clover.even.volume) {
    errorQuda("Spinor volume %d doesn't match even clover volume %d",
	      out.Volume(), clover.even.volume);
  }
  if (out.Volume() != clover.odd.volume) {
    errorQuda("Spinor volume %d doesn't match odd clover volume %d",
	      out.Volume(), clover.odd.volume);
  }

}

// Protected method, also used for applying cloverInv
void DiracClover::cloverApply(cudaColorSpinorField &out, const FullClover &clover, 
			      const cudaColorSpinorField &in, const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride);
  if (!initClover) initCloverConstants(clover.even.stride);
  checkParitySpinor(in, out, clover);

  cloverCuda(&out, gauge, clover, &in, parity, blockClover, gridClover);

  flops += 504*in.volume;
}

// Public method to apply the clover term only
void DiracClover::Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const
{
  cloverApply(out, clover, in, parity);
}

// FIXME: create kernel to eliminate tmp
void DiracClover::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  cudaColorSpinorField *tmp=0; // this hack allows for tmp2 to be full or parity field
  if (tmp2) {
    if (tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = &(tmp2->Even());
    else tmp = tmp2;
  }
  bool reset = newTmp(&tmp, in.Even());

  Clover(*tmp, in.Odd(), QUDA_ODD_PARITY);
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, *tmp, -kappa);
  Clover(*tmp, in.Even(), QUDA_EVEN_PARITY);
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, *tmp, -kappa);

  deleteTmp(&tmp, reset);
}

void DiracClover::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);

  bool reset = newTmp(&tmp1, in);
  checkFullSpinor(*tmp1, in);

  M(*tmp1, in);
  Mdag(out, *tmp1);

  deleteTmp(&tmp1, reset);
}

void DiracClover::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src = &b;
  sol = &x;
}

void DiracClover::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType) const
{
  // do nothing
}

DiracCloverPC::DiracCloverPC(const DiracParam &param) : 
  DiracClover(param), cloverInv(*(param.cloverInv))
{
  for (int i=0; i<5; i++) {
    blockDslash[i] = dim3(64, 1, 1);
    blockDslashXpay[i] = dim3(64, 1, 1);
    gridDslash[i] = dim3((param.gauge->volumeCB+blockDslash[i].x-1)/blockDslash[i].x, 1, 1);
    gridDslashXpay[i] = dim3((param.gauge->volumeCB+blockDslashXpay[i].x-1)/blockDslashXpay[i].x, 1, 1);
  }
}

DiracCloverPC::DiracCloverPC(const DiracCloverPC &dirac) : 
  DiracClover(dirac), cloverInv(dirac.clover)
{
  for (int i=0; i<5; i++) {
    blockDslash[i] = dirac.blockDslash[i];
    blockDslashXpay[i] = dirac.blockDslashXpay[i];
    gridDslash[i] = dirac.gridDslash[i];
    gridDslashXpay[i] = dirac.gridDslashXpay[i];
  }
}

DiracCloverPC::~DiracCloverPC()
{

}

DiracCloverPC& DiracCloverPC::operator=(const DiracCloverPC &dirac)
{
  if (&dirac != this) {
    DiracClover::operator=(dirac);
    for (int i=0; i<5; i++) {
      blockDslash[i] = dirac.blockDslash[i];
      blockDslashXpay[i] = dirac.blockDslashXpay[i];
      gridDslash[i] = dirac.gridDslash[i];
      gridDslashXpay[i] = dirac.gridDslashXpay[i];
    }
    cloverInv = dirac.cloverInv;
  }

  return *this;
}

// Find the best block size parameters for the Dslash and DslashXpay kernels
void DiracCloverPC::Tune(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		       const cudaColorSpinorField &x) {
  DiracClover::Tune(out, in, x);

  setDslashTuning(QUDA_TUNE_YES);

  { // Tune Dslash
    TuneDiracCloverDslash dslashTune(*this, out, in);
    dslashTune.Benchmark(blockDslash[0], gridDslash[0]);
    for (int i=0; i<4; i++) 
      if (commDimPartitioned(i)) dslashTune.Benchmark(blockDslash[i+1], gridDslash[i+1]);
  }

  { // Tune DslashXpay
    TuneDiracCloverDslashXpay dslashXpayTune(*this, out, in, x);
    dslashXpayTune.Benchmark(blockDslashXpay[0], gridDslashXpay[0]);
    for (int i=0; i<4; i++) 
      if (commDimPartitioned(i)) dslashXpayTune.Benchmark(blockDslashXpay[i+1], gridDslashXpay[i+1]);
  }

  setDslashTuning(QUDA_TUNE_NO);
}

// Public method
void DiracCloverPC::CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			      const QudaParity parity) const
{
  cloverApply(out, cloverInv, in, parity);
}

// apply hopping term, then clover: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (A_ee^-1 D^dagger_eo) or (A_oo^-1 D^dagger_oe)
// NOTE - this isn't Dslash dagger since order should be reversed!
void DiracCloverPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			   const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride);
  if (!initClover) initCloverConstants(clover.even.stride);
  checkParitySpinor(in, out, cloverInv);
  checkSpinorAlias(in, out);

  setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda

  cloverDslashCuda(&out, gauge, cloverInv, &in, parity, dagger, 0, 0.0, 
		   blockDslash, gridDslash, commDim);

  flops += (1320+504)*in.volume;
}

// xpay version of the above
void DiracCloverPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			       const QudaParity parity, const cudaColorSpinorField &x,
			       const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride);
  if (!initClover) initCloverConstants(clover.even.stride);
  checkParitySpinor(in, out, cloverInv);
  checkSpinorAlias(in, out);

  setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda

  cloverDslashCuda(&out, gauge, cloverInv, &in, parity, dagger, &x, k, 
		   blockDslashXpay, gridDslashXpay, commDim);

  flops += (1320+504+48)*in.volume;
}

// Apply the even-odd preconditioned clover-improved Dirac operator
void DiracCloverPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  double kappa2 = -kappa*kappa;

  // FIXME: For asymmetric, a "DslashCxpay" kernel would improve performance.
  bool reset = newTmp(&tmp1, in);

  if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY);
    Clover(out, in, QUDA_EVEN_PARITY);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, out, kappa2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY);
    Clover(out, in, QUDA_ODD_PARITY);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_ODD_PARITY, out, kappa2);
  } else if (!dagger) { // symmetric preconditioning
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("Invalid matpcType");
    }
  } else { // symmetric preconditioning, dagger
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      CloverInv(out, in, QUDA_EVEN_PARITY); 
      Dslash(*tmp1, out, QUDA_ODD_PARITY);
      DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      CloverInv(out, in, QUDA_ODD_PARITY); 
      Dslash(*tmp1, out, QUDA_EVEN_PARITY);
      DiracWilson::DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
    }
  }
  
  deleteTmp(&tmp1, reset);
}

void DiracCloverPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  // need extra temporary because of symmetric preconditioning dagger
  bool reset = newTmp(&tmp2, in);

  M(*tmp2, in);
  Mdag(out, *tmp2);

  deleteTmp(&tmp2, reset);
}

void DiracCloverPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol, 
			    cudaColorSpinorField &x, cudaColorSpinorField &b, 
			    const QudaSolutionType solType) const
{
  // we desire solution to preconditioned system
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    src = &b;
    sol = &x;
    return;
  }

  bool reset = newTmp(&tmp1, b.Even());
  
  // we desire solution to full system
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
    src = &(x.Odd());
    CloverInv(*src, b.Odd(), QUDA_ODD_PARITY);
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
    CloverInv(*src, *tmp1, QUDA_EVEN_PARITY);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
    src = &(x.Even());
    CloverInv(*src, b.Even(), QUDA_EVEN_PARITY);
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
    CloverInv(*src, *tmp1, QUDA_ODD_PARITY);
    sol = &(x.Odd());
  } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // src = b_e + k D_eo A_oo^-1 b_o
    src = &(x.Odd());
    CloverInv(*tmp1, b.Odd(), QUDA_ODD_PARITY); // safe even when *tmp1 = b.odd
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // src = b_o + k D_oe A_ee^-1 b_e
    src = &(x.Even());
    CloverInv(*tmp1, b.Even(), QUDA_EVEN_PARITY); // safe even when *tmp1 = b.even
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
    sol = &(x.Odd());
  } else {
    errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
  }

  // here we use final solution to store parity solution and parity source
  // b is now up for grabs if we want

  deleteTmp(&tmp1, reset);

}

void DiracCloverPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    return;
  }

  checkFullSpinor(x, b);

  bool reset = newTmp(&tmp1, b.Even());

  // create full solution

  if (matpcType == QUDA_MATPC_EVEN_EVEN ||
      matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // x_o = A_oo^-1 (b_o + k D_oe x_e)
    DiracWilson::DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
    CloverInv(x.Odd(), *tmp1, QUDA_ODD_PARITY);
  } else if (matpcType == QUDA_MATPC_ODD_ODD ||
      matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // x_e = A_ee^-1 (b_e + k D_eo x_o)
    DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    CloverInv(x.Even(), *tmp1, QUDA_EVEN_PARITY);
  } else {
    errorQuda("MatPCType %d not valid for DiracCloverPC", matpcType);
  }

  deleteTmp(&tmp1, reset);

}

