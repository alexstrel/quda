#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

DiracTwistedMass::DiracTwistedMass(const DiracParam &param)
  : DiracWilson(param), mu(param.mu), epsilon(param.epsilon)//!NEW
{

}

DiracTwistedMass::DiracTwistedMass(const DiracTwistedMass &dirac) 
  : DiracWilson(dirac), mu(dirac.mu), epsilon(dirac.epsilon)//!NEW
{

}

DiracTwistedMass::~DiracTwistedMass()
{

}

DiracTwistedMass& DiracTwistedMass::operator=(const DiracTwistedMass &dirac)
{
  if (&dirac != this) {
    DiracWilson::operator=(dirac);
  }
  return *this;
}

// Protected method for applying twist
void DiracTwistedMass::twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in,
				    const QudaTwistGamma5Type twistType) const
{
  checkParitySpinor(out, in);
  
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);

  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);

  double flavor_mu = in.twistFlavor * mu;
  
  twistGamma5Cuda(out.v, out.norm, in.v, in.norm, dagger, kappa, flavor_mu, 
  		  in.volume, in.length, in.precision, twistType);
}


// Public method to apply the twist
void DiracTwistedMass::Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
}

void DiracTwistedMass::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  if (in.twistFlavor != out.twistFlavor) 
    errorQuda("Twist flavors %d %d don't match", in.twistFlavor, out.twistFlavor);

  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID) {
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);
  }

  // We can eliminate this temporary at the expense of more kernels (like clover)
  cudaColorSpinorField *tmp=0; // this hack allows for tmp2 to be full or parity field
  if (tmp2) {
    if (tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = &(tmp2->Even());
    else tmp = tmp2;
  }
  bool reset = newTmp(&tmp, in.Even());

  Twist(*tmp, in.Odd());
  DslashXpay(out.Odd(), in.Even(), QUDA_ODD_PARITY, *tmp, -kappa);
  Twist(*tmp, in.Even());
  DslashXpay(out.Even(), in.Odd(), QUDA_EVEN_PARITY, *tmp, -kappa);

  deleteTmp(&tmp, reset);

}

void DiracTwistedMass::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  checkFullSpinor(out, in);
  bool reset = newTmp(&tmp1, in);

  M(*tmp1, in);
  Mdag(out, *tmp1);

  deleteTmp(&tmp1, reset);
}


void DiracTwistedMass::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
			  cudaColorSpinorField &x, cudaColorSpinorField &b, 
			  const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src = &b;
  sol = &x;
}

void DiracTwistedMass::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			      const QudaSolutionType solType) const
{
  // do nothing
}

DiracTwistedMassPC::DiracTwistedMassPC(const DiracParam &param)
  : DiracTwistedMass(param)
{

}

DiracTwistedMassPC::DiracTwistedMassPC(const DiracTwistedMassPC &dirac) 
  : DiracTwistedMass(dirac)
{

}

DiracTwistedMassPC::~DiracTwistedMassPC()
{

}

DiracTwistedMassPC& DiracTwistedMassPC::operator=(const DiracTwistedMassPC &dirac)
{
  if (&dirac != this) {
    DiracTwistedMass::operator=(dirac);
  }
  return *this;
}

// Public method to apply the inverse twist
void DiracTwistedMassPC::TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  twistedApply(out, in, QUDA_TWIST_GAMMA5_INVERSE);
}

// apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
void DiracTwistedMassPC::Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  if (in.twistFlavor != out.twistFlavor) 
    errorQuda("Twist flavors %d %d don't match", in.twistFlavor, out.twistFlavor);
  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);

  if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    double flavor_mu = in.twistFlavor * mu;
    twistedMassDslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 
    			  0, 0, kappa, flavor_mu, 0.0, out.volume, out.length, in.Precision());
    flops += (1320+72)*in.volume;
  } else { // safe to use tmp2 here which may alias in
    bool reset = newTmp(&tmp2, in);

    TwistInv(*tmp2, in);
    DiracWilson::Dslash(out, *tmp2, parity);

    flops += 72*in.volume;

    // if the pointers alias, undo the twist
    if (tmp2->v == in.v) Twist(*tmp2, *tmp2); 

    deleteTmp(&tmp2, reset);
  }

}

// xpay version of the above
void DiracTwistedMassPC::DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
				    const QudaParity parity, const cudaColorSpinorField &x,
				    const double &k) const
{
  if (!initDslash) initDslashConstants(gauge, in.stride, 0);
  checkParitySpinor(in, out);
  checkSpinorAlias(in, out);

  if (in.twistFlavor != out.twistFlavor) 
    errorQuda("Twist flavors %d %d don't match", in.twistFlavor, out.twistFlavor);
  if (in.twistFlavor == QUDA_TWIST_NO || in.twistFlavor == QUDA_TWIST_INVALID)
    errorQuda("Twist flavor not set %d\n", in.twistFlavor);  

  if (!dagger) {
    double flavor_mu = in.twistFlavor * mu;
    twistedMassDslashCuda(out.v, out.norm, gauge, in.v, in.norm, parity, dagger, 
			  x.v, x.norm, kappa, flavor_mu, k, out.volume, out.length, in.Precision());
    flops += (1320+96)*in.volume;
  } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
    bool reset = newTmp(&tmp1, in);

    TwistInv(*tmp1, in);
    DiracWilson::Dslash(out, *tmp1, parity);
    xpayCuda(x, k, out);
    flops += 96*in.volume;

    // if the pointers alias, undo the twist
    if (tmp1->v == in.v) Twist(*tmp1, *tmp1); 

    deleteTmp(&tmp1, reset);
  }

}

void DiracTwistedMassPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  double kappa2 = -kappa*kappa;

  bool reset = newTmp(&tmp1, in);

  if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_ODD_PARITY); // fused kernel
    Twist(out, in);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, out, kappa2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    Dslash(*tmp1, in, QUDA_EVEN_PARITY); // fused kernel
    Twist(out, in);
    DiracWilson::DslashXpay(out, *tmp1, QUDA_ODD_PARITY, out, kappa2);
  } else { // symmetric preconditioning
    if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(*tmp1, in, QUDA_ODD_PARITY);
      DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(*tmp1, in, QUDA_EVEN_PARITY);
      DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
    } else {
      errorQuda("Invalid matpcType");
    }
  }

  deleteTmp(&tmp1, reset);

}

void DiracTwistedMassPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
{
  // need extra temporary because of symmetric preconditioning dagger
  bool reset = newTmp(&tmp2, in);

  M(*tmp2, in);
  Mdag(out, *tmp2);

  deleteTmp(&tmp2, reset);
}

void DiracTwistedMassPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
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
    TwistInv(*src, b.Odd());
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
    TwistInv(*src, *tmp1);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
    src = &(x.Even());
    TwistInv(*src, b.Even());
    DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
    TwistInv(*src, *tmp1);
    sol = &(x.Odd());
  } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // src = b_e + k D_eo A_oo^-1 b_o
    src = &(x.Odd());
    TwistInv(*tmp1, b.Odd()); // safe even when *tmp1 = b.odd
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
    sol = &(x.Even());
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // src = b_o + k D_oe A_ee^-1 b_e
    src = &(x.Even());
    TwistInv(*tmp1, b.Even()); // safe even when *tmp1 = b.even
    DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
    sol = &(x.Odd());
  } else {
    errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
  }

  // here we use final solution to store parity solution and parity source
  // b is now up for grabs if we want

  deleteTmp(&tmp1, reset);
}

void DiracTwistedMassPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
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
    TwistInv(x.Odd(), *tmp1);
  } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	     matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // x_e = A_ee^-1 (b_e + k D_eo x_o)
    DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
    TwistInv(x.Even(), *tmp1);
  } else {
    errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
  }
  
  deleteTmp(&tmp1, reset);
}


//BEGIN FLAVOR DUPLET STUFF

// This is flavor duplet version of the above (single flavor) twistedApply method:
void DiracTwistedMass::twistedApply(cudaColorSpinorField &out1, cudaColorSpinorField &out2, 
				    const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,  
				    const double &a, const double &b, const double &c) const
{
  checkParitySpinor(out1, in1);
  checkParitySpinor(out2, in2);  
  
  if (!initDslash) initDslashConstants(gauge, in1.stride, 0);

  if (in1.twistFlavor != QUDA_TWIST_DUPLET || in2.twistFlavor != QUDA_TWIST_DUPLET)
    errorQuda("This operator must be applied to flavor duplet!\n");
   
  twistNDGamma5Cuda(out1.v, out1.norm, out2.v, out2.norm, in1.v, in1.norm, in2.v, in2.norm, dagger, a, b, c, 
		    in1.volume, in1.length, in1.precision);
}


void DiracTwistedMass::M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const
{
  checkFullSpinor(out1, in1);
  checkFullSpinor(out2, in2);

  if (in1.twistFlavor != QUDA_TWIST_DUPLET || in2.twistFlavor != QUDA_TWIST_DUPLET) {
    errorQuda("Twist flavor incorrect. Must be QUDA_TWIST_DUPLET.\n");
  }

  DslashXpay(out1.Odd(), out2.Odd(), in1.Even(), in2.Even(), QUDA_ODD_PARITY, in1.Odd(), in2.Odd());
  DslashXpay(out1.Even(), out2.Even(), in1.Odd(),  in2.Odd(), QUDA_EVEN_PARITY, in1.Even(), in2.Even());
}

void DiracTwistedMass::MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const
{
  checkFullSpinor(out1, in1);
  checkFullSpinor(out2, in2);
  
  bool reset1 = newTmp(&tmp1, in1);
  bool reset2 = newTmp(&tmp2, in2);  

  M(*tmp1, *tmp2, in1, in2);
  Mdag(out1, out2, *tmp1, *tmp2);

  deleteTmp(&tmp1, reset1);
  deleteTmp(&tmp2, reset2);  
}


void DiracTwistedMass::prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, 
			       cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
			       cudaColorSpinorField &x1, cudaColorSpinorField &x2, 
			       cudaColorSpinorField &b1, cudaColorSpinorField &b2,
			       const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    errorQuda("Preconditioned solution requires a preconditioned solve_type");
  }

  src1 = &b1;
  src2 = &b2;  
  sol1 = &x1;
  sol2 = &x2;  
}


//!For PC dslash 

// apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
// and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
void DiracTwistedMassPC::Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, 
				const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
				const QudaParity parity) const
{
  if (!initDslash) initDslashConstants(gauge, in1.stride, 0);
  checkParitySpinor(in1, out1);
  checkParitySpinor(in2, out2);  

  checkSpinorAlias(in1, out1);
  checkSpinorAlias(in2, out2);
   
  if (in1.twistFlavor != QUDA_TWIST_DUPLET || in2.twistFlavor != QUDA_TWIST_DUPLET) {
    errorQuda("Twist flavor incorrect. Must be QUDA_TWIST_DUPLET.\n");
  }

  double a = 2.0 * kappa * mu;  
  double b = 2.0 * kappa * epsilon;
  
  double d = (1.0 + a*a - b*b);
  if(d <= 0) errorQuda("Invalid twisted mass parameter\n");
  double c = 1.0 / d;
    

  if(!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
  {
    twistedNDMassDslashCuda(out1.v, out1.norm, out2.v, out2.norm, gauge, in1.v, in1.norm, in2.v, in2.norm, parity, dagger, 
    			  0, 0, 0, 0, a, b, c, in1.volume, in1.length, in1.Precision());
    flops += (1320+72+24)*2*in1.volume;			  
  }
  else
  {
    cudaColorSpinorField *dslashTmp1=0; 
    cudaColorSpinorField *dslashTmp2=0;   
  
    bool reset1 = newTmp(&dslashTmp1, in1);
    bool reset2 = newTmp(&dslashTmp2, in2);
    
    a *= -1.0;
    twistedApply(*dslashTmp1, *dslashTmp2, in1, in2, a, b, c);
   
    a = 0.0, b = 0.0, c = 1.0;
    twistedNDMassDslashCuda(out1.v, out1.norm, out2.v, out2.norm, gauge, dslashTmp1->v, dslashTmp1->norm, dslashTmp2->v, dslashTmp2->norm, parity, dagger, 
    			  0, 0, 0, 0, a, b, c, in1.volume, in1.length, in1.Precision());

    flops += (1320+72+24)*2*in1.volume;

    deleteTmp(&dslashTmp1, reset1);
    deleteTmp(&dslashTmp2, reset2);    
  }
}

// xpay version of the above
void DiracTwistedMassPC::DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, 
				    const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
				    const QudaParity parity, const cudaColorSpinorField &x1, const cudaColorSpinorField &x2) const
{
  if (!initDslash) initDslashConstants(gauge, in1.stride, 0);
  checkParitySpinor(in1, out1);
  checkParitySpinor(in2, out2);
  //
  checkSpinorAlias(in1, out1);
  checkSpinorAlias(in2, out2);
  
  if (in1.twistFlavor != QUDA_TWIST_DUPLET || in2.twistFlavor != QUDA_TWIST_DUPLET) {
    errorQuda("Twist flavor incorrect. Must be QUDA_TWIST_DUPLET.\n");
  }
  
  double a = 2.0 * kappa * mu;  
  double b = 2.0 * kappa * epsilon;
  
  double d = (1.0 + a*a - b*b);
  if(d <= 0) errorQuda("Invalid twisted mass parameter\n");
  double c = 1.0 / d;
    
    
  if(!dagger)
  {    
    c *= (-kappa * kappa);
    twistedNDMassDslashCuda(out1.v, out1.norm, out2.v, out2.norm, gauge, in1.v, in1.norm, in2.v, in2.norm, parity, dagger, 
    			  x1.v, x1.norm, x2.v, x2.norm, a, b, c, out1.volume, out1.length, in1.Precision());
    flops += (1320+96+24)*2*in1.volume;			  
  }
  else
  {
    cudaColorSpinorField *dslashTmp1=0; 
    cudaColorSpinorField *dslashTmp2=0;   
  
    bool reset1 = newTmp(&dslashTmp1, in1);
    bool reset2 = newTmp(&dslashTmp2, in2);
    
    a *= -1.0;
    twistedApply(*dslashTmp1, *dslashTmp2, in1, in2, a, b, c);
    
    a = 0.0, b = 0.0, c = (-kappa * kappa);    
    twistedNDMassDslashCuda(out1.v, out1.norm, out2.v, out2.norm, gauge, dslashTmp1->v, dslashTmp1->norm, dslashTmp2->v, dslashTmp2->norm, parity, dagger, 
    			    x1.v, x1.norm, x2.v, x2.norm, a, b, c, in1.volume, in1.length, in1.Precision());

    flops += (1320+96+24)*2*in1.volume;
    
    deleteTmp(&dslashTmp1, reset1);
    deleteTmp(&dslashTmp2, reset2);    
  }
}

void DiracTwistedMassPC::M(cudaColorSpinorField &out1,  cudaColorSpinorField &out2,
			   const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const
{
  bool reset1 = newTmp(&tmp1, in1);
  bool reset2 = newTmp(&tmp2, in2);
  
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    Dslash(*tmp1, *tmp2, in1, in2, QUDA_ODD_PARITY); // fused kernel
    DslashXpay(out1, out2, *tmp1, *tmp2, QUDA_EVEN_PARITY, in1, in2); // safe since out is not read after writing
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	    Dslash(*tmp1, *tmp2, in1, in2, QUDA_EVEN_PARITY); // fused kernel
	    DslashXpay(out1, out2, *tmp1, *tmp2, QUDA_ODD_PARITY, in1, in2);
  } 
  else {// asymmetric preconditioning
	//Parameter for invert twist (note the implemented operator: c*(1 - i *a * gamma_5 tau_3 + b * tau_1)):
	double a = !dagger ? -2.0 * kappa * mu : 2.0 * kappa * mu;  
	double b = -2.0 * kappa * epsilon;
	
        double kappa2 = -kappa*kappa;	
        if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	   Dslash(*tmp1, *tmp2, in1, in2, QUDA_ODD_PARITY); // fused kernel
           twistedApply(out1, out2, in1, in2, a, b, 1.0);	   
           DiracWilson::DslashXpay(out1, *tmp1, QUDA_EVEN_PARITY, out1, kappa2);	   
           DiracWilson::DslashXpay(out2, *tmp2, QUDA_EVEN_PARITY, out2, kappa2);	   	   
	} else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
		  Dslash(*tmp1, *tmp2, in1, in2, QUDA_EVEN_PARITY); // fused kernel
		  twistedApply(out1, out2, in1, in2, a, b, 1.0);	   
		  DiracWilson::DslashXpay(out1, *tmp1, QUDA_ODD_PARITY, out1, kappa2);	   
		  DiracWilson::DslashXpay(out2, *tmp2, QUDA_ODD_PARITY, out2, kappa2);	   	   
	}    
  }
  deleteTmp(&tmp1, reset1);
  deleteTmp(&tmp2, reset2);
}

void DiracTwistedMassPC::MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, 
			       const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const
{
  // need extra temporary 
  cudaColorSpinorField *ftmp1=0; 
  cudaColorSpinorField *ftmp2=0;   
  
  bool reset1 = newTmp(&ftmp1, in1);
  bool reset2 = newTmp(&ftmp2, in2);
 
  M(*ftmp1, *ftmp2,  in1, in2);
  Mdag(out1, out2, *ftmp1, *ftmp2);

  deleteTmp(&ftmp1, reset1);
  deleteTmp(&ftmp2, reset2);
}

void DiracTwistedMassPC::prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, 
				 cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
			         cudaColorSpinorField &x1, cudaColorSpinorField &x2,
				 cudaColorSpinorField &b1, cudaColorSpinorField &b2, 
			         const QudaSolutionType solType) const
{
  // we desire solution to preconditioned system
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    src1 = &b1, src2 = &b2;
    sol1 = &x1, sol2 = &x2;
    return;
  }

  bool reset1 = newTmp(&tmp1, b1.Even());
  bool reset2 = newTmp(&tmp2, b2.Even());  

  double a = 2.0 * kappa * mu;  
  double b = 2.0 * kappa * epsilon;
  
  double d = (1.0 + a*a - b*b);
  if(d <= 0) errorQuda("Invalid twisted mass parameter\n");
  double c = 1.0 / d;
 
  // we desire solution to full system
  if (matpcType == QUDA_MATPC_EVEN_EVEN) {
    // src = A_ee^-1(b_e + k D_eo A_oo^-1 b_o)
    src1 = &(x1.Odd()), src2 = &(x2.Odd());
    twistedApply(*src1, *src2, b1.Odd(), b2.Odd(), a, b, c);
    
    twistedNDMassDslashCuda(tmp1->v, tmp1->norm, tmp2->v, tmp2->norm, gauge, src1->v, src1->norm, src2->v, src2->norm, 
			    QUDA_EVEN_PARITY, dagger, b1.Even().v, b1.Even().norm, b2.Even().v, b2.Even().norm, 
			    0.0, 0.0, kappa, src1->volume, src1->length, src1->Precision());
			  
    twistedApply(*src1, *src2, *tmp1, *tmp2, a, b, c);    
    sol1 = &(x1.Even()), sol2 = &(x2.Even());
        
  } else if (matpcType == QUDA_MATPC_ODD_ODD) {
    // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)    
    src1 = &(x1.Even()), src2 = &(x2.Even());
    twistedApply(*src1, *src2, b1.Even(), b2.Even(), a, b, c);
    
    twistedNDMassDslashCuda(tmp1->v, tmp1->norm, tmp2->v, tmp2->norm, gauge, src1->v, src1->norm, src2->v, src2->norm, 
			    QUDA_ODD_PARITY, dagger, b1.Odd().v, b1.Odd().norm, b2.Odd().v, b2.Odd().norm, 
			    0.0, 0.0, kappa, src1->volume, src1->length, src1->Precision());
    
    
    twistedApply(*src1, *src2, *tmp1, *tmp2, a, b, c);
    
    sol1 = &(x1.Odd()), sol2 = &(x2.Odd());
    
  } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // src = b_e + k D_eo A_oo^-1 b_o
    src1 = &(x1.Odd()), src2 = &(x2.Odd());
    twistedApply(*tmp1, *tmp2, b1.Odd(), b2.Odd(), a, b, c);

    twistedNDMassDslashCuda(src1->v, src1->norm, src2->v, src2->norm, gauge, tmp1->v, tmp1->norm, tmp2->v, tmp2->norm, 
			    QUDA_EVEN_PARITY, dagger, b1.Even().v, b1.Even().norm, b2.Even().v, b2.Even().norm, 
			    0.0, 0.0, kappa, src1->volume, src1->length, src1->Precision());
    sol1 = &(x1.Even()), sol2 = &(x2.Even());
    
  } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // src = b_o + k D_oe A_ee^-1 b_e
    src1 = &(x1.Even()), src2 = &(x2.Even());
    twistedApply(*tmp1, *tmp2, b1.Even(), b2.Even(), a, b, c);
    
    twistedNDMassDslashCuda(src1->v, src1->norm, src2->v, src2->norm, gauge, tmp1->v, tmp1->norm, tmp2->v, tmp2->norm, 
			    QUDA_ODD_PARITY, dagger, b1.Odd().v, b1.Odd().norm, b2.Odd().v, b2.Odd().norm, 
			    0.0, 0.0, kappa, src1->volume, src1->length, src1->Precision());
    sol1 = &(x1.Odd()), sol2 = &(x2.Odd());

  } else {
    errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
  }

  // here we use final solution to store parity solution and parity source
  // b is now up for grabs if we want

  deleteTmp(&tmp1, reset1);
  deleteTmp(&tmp2, reset2);  
}

void DiracTwistedMassPC::reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, 
				         const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
				         const QudaSolutionType solType) const
{
  if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
    return;
  }				

  checkFullSpinor(x1, b1);
  checkFullSpinor(x2, b2);  
  
  bool reset1 = newTmp(&tmp1, b1.Even());
  bool reset2 = newTmp(&tmp2, b2.Even());
  
  // create full solution
  
  double a = 2.0 * kappa * mu;  
  double b = 2.0 * kappa * epsilon;
  
  double d = (1.0 + a*a - b*b);
  if(d <= 0) errorQuda("Invalid twisted mass parameter\n");
  double c = 1.0 / d;
    
  
  if (matpcType == QUDA_MATPC_EVEN_EVEN ||
      matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    // x_o = A_oo^-1 (b_o + k D_oe x_e)
    twistedNDMassDslashCuda(tmp1->v, tmp1->norm, tmp2->v, tmp2->norm, gauge, x1.Even().v, x1.Even().norm, x2.Even().v, x2.Even().norm, 
			    QUDA_ODD_PARITY, dagger, b1.Odd().v, b1.Odd().norm, b2.Odd().v, b2.Odd().norm, 
			    0.0, 0.0, kappa, x1.Even().volume, x1.Even().length, x1.Even().Precision());

    twistedApply(x1.Odd(), x2.Odd(), *tmp1, *tmp2, a, b, c);
  
  } else if (matpcType == QUDA_MATPC_ODD_ODD ||
	     matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    // x_e = A_ee^-1 (b_e + k D_eo x_o)    
    twistedNDMassDslashCuda(tmp1->v, tmp1->norm, tmp2->v, tmp2->norm, gauge, x1.Odd().v, x1.Odd().norm, x2.Odd().v, x2.Odd().norm, 
			    QUDA_EVEN_PARITY, dagger, b1.Even().v, b1.Even().norm, b2.Even().v, b2.Even().norm, 
			    0.0, 0.0, kappa, x1.Odd().volume, x1.Odd().length, x1.Odd().Precision());
  
    twistedApply(x1.Even(), x2.Even(), *tmp1, *tmp2, a, b, c);
  } else {
    errorQuda("MatPCType %d not valid for DiracTwistedMassPC", matpcType);
  }
  
  deleteTmp(&tmp1, reset1);
  deleteTmp(&tmp2, reset2);  
}


//END FLAVOR DUPLET STUFF
