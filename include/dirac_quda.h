#ifndef _DIRAC_QUDA_H
#define _DIRAC_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <dslash_quda.h>

// Params for Dirac operator
class DiracParam {

 public:
  QudaDiracType type;
  double kappa;
  double mass;
  double m5; // used by domain wall only
  MatPCType matpcType;
  DagType dagger;
  FullGauge *gauge;
  FullGauge *fatGauge;  // used by staggered only
  FullGauge *longGauge; // used by staggered only
  FullClover *clover;
  FullClover *cloverInv;
  
  double mu; // used by twisted mass only
  double epsilon; //!NEW

  cudaColorSpinorField *tmp1;
  cudaColorSpinorField *tmp2; // used only by Clover and TM

  QudaVerbosity verbose;

  DiracParam() 
    : type(QUDA_INVALID_DIRAC), kappa(0.0), m5(0.0), matpcType(QUDA_MATPC_INVALID),
    dagger(QUDA_DAG_INVALID), gauge(0), clover(0), cloverInv(0), mu(0.0), epsilon(0.0), 
    tmp1(0), tmp2(0), verbose(QUDA_SILENT)
  {

  }

};

void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);
void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, bool pc);

// forward declarations
class DiracM;
class DiracMdagM;
class DiracMdag;

// Abstract base class
class Dirac {

  friend class DiracM;
  friend class DiracMdagM;
  friend class DiracMdag;

 protected:
  FullGauge &gauge;
  double kappa;
  double mass;
  MatPCType matpcType;
  mutable DagType dagger; // mutable to simplify implementation of Mdag
  mutable unsigned long long flops;
  mutable cudaColorSpinorField *tmp1; // temporary hack
  mutable cudaColorSpinorField *tmp2; // temporary hack
  
  bool newTmp(cudaColorSpinorField **, const cudaColorSpinorField &) const;
  void deleteTmp(cudaColorSpinorField **, const bool &reset) const;

 public:
  Dirac(const DiracParam &param);
  Dirac(const Dirac &dirac);
  virtual ~Dirac();
  Dirac& operator=(const Dirac &dirac);

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  virtual void checkFullSpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  void checkSpinorAlias(const cudaColorSpinorField &, const cudaColorSpinorField &) const;

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const = 0;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x,
			  const double &k) const = 0;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
  void Mdag(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  // required methods to use e-o preconditioning for solving full system
  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const = 0;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const = 0;

			   
//BEGIN NEW
  virtual void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const = 0;
  virtual void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const = 0;
			  
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const = 0;
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const = 0;
  void Mdag(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const;

  // required methods to use e-o preconditioning for solving full system
  
  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const = 0;
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const = 0;			   

//END NEW
  // Dirac operator factory
  static Dirac* create(const DiracParam &param);

  unsigned long long Flops() const { unsigned long long rtn = flops; flops = 0; return rtn; }
};

// Full Wilson
class DiracWilson : public Dirac {

 protected:

 public:
  DiracWilson(const DiracParam &param);
  DiracWilson(const DiracWilson &dirac);
  virtual ~DiracWilson();
  DiracWilson& operator=(const DiracWilson &dirac);

  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
			   
//BEGIN   NEW (TEMP HACK)		   
  virtual void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const {};
  virtual void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const {};
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW			   
};

// Even-odd preconditioned Wilson
class DiracWilsonPC : public DiracWilson {

 private:

 public:
  DiracWilsonPC(const DiracParam &param);
  DiracWilsonPC(const DiracWilsonPC &dirac);
  virtual ~DiracWilsonPC();
  DiracWilsonPC& operator=(const DiracWilsonPC &dirac);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
//BEGIN   NEW (TEMP HACK)		   
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW		   
};

// Full clover
class DiracClover : public DiracWilson {

 protected:
  FullClover &clover;
  void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &, 
			 const FullClover &) const;
  void cloverApply(cudaColorSpinorField &out, const FullClover &clover, const cudaColorSpinorField &in, 
		   const QudaParity parity) const;

 public:
  DiracClover(const DiracParam &param);
  DiracClover(const DiracClover &dirac);
  virtual ~DiracClover();
  DiracClover& operator=(const DiracClover &dirac);

  void Clover(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
//BEGIN   NEW (TEMP HACK)		   
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW			   
};

// Even-odd preconditioned clover
class DiracCloverPC : public DiracClover {

 private:
  FullClover &cloverInv;

 public:
  DiracCloverPC(const DiracParam &param);
  DiracCloverPC(const DiracCloverPC &dirac);
  virtual ~DiracCloverPC();
  DiracCloverPC& operator=(const DiracCloverPC &dirac);

  void CloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const;
  void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
	      const QudaParity parity) const;
  void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
//BEGIN   NEW (TEMP HACK)		   
  virtual void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const {};
  virtual void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const {};
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW		   
};



// Full domain wall 
class DiracDomainWall : public DiracWilson {

 protected:
  double m5;
  double kappa5;

 public:
  DiracDomainWall(const DiracParam &param);
  DiracDomainWall(const DiracDomainWall &dirac);
  virtual ~DiracDomainWall();
  DiracDomainWall& operator=(const DiracDomainWall &dirac);

  void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
	      const QudaParity parity) const;
  void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;

  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
//BEGIN   NEW (TEMP HACK)		   
  virtual void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const {};
  virtual void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const {};
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW			   
};

// 5d Even-odd preconditioned domain wall
class DiracDomainWallPC : public DiracDomainWall {

 private:

 public:
  DiracDomainWallPC(const DiracParam &param);
  DiracDomainWallPC(const DiracDomainWallPC &dirac);
  virtual ~DiracDomainWallPC();
  DiracDomainWallPC& operator=(const DiracDomainWallPC &dirac);

  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
//BEGIN   NEW (TEMP HACK)		   
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1,  const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW		   
};

// Full staggered
class DiracStaggered : public Dirac {

 protected:
    FullGauge *fatGauge;
    FullGauge *longGauge;

 public:
  DiracStaggered(const DiracParam &param);
  DiracStaggered(const DiracStaggered &dirac);
  virtual ~DiracStaggered();
  DiracStaggered& operator=(const DiracStaggered &dirac);

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  
  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
			   
//BEGIN   NEW (TEMP HACK)		   
  virtual void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const {};
  virtual void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const {};
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW			   
};

// Even-odd preconditioned staggered
class DiracStaggeredPC : public Dirac {

 protected:
  FullGauge *fatGauge;
  FullGauge *longGauge;

 public:
  DiracStaggeredPC(const DiracParam &param);
  DiracStaggeredPC(const DiracStaggeredPC &dirac);
  virtual ~DiracStaggeredPC();
  DiracStaggeredPC& operator=(const DiracStaggeredPC &dirac);

  virtual void checkParitySpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
  
  virtual void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  virtual void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
			   
//BEGIN   NEW (TEMP HACK)		   
  virtual void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const {};
  virtual void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const {};
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const {};

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const {};
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1,  const cudaColorSpinorField &b2,
		   const QudaSolutionType) const {};		   
//END	 NEW			   
			   
};

// Full twisted mass
class DiracTwistedMass : public DiracWilson {

 protected:
  double mu;
  double epsilon;//!NEW
  
  void twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		    const QudaTwistGamma5Type twistType) const;
//!NEW:		    
  void twistedApply(cudaColorSpinorField &out1, cudaColorSpinorField &out2, 
		    const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
		    const double &a, const double &b, const double &c) const;
		    

		    
 public:
  DiracTwistedMass(const DiracParam &param);
  DiracTwistedMass(const DiracTwistedMass &dirac);
  virtual ~DiracTwistedMass();
  DiracTwistedMass& operator=(const DiracTwistedMass &dirac);

  void Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  virtual void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  virtual void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
		       cudaColorSpinorField &x, cudaColorSpinorField &b, 
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
			   const QudaSolutionType) const;
			   
//BEGIN NEW			   
  virtual void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const;
  virtual void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const;

  virtual void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
		       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
		       const QudaSolutionType) const;
  virtual void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2, 
			   const QudaSolutionType) const{};//not implemented?			   
//END NEW		   
};

// Even-odd preconditioned twisted mass
class DiracTwistedMassPC : public DiracTwistedMass {

 private:

 public:
  DiracTwistedMassPC(const DiracParam &param);
  DiracTwistedMassPC(const DiracTwistedMassPC &dirac);
  virtual ~DiracTwistedMassPC();
  DiracTwistedMassPC& operator=(const DiracTwistedMassPC &dirac);

  void TwistInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void Dslash(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		      const QudaParity parity) const;
  void DslashXpay(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
			  const QudaParity parity, const cudaColorSpinorField &x, const double &k) const;
  void M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  void MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

  void prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
	       cudaColorSpinorField &x, cudaColorSpinorField &b, 
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
		   const QudaSolutionType) const;
		   
//BEGIN   NEW		   
  void Dslash(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		      const QudaParity parity) const;
  void DslashXpay(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, 
			  const QudaParity parity, const cudaColorSpinorField &x1,  const cudaColorSpinorField &x2) const;
  void M(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const;
  void MdagM(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2) const;

  void prepare(cudaColorSpinorField* &src1, cudaColorSpinorField* &src2, cudaColorSpinorField* &sol1, cudaColorSpinorField* &sol2,
	       cudaColorSpinorField &x1, cudaColorSpinorField &x2, cudaColorSpinorField &b1, cudaColorSpinorField &b2,
	       const QudaSolutionType) const;
  void reconstruct(cudaColorSpinorField &x1, cudaColorSpinorField &x2, const cudaColorSpinorField &b1, const cudaColorSpinorField &b2,  const QudaSolutionType) const;		   
//END	 NEW	   
};

// Functor base class for applying a given Dirac matrix (M, MdagM, etc.)
class DiracMatrix {

 protected:
  const Dirac *dirac;

 public:
  DiracMatrix(const Dirac &d) : dirac(&d) { }
  DiracMatrix(const Dirac *d) : dirac(d) { }
  virtual ~DiracMatrix() = 0;

  virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const = 0;
  virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in,
			  cudaColorSpinorField &tmp) const = 0;
  virtual void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in,
			  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const = 0;
			  
//BEGIN NEW
  virtual void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, const int x) const = 0;
  virtual void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
			  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const = 0; //Tmp# not changed!
//END NEW

  unsigned long long flops() const { return dirac->Flops(); }
};

inline DiracMatrix::~DiracMatrix()
{

}

class DiracM : public DiracMatrix {

 public:
  DiracM(const Dirac &d) : DiracMatrix(d) { }
  DiracM(const Dirac *d) : DiracMatrix(d) { }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    dirac->M(out, in);
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
  {
    dirac->tmp1 = &tmp;
    dirac->M(out, in);
    dirac->tmp1 = NULL;
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->M(out, in);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }
  
//BEGIN NEW
  void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, const int x) const
  {
    dirac->M(out1, out2, in1, in2);
  }

  void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->M(out1, out2, in1, in2);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }

//END NEW
};

class DiracMdagM : public DiracMatrix {

 public:
  DiracMdagM(const Dirac &d) : DiracMatrix(d) { }
  DiracMdagM(const Dirac *d) : DiracMatrix(d) { }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    dirac->MdagM(out, in);
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
  {
    dirac->tmp1 = &tmp;
    dirac->MdagM(out, in);
    dirac->tmp1 = NULL;
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->MdagM(out, in);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }
  
//BEGIN NEW
  void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, const int x) const
  {
    dirac->MdagM(out1, out2, in1, in2);
  }

  void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->MdagM(out1, out2, in1, in2);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }

//END NEW
};

class DiracMdag : public DiracMatrix {

 public:
  DiracMdag(const Dirac &d) : DiracMatrix(d) { }
  DiracMdag(const Dirac *d) : DiracMatrix(d) { }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    dirac->Mdag(out, in);
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, cudaColorSpinorField &tmp) const
  {
    dirac->tmp1 = &tmp;
    dirac->Mdag(out, in);
    dirac->tmp1 = NULL;
  }

  void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, 
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->Mdag(out, in);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }

//BEGIN NEW
  void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2, const int x) const
  {
    dirac->Mdag(out1, out2, in1, in2);
  }

  void operator()(cudaColorSpinorField &out1, cudaColorSpinorField &out2, const cudaColorSpinorField &in1, const cudaColorSpinorField &in2,
		  cudaColorSpinorField &Tmp1, cudaColorSpinorField &Tmp2) const
  {
    dirac->tmp1 = &Tmp1;
    dirac->tmp2 = &Tmp2;
    dirac->Mdag(out1, out2, in1, in2);
    dirac->tmp2 = NULL;
    dirac->tmp1 = NULL;
  }
//END NEW

};

#endif // _DIRAC_QUDA_H
