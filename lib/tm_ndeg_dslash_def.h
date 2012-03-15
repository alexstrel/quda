// tm_dslash_def.h - Twisted Mass Dslash kernel definitions

// There are currently 36 different variants of the Twisted Mass
// Wilson Dslash kernel, each one characterized by a set of 5 options, 
// where each option can take one of several values (3*2*2*3 = 36).  
// This file is structured so that the C preprocessor loops through all 36
// variants (in a manner resembling a counter), sets the appropriate
// macros, and defines the corresponding functions.
//
// As an example of the function naming conventions, consider
//
// twistedMassDslash12DaggerXpayKernel(float4* out, ...).
//
// This is a twisted mass Dslash^dagger kernel where the result is
// multiplied by "a" and summed with an input vector (Xpay), and the
// gauge matrix is reconstructed from 12 real numbers.  More
// generally, each function name is given by the concatenation of the
// following 4 fields, with "Kernel" at the end:
//
// DD_NAME_F = twistedMassDslash
// DD_RECON_F = 8, 12, 18
// DD_DAG_F = Dagger, [blank]
// DD_XPAY_F = Xpay, [blank]
//
// In addition, the kernels are templated on the precision of the
// fields (double, single, or half).

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_DAG 0
#define DD_XPAY 0
#define DD_RECON 0
#define DD_PREC 0
#endif

// set options for current iteration

#define DD_NAME_F twistedNDMassDslash

#if (DD_DAG==0) // no dagger
#define DD_DAG_F
#else           // dagger
#define DD_DAG_F Dagger
#endif

#if (DD_XPAY==0) // no xpay 
#define DD_XPAY_F 
#if (DD_PREC == 0)
#define DD_PARAM4 const int oddBit, const double a, const double b, const double c
#else
#define DD_PARAM4 const int oddBit, const float a, const float b, const float c
#endif
#else            // xpay
#define DSLASH_XPAY
#define DD_XPAY_F Xpay
#if (DD_PREC == 0)
#define DD_PARAM4 const int oddBit, const double a, const double b, const double c, const double2 *acc1, const float *accNorm1, const double2 *acc2, const float *accNorm2
#elif (DD_PREC == 1) 
#define DD_PARAM4 const int oddBit, const float a, const float b, const float c,  const float4 *acc1, const float *accNorm1, const float4 *acc2, const float *accNorm2
#else
#define DD_PARAM4 const int oddBit, const float a, const float b, const float c,  const short4 *acc1, const float *accNorm1, const short4 *acc2, const float *accNorm2
#endif
#endif

#if (DD_RECON==0) // reconstruct from 8 reals
#define DD_RECON_F 8
#if (DD_PREC==0)
#define DD_PARAM2 const double2 *gauge0, const double2 *gauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_DOUBLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_DOUBLE
#elif (DD_PREC==1)
#define DD_PARAM2 const float4 *gauge0, const float4 *gauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_SINGLE
#else
#define DD_PARAM2 const short4 *gauge0, const short4* gauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
#endif
#elif (DD_RECON==1) // reconstruct from 12 reals
#define DD_RECON_F 12
#if (DD_PREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_DOUBLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_DOUBLE
#define DD_PARAM2 const double2 *gauge0, const double2 *gauge1
#elif (DD_PREC==1)
#define DD_PARAM2 const float4 *gauge0, const float4 *gauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_SINGLE
#else
#define DD_PARAM2 const short4 *gauge0, const short4 *gauge1
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_SINGLE
#endif
#else               // no reconstruct, load all components
#define DD_RECON_F 18
#define GAUGE_FLOAT2
#if (DD_PREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_18_DOUBLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_DOUBLE
#define DD_PARAM2 const double2 *gauge0, const double2 *gauge1
#elif (DD_PREC==1)
#define DD_PARAM2 const float4 *gauge0, const float4 *gauge1 // FIXME for direct reading, really float2
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_18_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_SINGLE
#else
#define DD_PARAM2 const short4 *gauge0, const short4 *gauge1 // FIXME for direct reading, really short2
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_18_SINGLE
#define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_18_SINGLE
#endif
#endif

#if (DD_PREC==0) // double-precision fields

// double-precision gauge field
#define GAUGE0TEX gauge0TexDouble2
#define GAUGE1TEX gauge1TexDouble2
#define GAUGE_FLOAT2

// double-precision spinor fields
#define DD_PARAM1 double2* out1, float *null1_1, double2* out2, float *null1_2
#define DD_PARAM3 const double2* in1, const float *null4_1, const double2* in2, const float *null4_2
#define READ_SPINOR READ_FLAVOR_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_FLAVOR_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_FLAVOR_SPINOR_DOUBLE_DOWN
#define FLAVORSPINORTEX1 flvSpinorTexDouble1
#define FLAVORSPINORTEX2 flvSpinorTexDouble2
#define FLAVORSPINORTEX1NORM 0
#define FLAVORSPINORTEX2NORM 0
#define SPINOR_DOUBLE
#define WRITE_FLAVOR_SPINOR WRITE_FLAVOR_SPINOR_DOUBLE2
#if (DD_XPAY==1)
#define FLAVORACCUMTEX1 flvAccumTexDouble1
#define FLAVORACCUMTEX2 flvAccumTexDouble2
#define FLAVORACCUMTEX1NORM 0
#define FLAVORACCUMTEX2NORM 0
#define READ_ACCUM READ_FLAVOR_ACCUM_DOUBLE
#endif

#elif (DD_PREC==1) // single-precision fields

// single-precision gauge field
#if (DD_RECON_F == 18)
#define GAUGE0TEX gauge0TexSingle2
#define GAUGE1TEX gauge1TexSingle2
#else
#define GAUGE0TEX gauge0TexSingle4
#define GAUGE1TEX gauge1TexSingle4
#endif

// single-precision spinor fields
#define DD_PARAM1 float4* out1, float *null1_1, float4* out2, float *null1_2
#define DD_PARAM3 const float4* in1, const float *null4_1, const float4* in2, const float *null4_2
#define READ_SPINOR READ_FLAVOR_SPINOR_SINGLE
#define READ_SPINOR_UP READ_FLAVOR_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_FLAVOR_SPINOR_SINGLE_DOWN
#define FLAVORSPINORTEX1 flvSpinorTexSingle1
#define FLAVORSPINORTEX2 flvSpinorTexSingle2
#define FLAVORSPINORTEX1NORM 0
#define FLAVORSPINORTEX2NORM 0
#define WRITE_FLAVOR_SPINOR WRITE_FLAVOR_SPINOR_FLOAT4
#if (DD_XPAY==1)
#define FLAVORACCUMTEX1 flvAccumTexSingle1
#define FLAVORACCUMTEX2 flvAccumTexSingle2
#define FLAVORACCUMTEX1NORM 0
#define FLAVORACCUMTEX2NORM 0
#define READ_ACCUM READ_FLAVOR_ACCUM_SINGLE
#endif

#else             // half-precision fields

// half-precision gauge field
#if (DD_RECON_F == 18)
#define GAUGE0TEX gauge0TexHalf2
#define GAUGE1TEX gauge1TexHalf2
#else
#define GAUGE0TEX gauge0TexHalf4
#define GAUGE1TEX gauge1TexHalf4
#endif

// half-precision spinor fields
#define READ_SPINOR READ_FLAVOR_SPINOR_HALF
#define READ_SPINOR_UP READ_FLAVOR_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_FLAVOR_SPINOR_HALF_DOWN
#define FLAVORSPINORTEX1 flvSpinorTexHalf1
#define FLAVORSPINORTEX2 flvSpinorTexHalf2
#define FLAVORSPINORTEX1NORM flvSpinorTexNorm1
#define FLAVORSPINORTEX2NORM flvSpinorTexNorm2
#define DD_PARAM1 short4* out1, float *outNorm1, short4* out2, float *outNorm2
#define DD_PARAM3 const short4* in1, const float *inNorm1, const short4* in2, const float *inNorm2
#define WRITE_FLAVOR_SPINOR WRITE_FLAVOR_SPINOR_SHORT4
#if (DD_XPAY==1)
#define FLAVORACCUMTEX1 flvAccumTexHalf1
#define FLAVORACCUMTEX2 flvAccumTexHalf2
#define FLAVORACCUMTEX1NORM flvAccumTexNorm1
#define FLAVORACCUMTEX2NORM flvAccumTexNorm2
#define READ_ACCUM READ_FLAVOR_ACCUM_HALF
#endif

#endif

// only build double precision if supported
#if !(__CUDA_ARCH__ < 130 && DD_PREC == 0) 

#define DD_CONCAT(n,r,d,x) n ## r ## d ## x ## Kernel
#define DD_FUNC(n,r,d,x) DD_CONCAT(n,r,d,x)

// define the kernel

__global__ void	DD_FUNC(DD_NAME_F, DD_RECON_F, DD_DAG_F, DD_XPAY_F)
     (DD_PARAM1, DD_PARAM2, DD_PARAM3, DD_PARAM4) {

#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
#if DD_DAG
#include "tm_ndeg_dslash_dagger_core.h"
#else
#include "tm_ndeg_dslash_core.h"
#endif
#endif

}

#endif

// clean up

#undef DD_NAME_F
#undef DD_RECON_F
#undef DD_DAG_F
#undef DD_XPAY_F
#undef DD_PARAM1
#undef DD_PARAM2
#undef DD_PARAM3
#undef DD_PARAM4
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_XPAY
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_GAUGE_MATRIX
#undef GAUGE0TEX
#undef GAUGE1TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef FLAVORSPINORTEX1
#undef FLAVORSPINORTEX2
#undef FLAVORSPINORTEX1NORM
#undef FLAVORSPINORTEX2NORM
#undef FLAVORACCUMTEX1
#undef FLAVORACCUMTEX2
#undef FLAVORACCUMTEX1NORM
#undef FLAVORACCUMTEX2NORM
#undef READ_ACCUM
#undef GAUGE_FLOAT2
#undef SPINOR_DOUBLE
#undef WRITE_FLAVOR_SPINOR


// prepare next set of options, or clean up after final iteration

#if (DD_DAG==0)
#undef DD_DAG
#define DD_DAG 1
#else
#undef DD_DAG
#define DD_DAG 0

#if (DD_XPAY==0)
#undef DD_XPAY
#define DD_XPAY 1
#else
#undef DD_XPAY
#define DD_XPAY 0

#if (DD_RECON==0)
#undef DD_RECON
#define DD_RECON 1
#elif (DD_RECON==1)
#undef DD_RECON
#define DD_RECON 2
#else
#undef DD_RECON
#define DD_RECON 0

#if (DD_PREC==0)
#undef DD_PREC
#define DD_PREC 1
#elif (DD_PREC==1)
#undef DD_PREC
#define DD_PREC 2

#else

#undef DD_LOOP
#undef DD_DAG
#undef DD_XPAY
#undef DD_RECON
#undef DD_PREC

#endif // DD_PREC
#endif // DD_RECON
#endif // DD_XPAY
#endif // DD_DAG

#ifdef DD_LOOP
#include "tm_ndeg_dslash_def.h"
#endif
