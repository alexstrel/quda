#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <typeinfo>
#include <quda.h>
#include <gauge_quda.h>
#include <quda_internal.h>
#include <face_quda.h>
#include "misc_helpers.h"
#include <assert.h>
#include <cuda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define ALIGNMENT 4096 

#ifdef MPI_COMMS
#include "face_quda.h"
#endif


#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) / 2.f)
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1))

static double Anisotropy;
static QudaTboundary tBoundary;
static int X[4];
static int faceVolumeCB[4]; // checkboarded face volume
static int volumeCB; // checkboarded volume
extern float fat_link_max;

template <typename Float>
inline short FloatToShort(const Float &a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

template <typename Float>
inline void ShortToFloat(Float &a, const short &b) {
  a = ((Float)b/SCALE_FLOAT-SHIFT_FLOAT);
}

// Routines used to pack the gauge field matrices

template <typename Float>
inline void pack8(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*4*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  for (int j=1; j<4; j++) {
    r[j*V].x = g[2*j+0];
    r[j*V].y = g[2*j+1];
  }
}

template <typename Float>
inline void pack8(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*2*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  r[0].z = g[2];
  r[0].w = g[3];
  r[V].x = g[4];
  r[V].y = g[5];
  r[V].z = g[6];
  r[V].w = g[7];
}

template <typename Float>
inline void pack8(float2 *res, Float *g, int dir, int V) {
  float2 *r = res + dir*4*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  for (int j=1; j<4; j++) {
    r[j*V].x = g[2*j+0];
    r[j*V].y = g[2*j+1];
  }
}

template <typename Float>
inline void pack8(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*2*V;
  r[0].x = FloatToShort(atan2(g[1], g[0]) / M_PI);
  r[0].y = FloatToShort(atan2(g[13], g[12]) / M_PI);
  r[0].z = FloatToShort(g[2]);
  r[0].w = FloatToShort(g[3]);
  r[V].x = FloatToShort(g[4]);
  r[V].y = FloatToShort(g[5]);
  r[V].z = FloatToShort(g[6]);
  r[V].w = FloatToShort(g[7]);
}

template <typename Float>
inline void pack8(short2 *res, Float *g, int dir, int V) {
  short2 *r = res + dir*4*V;
  r[0].x = FloatToShort(atan2(g[1], g[0]) / M_PI);
  r[0].y = FloatToShort(atan2(g[13], g[12]) / M_PI);
  for (int j=1; j<4; j++) {
    r[j*V].x = FloatToShort(g[2*j+0]);
    r[j*V].y = FloatToShort(g[2*j+1]);
  }
}

template <typename Float>
inline void pack12(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = g[j*2+0];
    r[j*V].y = g[j*2+1];
  }
}

template <typename Float>
inline void pack12(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*3*V;
  for (int j=0; j<3; j++) {
    r[j*V].x = g[j*4+0]; 
    r[j*V].y = g[j*4+1];
    r[j*V].z = g[j*4+2]; 
    r[j*V].w = g[j*4+3];
  }
}

template <typename Float>
inline void pack12(float2 *res, Float *g, int dir, int V) {
  float2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = g[j*2+0];
    r[j*V].y = g[j*2+1];
  }
}

template <typename Float>
inline void pack12(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*3*V;
  for (int j=0; j<3; j++) {
    r[j*V].x = FloatToShort(g[j*4+0]); 
    r[j*V].y = FloatToShort(g[j*4+1]);
    r[j*V].z = FloatToShort(g[j*4+2]);
    r[j*V].w = FloatToShort(g[j*4+3]);
  }
}

template <typename Float>
inline void pack12(short2 *res, Float *g, int dir, int V) {
  short2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = FloatToShort(g[j*2+0]);
    r[j*V].y = FloatToShort(g[j*2+1]);
  }
}

template <typename Float>
inline void pack18(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = g[j*2+0]; 
    r[j*V].y = g[j*2+1]; 
  }
}

template <typename Float>
inline void pack18(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*5*V;
  for (int j=0; j<4; j++) {
    r[j*V].x = g[j*4+0]; 
    r[j*V].y = g[j*4+1]; 
    r[j*V].z = g[j*4+2]; 
    r[j*V].w = g[j*4+3]; 
  }
  r[4*V].x = g[16]; 
  r[4*V].y = g[17]; 
  r[4*V].z = 0.0;
  r[4*V].w = 0.0;
}

template <typename Float>
inline void pack18(float2 *res, Float *g, int dir, int V) {
  float2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = g[j*2+0]; 
    r[j*V].y = g[j*2+1]; 
  }
}

template <typename Float>
inline void pack18(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*5*V;
  for (int j=0; j<4; j++) {
    r[j*V].x = FloatToShort(g[j*4+0]); 
    r[j*V].y = FloatToShort(g[j*4+1]); 
    r[j*V].z = FloatToShort(g[j*4+2]); 
    r[j*V].w = FloatToShort(g[j*4+3]); 
  }
  r[4*V].x = FloatToShort(g[16]); 
  r[4*V].y = FloatToShort(g[17]); 
  r[4*V].z = (short)0;
  r[4*V].w = (short)0;
}

template <typename Float>
inline void pack18(short2 *res, Float *g, int dir, int V) 
{
  short2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = FloatToShort(g[j*2+0]); 
    r[j*V].y = FloatToShort(g[j*2+1]); 
  }
}

template<typename Float>
inline void fatlink_short_pack18(short2 *d_gauge, Float *h_gauge, int dir, int V) 
{
  short2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    dg[j*V].x = FloatToShort((h_gauge[j*2+0]/fat_link_max)); 
    dg[j*V].y = FloatToShort((h_gauge[j*2+1]/fat_link_max)); 
  }
}



// a += b*c
template <typename Float>
inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign) {
  a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
  a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
}

// a = conj(b)*c
template <typename Float>
inline void complexDotProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] + b[1]*c[1];
    a[1] = b[0]*c[1] - b[1]*c[0];
}

// a += conj(b) * conj(c)
template <typename Float>
inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign) {
  a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
  a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
}

// a = conj(b)*conj(c)
template <typename Float>
inline void complexConjugateProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = -b[0]*c[1] - b[1]*c[0];
}


// Routines used to unpack the gauge field matrices
template <typename Float>
inline void reconstruct8(Float *mat, int dir, int idx) {
  // First reconstruct first row
  Float row_sum = 0.0;
  row_sum += mat[2]*mat[2];
  row_sum += mat[3]*mat[3];
  row_sum += mat[4]*mat[4];
  row_sum += mat[5]*mat[5];
  Float u0 = (dir < 3 ? Anisotropy : (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
  Float diff = 1.f/(u0*u0) - row_sum;
  Float U00_mag = sqrt(diff >= 0 ? diff : 0.0);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  diff = 1.f/(u0*u0) - column_sum;
  Float U20_mag = sqrt(diff >= 0 ? diff : 0.0);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  Float r_inv2 = 1.0/(u0*row_sum);

  // U11
  Float A[2];
  complexDotProduct(A, mat+0, mat+6);
  complexConjugateProduct(mat+8, mat+12, mat+4);
  accumulateComplexProduct(mat+8, A, mat+2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat+10, mat+12, mat+2);
  accumulateComplexProduct(mat+10, A, mat+4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat+0, mat+12);
  complexConjugateProduct(mat+14, mat+6, mat+4);
  accumulateComplexProduct(mat+14, A, mat+2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat+16, mat+6, mat+2);
  accumulateComplexProduct(mat+16, A, mat+4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
}

template <typename Float>
inline void unpack8(Float *h_gauge, double2 *d_gauge, int dir, int V, int idx) {
  double2 *dg = d_gauge + dir*4*V;
  for (int j=0; j<4; j++) {
    h_gauge[2*j+0] = dg[j*V].x;
    h_gauge[2*j+1] = dg[j*V].y;
  }
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, float4 *d_gauge, int dir, int V, int idx) {
  float4 *dg = d_gauge + dir*2*V;
  h_gauge[0] = dg[0].x;
  h_gauge[1] = dg[0].y;
  h_gauge[2] = dg[0].z;
  h_gauge[3] = dg[0].w;
  h_gauge[4] = dg[V].x;
  h_gauge[5] = dg[V].y;
  h_gauge[6] = dg[V].z;
  h_gauge[7] = dg[V].w;
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, float2 *d_gauge, int dir, int V, int idx) {
  float2 *dg = d_gauge + dir*4*V;
  for (int j=0; j<4; j++) {
    h_gauge[2*j+0] = dg[j*V].x;
    h_gauge[2*j+1] = dg[j*V].y;
  }
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, short4 *d_gauge, int dir, int V, int idx) {
  short4 *dg = d_gauge + dir*2*V;
  ShortToFloat(h_gauge[0], dg[0].x);
  ShortToFloat(h_gauge[1], dg[0].y);
  ShortToFloat(h_gauge[2], dg[0].z);
  ShortToFloat(h_gauge[3], dg[0].w);
  ShortToFloat(h_gauge[4], dg[V].x);
  ShortToFloat(h_gauge[5], dg[V].y);
  ShortToFloat(h_gauge[6], dg[V].z);
  ShortToFloat(h_gauge[7], dg[V].w);
  h_gauge[0] *= M_PI;
  h_gauge[1] *= M_PI;
  reconstruct8(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack8(Float *h_gauge, short2 *d_gauge, int dir, int V, int idx) {
  short2 *dg = d_gauge + dir*4*V;
  for (int j=0; j<4; j++) {
    ShortToFloat(h_gauge[2*j+0], dg[j*V].x);
    ShortToFloat(h_gauge[2*j+1], dg[j*V].y);
  }
  h_gauge[0] *= M_PI;
  h_gauge[1] *= M_PI;
  reconstruct8(h_gauge, dir, idx);
}

// do this using complex numbers (simplifies)
template <typename Float>
inline void reconstruct12(Float *mat, int dir, int idx) {
  Float *u = &mat[0*(3*2)];
  Float *v = &mat[1*(3*2)];
  Float *w = &mat[2*(3*2)];
  w[0] = 0.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0;
  accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
  accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
  accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
  accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
  accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
  accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
  Float u0 = (dir < 3 ? Anisotropy :
	      (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
  w[0]*=u0; w[1]*=u0; w[2]*=u0; w[3]*=u0; w[4]*=u0; w[5]*=u0;
}

template <typename Float>
inline void unpack12(Float *h_gauge, double2 *d_gauge, int dir, int V, int idx) {
  double2 *dg = d_gauge + dir*6*V;
  for (int j=0; j<6; j++) {
    h_gauge[j*2+0] = dg[j*V].x;
    h_gauge[j*2+1] = dg[j*V].y; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, float4 *d_gauge, int dir, int V, int idx) {
  float4 *dg = d_gauge + dir*3*V;
  for (int j=0; j<3; j++) {
    h_gauge[j*4+0] = dg[j*V].x;
    h_gauge[j*4+1] = dg[j*V].y; 
    h_gauge[j*4+2] = dg[j*V].z;
    h_gauge[j*4+3] = dg[j*V].w; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, float2 *d_gauge, int dir, int V, int idx) {
  float2 *dg = d_gauge + dir*6*V;
  for (int j=0; j<6; j++) {
    h_gauge[j*2+0] = dg[j*V].x;
    h_gauge[j*2+1] = dg[j*V].y; 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, short4 *d_gauge, int dir, int V, int idx) {
  short4 *dg = d_gauge + dir*3*V;
  for (int j=0; j<3; j++) {
    ShortToFloat(h_gauge[j*4+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*4+1], dg[j*V].y);
    ShortToFloat(h_gauge[j*4+2], dg[j*V].z);
    ShortToFloat(h_gauge[j*4+3], dg[j*V].w);
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack12(Float *h_gauge, short2 *d_gauge, int dir, int V, int idx) {
  short2 *dg = d_gauge + dir*6*V;
  for (int j=0; j<6; j++) {
    ShortToFloat(h_gauge[j*2+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*2+1], dg[j*V].y); 
  }
  reconstruct12(h_gauge, dir, idx);
}

template <typename Float>
inline void unpack18(Float *h_gauge, double2 *d_gauge, int dir, int V) {
  double2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    h_gauge[j*2+0] = dg[j*V].x; 
    h_gauge[j*2+1] = dg[j*V].y;
  }
}

template <typename Float>
inline void unpack18(Float *h_gauge, float4 *d_gauge, int dir, int V) {
  float4 *dg = d_gauge + dir*5*V;
  for (int j=0; j<4; j++) {
    h_gauge[j*4+0] = dg[j*V].x; 
    h_gauge[j*4+1] = dg[j*V].y;
    h_gauge[j*4+2] = dg[j*V].z; 
    h_gauge[j*4+3] = dg[j*V].w;
  }
  h_gauge[16] = dg[4*V].x; 
  h_gauge[17] = dg[4*V].y;
}

template <typename Float>
inline void unpack18(Float *h_gauge, float2 *d_gauge, int dir, int V) {
  float2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    h_gauge[j*2+0] = dg[j*V].x; 
    h_gauge[j*2+1] = dg[j*V].y;
  }
}

template <typename Float>
inline void unpack18(Float *h_gauge, short4 *d_gauge, int dir, int V) {
  short4 *dg = d_gauge + dir*5*V;
  for (int j=0; j<4; j++) {
    ShortToFloat(h_gauge[j*4+0], dg[j*V].x);
    ShortToFloat(h_gauge[j*4+1], dg[j*V].y);
    ShortToFloat(h_gauge[j*4+2], dg[j*V].z);
    ShortToFloat(h_gauge[j*4+3], dg[j*V].w);
  }
  ShortToFloat(h_gauge[16],dg[4*V].x);
  ShortToFloat(h_gauge[17],dg[4*V].y);

}

template <typename Float>
inline void unpack18(Float *h_gauge, short2 *d_gauge, int dir, int V) {
  short2 *dg = d_gauge + dir*9*V;
  for (int j=0; j<9; j++) {
    ShortToFloat(h_gauge[j*2+0], dg[j*V].x); 
    ShortToFloat(h_gauge[j*2+1], dg[j*V].y);
  }
}

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
// offset = 0 for body
// offset = Vh for face
// voxels = Vh for body
// voxels[i] = face volume[i]
template <typename Float, typename FloatN>
static void packQDPGaugeField(FloatN *d_gauge, Float **h_gauge, int oddBit, 
			      ReconstructType reconstruct, int Vh, int *voxels,
			      int pad, int offset, int nFace, QudaLinkType type) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      for (int i = 0; i < nMat; i++) pack12(d_gauge+offset+i, g+i*18, dir, Vh+pad);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      for (int i = 0; i < nMat; i++) pack8(d_gauge+offset+i, g+i*18, dir, Vh+pad);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      int nMat = nFace*voxels[dir];
      Float *g = h_gauge[dir] + oddBit*nMat*18;
      if(type == QUDA_ASQTAD_FAT_LINKS && typeid(FloatN) == typeid(short2) ){
	  //we know it is half precison with fatlink at this stage
	for (int i = 0; i < nMat; i++) 
	  fatlink_short_pack18((short2*)(d_gauge+offset+i), g+i*18, dir, Vh+pad);
      }else{
	for (int i = 0; i < nMat; i++) pack18(d_gauge+offset+i, g+i*18, dir, Vh+pad);
      }
    }
  }
}

// Assume the gauge field is "QDP" ordered: directions outside of
// space-time, row-column ordering, even-odd space-time
template <typename Float, typename FloatN>
static void unpackQDPGaugeField(Float **h_gauge, FloatN *d_gauge, int oddBit, 
				ReconstructType reconstruct, int V, int pad) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack12(g+i*18, d_gauge+i, dir, V+pad, i);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack8(g+i*18, d_gauge+i, dir, V+pad, i);
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) unpack18(g+i*18, d_gauge+i, dir, V+pad);
    }
  }
}

// transpose and scale the matrix
template <typename Float, typename Float2>
static void transposeScale(Float *gT, Float *g, const Float2 &a) {
  for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
    gT[(ic*3+jc)*2+r] = a*g[(jc*3+ic)*2+r];
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
static void packCPSGaugeField(FloatN *d_gauge, Float *h_gauge, int oddBit, 
			      ReconstructType reconstruct, int V, int pad) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack12(d_gauge+i, gT, dir, V+pad);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack8(d_gauge+i, gT, dir, V+pad);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	transposeScale(gT, g, 1.0 / Anisotropy);
	pack18(d_gauge+i, gT, dir, V+pad);
      }
    }
  }

}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
static void unpackCPSGaugeField(Float *h_gauge, FloatN *d_gauge, int oddBit, 
				ReconstructType reconstruct, int V, int pad) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack12(gT, d_gauge+i, dir, V+pad, i);
	transposeScale(hg, gT, Anisotropy);
      }
    } 
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack8(gT, d_gauge+i, dir, V+pad, i);
	transposeScale(hg, gT, Anisotropy);
      }
    }
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *hg = h_gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	unpack18(gT, d_gauge+i, dir, V+pad);
	transposeScale(hg, gT, Anisotropy);
      }
    }
  }

}

template <typename Float>
void packGhost(Float **cpuLink, Float **cpuGhost, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];

  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };

  for(int dir =0; dir < 4; dir++)
    {
      Float* even_src = cpuLink[dir];
      Float* odd_src = cpuLink[dir] + volumeCB*gaugeSiteSize;

      Float* even_dst;
      Float* odd_dst;
     
     //switching odd and even ghost cpuLink when that dimension size is odd
     //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
        even_dst = cpuGhost[dir];
        odd_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
     }else{
	even_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
        odd_dst = cpuGhost[dir];
     }

      int even_dst_index = 0;
      int odd_dst_index = 0;

      int d;
      int a,b,c;
      for(d = X[dir]- nFace; d < X[dir]; d++){
        for(a = 0; a < A[dir]; a++){
          for(b = 0; b < B[dir]; b++){
            for(c = 0; c < C[dir]; c++){
              int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
              int oddness = (a+b+c+d)%2;
              if (oddness == 0){ //even
                for(int i=0;i < 18;i++){
                  even_dst[18*even_dst_index+i] = even_src[18*index + i];
                }
                even_dst_index++;
              }else{ //odd
                for(int i=0;i < 18;i++){
                  odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
                }
                odd_dst_index++;
              }
            }//c
          }//b
        }//a
      }//d

      assert( even_dst_index == nFace*faceVolumeCB[dir]);
      assert( odd_dst_index == nFace*faceVolumeCB[dir]);
    }

}


template <typename Float>
void packGhostAllLinks(Float **cpuLink, Float **cpuGhostBack,Float**cpuGhostFwd, int dir, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];
  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };
  
  
  for(int ite = 0; ite < 2; ite++){
    //ite == 0: back
    //ite == 1: fwd
    Float** dst;
    if (ite == 0){
      dst = cpuGhostBack;
    }else{
      dst = cpuGhostFwd;
    }
    
    //collect back ghost gauge field
    //for(int dir =0; dir < 4; dir++){
      int d;
      int a,b,c;
      
      //we need copy all 4 links in the same location
      for(int linkdir=0; linkdir < 4; linkdir ++){
	Float* even_src = cpuLink[linkdir];
	Float* odd_src = cpuLink[linkdir] + volumeCB*gaugeSiteSize;

	Float* even_dst;
	Float* odd_dst;
	
	//switching odd and even ghost cpuLink when that dimension size is odd
	//only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
	if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
	  even_dst = dst[dir] + 2*linkdir* nFace *faceVolumeCB[dir]*gaugeSiteSize;	
	  odd_dst = even_dst + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
	}else{
	  odd_dst = dst[dir] + 2*linkdir* nFace *faceVolumeCB[dir]*gaugeSiteSize;
	  even_dst = dst[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
	}

	int even_dst_index = 0;
	int odd_dst_index = 0;
	
	int startd;
	int endd; 
	if(ite == 0){ //back
	  startd = 0; 
	  endd= nFace;
	}else{//fwd
	  startd = X[dir] - nFace;
	  endd =X[dir];
	}
	for(d = startd; d < endd; d++){
	  for(a = 0; a < A[dir]; a++){
	    for(b = 0; b < B[dir]; b++){
	      for(c = 0; c < C[dir]; c++){
		int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
		int oddness = (a+b+c+d)%2;
		if (oddness == 0){ //even
		  for(int i=0;i < 18;i++){
		    even_dst[18*even_dst_index+i] = even_src[18*index + i];
		  }
		  even_dst_index++;
		}else{ //odd
		  for(int i=0;i < 18;i++){
		    odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		  }
		  odd_dst_index++;
		}
	      }//c
	    }//b
	  }//a
	}//d
	assert( even_dst_index == nFace*faceVolumeCB[dir]);
	assert( odd_dst_index == nFace*faceVolumeCB[dir]);	
      }//linkdir
      
    //}//dir
  }//ite
}

void set_dim(int *XX) {

  volumeCB = 1;
  for (int i=0; i<4; i++) {
    X[i] = XX[i];
    volumeCB *= X[i];
    faceVolumeCB[i] = 1;
    for (int j=0; j<4; j++) {
      if (i==j) continue;
      faceVolumeCB[i] *= XX[j];
    }
    faceVolumeCB[i] /= 2;
  }
  volumeCB /= 2;

}

void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhost((double**)cpuLink, (double**)cpuGhost, nFace);
  } else {
    packGhost((float**)cpuLink, (float**)cpuGhost, nFace);
  }

}

void pack_ghost_all_links(void **cpuLink, void **cpuGhostBack, void** cpuGhostFwd, int dir, int nFace, QudaPrecision precision) {
  
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhostAllLinks((double**)cpuLink, (double**)cpuGhostBack, (double**) cpuGhostFwd, dir,  nFace);
  } else {
    packGhostAllLinks((float**)cpuLink, (float**)cpuGhostBack, (float**)cpuGhostFwd, dir, nFace);
  }
  
}




template <typename Float>
void packGhostAllStaples(Float *cpuStaple, Float **cpuGhostBack,Float**cpuGhostFwd, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];
  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };
  
  
  for(int ite = 0; ite < 2; ite++){
    //ite == 0: back
    //ite == 1: fwd
    Float** dst;
    if (ite == 0){
      dst = cpuGhostBack;
    }else{
      dst = cpuGhostFwd;
    }
    
    //collect back ghost staple
    for(int dir =0; dir < 4; dir++){
      int d;
      int a,b,c;
      
      //ther is only one staple in the same location
      for(int linkdir=0; linkdir < 1; linkdir ++){
	Float* even_src = cpuStaple;
	Float* odd_src = cpuStaple + volumeCB*gaugeSiteSize;

	Float* even_dst;
	Float* odd_dst;
	
	//switching odd and even ghost cpuLink when that dimension size is odd
	//only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
	if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
	  even_dst = dst[dir];
	  odd_dst = even_dst + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
	}else{
	  odd_dst = dst[dir];
	  even_dst = dst[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
	}

	int even_dst_index = 0;
	int odd_dst_index = 0;
	
	int startd;
	int endd; 
	if(ite == 0){ //back
	  startd = 0; 
	  endd= nFace;
	}else{//fwd
	  startd = X[dir] - nFace;
	  endd =X[dir];
	}
	for(d = startd; d < endd; d++){
	  for(a = 0; a < A[dir]; a++){
	    for(b = 0; b < B[dir]; b++){
	      for(c = 0; c < C[dir]; c++){
		int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
		int oddness = (a+b+c+d)%2;
		if (oddness == 0){ //even
		  for(int i=0;i < 18;i++){
		    even_dst[18*even_dst_index+i] = even_src[18*index + i];
		  }
		  even_dst_index++;
		}else{ //odd
		  for(int i=0;i < 18;i++){
		    odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
		  }
		  odd_dst_index++;
		}
	      }//c
	    }//b
	  }//a
	}//d
	assert( even_dst_index == nFace*faceVolumeCB[dir]);
	assert( odd_dst_index == nFace*faceVolumeCB[dir]);	
      }//linkdir
      
    }//dir
  }//ite
}


void pack_ghost_all_staples_cpu(void *staple, void **cpuGhostStapleBack, void** cpuGhostStapleFwd, int nFace, QudaPrecision precision) {
  
  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhostAllStaples((double*)staple, (double**)cpuGhostStapleBack, (double**) cpuGhostStapleFwd, nFace);
  } else {
    packGhostAllStaples((float*)staple, (float**)cpuGhostStapleBack, (float**)cpuGhostStapleFwd, nFace);
  }
  
}

void pack_gauge_diag(void* buf, int* X, void** sitelink, int nu, int mu, int dir1, int dir2, QudaPrecision prec)
{
    /*
      nu |          |
         |__________|
            mu 
    *	    
    * nu, mu are the directions we are working on
    * Since we are packing our own data, we need to go to the north-west corner in the diagram
    * i.e. x[nu] = X[nu]-1, x[mu]=0, and looop throught x[dir1],x[dir2]
    * in the remaining two directions (dir1/dir2), dir2 is the slowest changing dim when computing
    * index
    */


  int mul_factor[4]={
    1, X[0], X[1]*X[0], X[2]*X[1]*X[0],
  };

  int even_dst_idx = 0;
  int odd_dst_idx = 0;
  char* dst_even =(char*)buf;
  char* dst_odd = dst_even + (X[dir1]*X[dir2]/2)*gaugeSiteSize*prec;
  char* src_even = (char*)sitelink[nu];
  char* src_odd = src_even + (X[0]*X[1]*X[2]*X[3]/2)*gaugeSiteSize*prec;

  if( (X[nu]+X[mu]) % 2 == 1){
    //oddness will change between me and the diagonal neighbor
    //switch it now
    char* tmp = dst_odd;
    dst_odd = dst_even;
    dst_even = tmp;
  }

  for(int i=0;i < X[dir2]; i++){
    for(int j=0; j < X[dir1]; j++){
      int src_idx = ((X[nu]-1)*mul_factor[nu]+ 0*mul_factor[mu]+i*mul_factor[dir2]+j*mul_factor[dir1])>>1;
      int dst_idx = (i*X[dir1]+j) >> 1; 
      int oddness = ( (X[nu]-1) + 0 + i + j) %2;

      if(oddness==0){
	for(int tmpidx = 0; tmpidx < gaugeSiteSize; tmpidx++){
	  memcpy(&dst_even[(18*even_dst_idx+tmpidx)*prec], &src_even[(18*src_idx + tmpidx)*prec], prec);
	}
	even_dst_idx++;
      }else{
	for(int tmpidx = 0; tmpidx < gaugeSiteSize; tmpidx++){	
	  memcpy(&dst_odd[(18*odd_dst_idx+tmpidx)*prec], &src_odd[(18*src_idx + tmpidx)*prec], prec);
	}
	odd_dst_idx++;
      }//if

    }//for j
  }//for i
      
  if( (even_dst_idx != X[dir1]*X[dir2]/2)|| (odd_dst_idx != X[dir1]*X[dir2]/2)){
    errorQuda("even_dst_idx/odd_dst_idx(%d/%d) does not match the value of X[dir1]*X[dir2]/2 (%d)\n",
	      even_dst_idx, odd_dst_idx, X[dir1]*X[dir2]/2);
  }
  return ;
  

}


static void allocateGaugeField(FullGauge *cudaGauge, ReconstructType reconstruct, QudaPrecision precision) {

  cudaGauge->reconstruct = reconstruct;
  cudaGauge->precision = precision;

  cudaGauge->Nc = 3;

  int floatSize;
  if (precision == QUDA_DOUBLE_PRECISION) floatSize = sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) floatSize = sizeof(float);
  else floatSize = sizeof(float)/2;

  if (cudaGauge->even || cudaGauge->odd){
    errorQuda("Error: even/odd field is not null, probably already allocated(even=%p, odd=%p)\n", cudaGauge->even, cudaGauge->odd);
  }
 
  cudaGauge->bytes = 4*cudaGauge->stride*reconstruct*floatSize;
  if (!cudaGauge->even) {
    if (cudaMalloc((void **)&cudaGauge->even, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating even gauge field");
    }
  }
   
  if (!cudaGauge->odd) {
    if (cudaMalloc((void **)&cudaGauge->odd, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating even odd gauge field");
    }
  }

}


void freeGaugeField(FullGauge *cudaGauge) {
  if (cudaGauge->even) cudaFree(cudaGauge->even);
  if (cudaGauge->odd) cudaFree(cudaGauge->odd);
  cudaGauge->even = NULL;
  cudaGauge->odd = NULL;
}

template <typename Float, typename FloatN>
static void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, 
			   GaugeFieldOrder gauge_order, ReconstructType reconstruct, 
			   int bytes, int Vh, int pad, QudaLinkType type) {
  

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);

    

  if( ! packedEven ) errorQuda( "packedEven is borked\n");
  if( ! packedOdd ) errorQuda( "packedOdd is borked\n");
  if( ! even ) errorQuda( "even is borked\n");
  if( ! odd ) errorQuda( "odd is borked\n");
  if( ! cpuGauge ) errorQuda( "cpuGauge is borked\n");

#ifdef MULTI_GPU
  if (gauge_order != QUDA_QDP_GAUGE_ORDER)
    errorQuda("Only QUDA_QDP_GAUGE_ORDER is supported for multi-gpu\n");
#endif

  //for QUDA_ASQTAD_FAT_LINKS, need to find out the max value
  //fat_link_max will be used in encoding half precision fat link
  if(type == QUDA_ASQTAD_FAT_LINKS){
    for(int dir=0; dir < 4; dir++){
      for(int i=0;i < 2*Vh*gaugeSiteSize; i++){
	Float** tmp = (Float**)cpuGauge;
	if( tmp[dir][i] > fat_link_max ){
	  fat_link_max = tmp[dir][i];
	}
      }
    }
  }
  
  double fat_link_max_double = fat_link_max;
#ifdef MULTI_GPU
  reduceMaxDouble(fat_link_max_double);
#endif
  fat_link_max = fat_link_max_double;

  int voxels[] = {Vh, Vh, Vh, Vh};
  int nFace = 1;

  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    packQDPGaugeField(packedEven, (Float**)cpuGauge, 0, reconstruct, Vh, 
		      voxels, pad, 0, nFace, type);
    packQDPGaugeField(packedOdd,  (Float**)cpuGauge, 1, reconstruct, Vh, 
		      voxels, pad, 0, nFace, type);
  } else if (gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    packCPSGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh, pad);
    packCPSGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh, pad);    
  } else {
    errorQuda("Invalid gauge_order");
  }



#ifdef MULTI_GPU
#if 1
  // three step approach
  // 1. get the links into a contiguous buffer
  // 2. communicate between nodes
  // 3. pack into padded regions

  if (type == QUDA_ASQTAD_LONG_LINKS) nFace = 3;
  else nFace = 1;

  Float *ghostLink[4]; // ghost zone links
  Float *sendLink[4]; // send buffer for communication
  for(int i=0;i < 4;i++){
    ghostLink[i] = new Float[2*nFace*faceVolumeCB[i]*gaugeSiteSize*sizeof(Float)];
    if(ghostLink[i] == NULL) errorQuda("malloc failed for ghostLink[%d]\n", i);
    sendLink[i] = new Float[2*nFace*faceVolumeCB[i]*gaugeSiteSize*sizeof(Float)];
    if(sendLink[i] == NULL) errorQuda("malloc failed for sendLink[%d]\n", i);
  }

  packGhost((Float**)cpuGauge, sendLink, nFace); // pack the ghost zones into a contiguous buffer

  FaceBuffer face(X, 4, 18, nFace, QudaPrecision(sizeof(Float))); // this is the precision of the CPU field
  face.exchangeCpuLink((void**)ghostLink, (void**)sendLink);

  packQDPGaugeField(packedEven, ghostLink, 0, reconstruct, Vh,
		    faceVolumeCB, pad, Vh, nFace, type);
  packQDPGaugeField(packedOdd,  ghostLink, 1, reconstruct, Vh, 
		    faceVolumeCB, pad, Vh, nFace, type);
  
  for(int i=0;i < 4;i++) {
    delete []ghostLink[i];
    delete []sendLink[i];
  }
#else
  // Old QMP T-split code
  QudaPrecision precision = (QudaPrecision) sizeof(even->x);
  int Nvec = sizeof(FloatN)/precision;

  // one step approach
  // transfers into the pads directly
  transferGaugeFaces((void *)packedEven, (void *)(packedEven + Vh), precision, Nvec, reconstruct, Vh, pad);
  transferGaugeFaces((void *)packedOdd, (void *)(packedOdd + Vh), precision, Nvec, reconstruct, Vh, pad);
#endif
#endif
  checkCudaError();
  cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
    
  cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
  
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}


template <typename Float, typename FloatN>
static void retrieveGaugeField(Float *cpuGauge, FloatN *even, FloatN *odd, GaugeFieldOrder gauge_order,
			       ReconstructType reconstruct, int bytes, int Vh, int pad) {

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);
    
  cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);    
    
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    unpackQDPGaugeField((Float**)cpuGauge, packedEven, 0, reconstruct, Vh, pad);
    unpackQDPGaugeField((Float**)cpuGauge, packedOdd, 1, reconstruct, Vh, pad);
  } else if (gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    unpackCPSGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, Vh, pad);
    unpackCPSGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, Vh, pad);
  } else {
    errorQuda("Invalid gauge_order");
  }
    
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}

void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
		      GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
		      Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, QudaLinkType type)
{

  if (cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }

  Anisotropy = anisotropy;
  tBoundary = t_boundary;

  cudaGauge->anisotropy = anisotropy;
  cudaGauge->tadpole_coeff = tadpole_coeff;
  cudaGauge->volumeCB = 1;
  for (int d=0; d<4; d++) {
    cudaGauge->X[d] = XX[d];
    cudaGauge->volumeCB *= XX[d];
    X[d] = XX[d];
  }
  //cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
  cudaGauge->volumeCB /= 2;
  cudaGauge->pad = pad;

  /* test disabled because of staggered pad won't pass
#ifdef MULTI_GPU
  if (pad != cudaGauge->X[0]*cudaGauge->X[1]*cudaGauge->X[2]) {
    errorQuda("Gauge padding must match spatial volume");
  }
#endif
  */

  cudaGauge->stride = cudaGauge->volumeCB + cudaGauge->pad;
  cudaGauge->gauge_fixed = gauge_fixed;
  cudaGauge->t_boundary = t_boundary;
  
  allocateGaugeField(cudaGauge, reconstruct, cuda_prec);

  set_dim(XX);

  if (cuda_prec == QUDA_DOUBLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
		     gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (float*)cpuGauge, 
		     gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
    }

  } else if (cuda_prec == QUDA_SINGLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      } else {
	loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      }
    }

  } else if (cuda_prec == QUDA_HALF_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION){
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      }
    }

  }

}




void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order)
{
  if (cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }

  if (cudaGauge->precision == QUDA_DOUBLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      retrieveGaugeField((double*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      retrieveGaugeField((float*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
    }

  } else if (cudaGauge->precision == QUDA_SINGLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpuGauge, (float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((double*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpuGauge, (float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((float*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    }

  } else if (cudaGauge->precision == QUDA_HALF_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpuGauge, (short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((double*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpuGauge, (short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((float*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    }

  }

}


/********************** Staple code, used by link fattening **************/

#ifdef GPU_FATLINK

static void 
allocateStapleQuda(FullStaple *cudaStaple, QudaPrecision precision) 
{
    cudaStaple->precision = precision;    
    cudaStaple->Nc = 3;
    
    if (precision == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: stape does not support half precision\n");
    }
    
    int elements = 18;
    
    cudaStaple->bytes = cudaStaple->stride*elements*precision;
    
    cudaMalloc((void **)&cudaStaple->even, cudaStaple->bytes);
    cudaMalloc((void **)&cudaStaple->odd, cudaStaple->bytes); 	    
}

void
createStapleQuda(FullStaple* cudaStaple, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
      errorQuda("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
    }
    
    cudaStaple->volume = 1;
    for (int d=0; d<4; d++) {
      cudaStaple->X[d] = param->X[d];
      cudaStaple->volume *= param->X[d];
    }
    //cudaStaple->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaStaple->volume /= 2;    
    cudaStaple->pad = param->staple_pad;
    cudaStaple->stride = cudaStaple->volume + cudaStaple->pad;
    allocateStapleQuda(cudaStaple,  param->cuda_prec);
    
    return;
}


void
freeStapleQuda(FullStaple *cudaStaple) 
{
    if (cudaStaple->even) {
	cudaFree(cudaStaple->even);
    }
    if (cudaStaple->odd) {
	cudaFree(cudaStaple->odd);
    }
    cudaStaple->even = NULL;
    cudaStaple->odd = NULL;
}
void
packGhostStaple(FullStaple* cudaStaple, int dir, int whichway,
		void** fwd_nbr_buf_gpu, void** back_nbr_buf_gpu,
		void** fwd_nbr_buf, void** back_nbr_buf,
		cudaStream_t* stream)
{
  int* X = cudaStaple->X;
  int Vs_x, Vs_y, Vs_z, Vs_t;
  
  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];  
  int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  
  if (dir != 3){ //the code would work for dir=3 as well
    //even and odd ness switch (if necessary) is taken caren of in collectGhostStaple();
    int prec= cudaStaple->precision;
    void* gpu_buf;
    int i =dir;
    if (whichway ==  QUDA_BACKWARDS){
      gpu_buf = back_nbr_buf_gpu[i];
      collectGhostStaple(cudaStaple, gpu_buf, i, whichway, stream);
      cudaMemcpyAsync(back_nbr_buf[i], gpu_buf, Vs[i]*gaugeSiteSize*prec, cudaMemcpyDeviceToHost, *stream);
    }else{//whichway is  QUDA_FORWARDS;
      gpu_buf = fwd_nbr_buf_gpu[i];
      collectGhostStaple(cudaStaple, gpu_buf, i, whichway, stream);
      cudaMemcpyAsync(fwd_nbr_buf[i], gpu_buf, Vs[i]*gaugeSiteSize*prec, cudaMemcpyDeviceToHost, *stream);        
    }
  }else{ //special case for dir=3 since no gather kernel is required
    void* even = cudaStaple->even;
    void* odd = cudaStaple->odd;
    int Vh = cudaStaple->volume;
    int Vsh = cudaStaple->X[0]*cudaStaple->X[1]*cudaStaple->X[2]/2;
    int prec= cudaStaple->precision;
    int sizeOfFloatN = 2*prec;
    int len = Vsh*sizeOfFloatN;
    int i;
    if(cudaStaple->X[3] %2 == 0){
      //back,even
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + i*len ; 
	void* src = ((char*)even) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //back, odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)odd) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd,even
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + i*len ; 
	void* src = ((char*)even) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd, odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)odd) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
    }else{
      //reverse even and odd position
      //back,odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + i*len ; 
	void* src = ((char*)odd) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //back, even
      for(i=0;i < 9; i++){
	void* dst = ((char*)back_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)even) + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd,odd
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + i*len ; 
	void* src = ((char*)odd) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      //fwd, even
      for(i=0;i < 9; i++){
	void* dst = ((char*)fwd_nbr_buf[3]) + 9*len + i*len ; 
	void* src = ((char*)even) + (Vh-Vsh)*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
	cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
      }
      
    } 
  }
  
}


void 
unpackGhostStaple(FullStaple* cudaStaple, int dir, int whichway, void** fwd_nbr_buf, void** back_nbr_buf,
		  cudaStream_t* stream)
{

  int* X = cudaStaple->X;
  int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
  
  Vsh_x = X[1]*X[2]*X[3]/2;
  Vsh_y = X[0]*X[2]*X[3]/2;
  Vsh_z = X[0]*X[1]*X[3]/2;
  Vsh_t = X[0]*X[1]*X[2]/2;  
  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};

  char* even = (char*)cudaStaple->even;
  char* odd = (char*)cudaStaple->odd;
  int Vh = cudaStaple->volume;
  //int Vsh = cudaStaple->X[0]*cudaStaple->X[1]*cudaStaple->X[2]/2;
  int prec= cudaStaple->precision;
  int sizeOfFloatN = 2*prec;
  int len[4] = {
    Vsh_x*sizeOfFloatN,
    Vsh_y*sizeOfFloatN,
    Vsh_z*sizeOfFloatN,
    Vsh_t*sizeOfFloatN 
  };
  
  int tmpint[4] = {
    0,
    Vsh_x, 
    Vsh_x + Vsh_y, 
    Vsh_x + Vsh_y + Vsh_z, 
  };
  
  even = ((char*)cudaStaple->even) + Vh*sizeOfFloatN + 2*tmpint[dir]*sizeOfFloatN;
  odd = ((char*)cudaStaple->odd) + Vh*sizeOfFloatN +2*tmpint[dir]*sizeOfFloatN;
  
  if(whichway == QUDA_BACKWARDS){   
    //back,even
    for(int i=0;i < 9; i++){
      void* dst = even + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)back_nbr_buf[dir]) + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
    //back, odd
    for(int i=0;i < 9; i++){
      void* dst = odd + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)back_nbr_buf[dir]) + 9*len[dir] + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
  }else { //QUDA_FORWARDS
    //fwd,even
    for(int i=0;i < 9; i++){
      void* dst = even + Vsh[dir]*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)fwd_nbr_buf[dir]) + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
    //fwd, odd
    for(int i=0;i < 9; i++){
      void* dst = odd + Vsh[dir]*sizeOfFloatN + i*cudaStaple->stride*sizeOfFloatN;
      void* src = ((char*)fwd_nbr_buf[dir]) + 9*len[dir] + i*len[dir] ; 
      cudaMemcpyAsync(dst, src, len[dir], cudaMemcpyHostToDevice, *stream); CUERR;
    }
  }
}
#endif


/******************************** Mom code, used by Fermi force code ****************************/

#ifdef GPU_FERMION_FORCE
static void 
allocateMomQuda(FullMom *cudaMom, QudaPrecision precision) 
{
    cudaMom->precision = precision;    
    
    if (precision == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: stape does not support half precision\n");
    }
    
    int elements = 10;
     
    cudaMom->bytes = cudaMom->volume*elements*precision*4;
    
    cudaMalloc((void **)&cudaMom->even, cudaMom->bytes);
    cudaMalloc((void **)&cudaMom->odd, cudaMom->bytes); 	    
}

void
createMomQuda(FullMom* cudaMom, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
      errorQuda("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
      errorQuda("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
    }
    
    cudaMom->volume = 1;
    for (int d=0; d<4; d++) {
      cudaMom->X[d] = param->X[d];
      cudaMom->volume *= param->X[d];
    }
    cudaMom->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaMom->volume /= 2;    
    
    allocateMomQuda(cudaMom,  param->cuda_prec);
    
    return;
}


void
freeMomQuda(FullMom *cudaMom) 
{
    if (cudaMom->even) {
	cudaFree(cudaMom->even);
    }
    if (cudaMom->odd) {
	cudaFree(cudaMom->odd);
    }
    cudaMom->even = NULL;
    cudaMom->odd = NULL;
}

template <typename Float, typename Float2>
inline void pack10(Float2 *res, Float *m, int dir, int Vh) 
{
    Float2 *r = res + dir*5*Vh;
    for (int j=0; j<5; j++) {
	r[j*Vh].x = (float)m[j*2+0]; 
	r[j*Vh].y = (float)m[j*2+1]; 
    }
}

template <typename Float, typename Float2>
void packMomField(Float2 *res, Float *mom, int oddBit, int Vh) 
{    
    for (int dir = 0; dir < 4; dir++) {
	Float *g = mom + (oddBit*Vh*4 + dir)*momSiteSize;
	for (int i = 0; i < Vh; i++) {
	    pack10(res+i, g + 4*i*momSiteSize, dir, Vh);
	}
    }      
}

template <typename Float, typename Float2>
void loadMomField(Float2 *even, Float2 *odd, Float *mom,
		  int bytes, int Vh) 
{
    
    Float2 *packedEven, *packedOdd;
    cudaMallocHost((void**)&packedEven, bytes); 
    cudaMallocHost((void**)&packedOdd, bytes);
    
    packMomField(packedEven, (Float*)mom, 0, Vh);
    packMomField(packedOdd,  (Float*)mom, 1, Vh);
    
    cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice); 
    
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
}




void
loadMomToGPU(FullMom cudaMom, void* mom, QudaGaugeParam* param)
{
    if (param->cuda_prec == QUDA_DOUBLE_PRECISION) {
	//loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
	//cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    } else { //single precision
	loadMomField((float2*)(cudaMom.even), (float2*)(cudaMom.odd), (float*)mom, 
		     cudaMom.bytes, cudaMom.volume);
	
    }
}


template <typename Float, typename Float2>
inline void unpack10(Float* m, Float2 *res, int dir, int Vh) 
{
    Float2 *r = res + dir*5*Vh;
    for (int j=0; j<5; j++) {
	m[j*2+0] = r[j*Vh].x;
	m[j*2+1] = r[j*Vh].y;
    }
    
}

template <typename Float, typename Float2>
void 
unpackMomField(Float* mom, Float2 *res, int oddBit, int Vh) 
{
    int dir, i;
    Float *m = mom + oddBit*Vh*momSiteSize*4;
    
    for (i = 0; i < Vh; i++) {
	for (dir = 0; dir < 4; dir++) {	
	    Float* thismom = m + (4*i+dir)*momSiteSize;
	    unpack10(thismom, res+i, dir, Vh);
	}
    }
}

template <typename Float, typename Float2>
void 
storeMomToCPUArray(Float* mom, Float2 *even, Float2 *odd, 
		   int bytes, int Vh) 
{    
    Float2 *packedEven, *packedOdd;   
    cudaMallocHost((void**)&packedEven, bytes); 
    cudaMallocHost((void**)&packedOdd, bytes); 
    cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);  

    unpackMomField((Float*)mom, packedEven,0, Vh);
    unpackMomField((Float*)mom, packedOdd, 1, Vh);
        
    cudaFreeHost(packedEven); 
    cudaFreeHost(packedOdd); 
}

void 
storeMomToCPU(void* mom, FullMom cudaMom, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec != cuda_prec){
	printf("Error:%s: cpu and gpu precison has to be the same at this moment \n", __FUNCTION__);
	exit(1);	
    }
    
    if (cpu_prec == QUDA_HALF_PRECISION){
	printf("ERROR: %s:  half precision is not supported at this moment\n", __FUNCTION__);
	exit(1);
    }
    
    if (cpu_prec == QUDA_DOUBLE_PRECISION){
	
    }else { //SINGLE PRECISIONS
	storeMomToCPUArray( (float*)mom, (float2*) cudaMom.even, (float2*)cudaMom.odd, 
			    cudaMom.bytes, cudaMom.volume);	
    }
    
}
#endif

/********** link code, used by link fattening  **********************/

#ifdef GPU_FATLINK

void
createLinkQuda(FullGauge* cudaGauge, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
	printf("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
	exit(-1);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
	exit(-1);
    }
        
    cudaGauge->anisotropy = param->anisotropy;
    cudaGauge->volumeCB = 1;
    for (int d=0; d<4; d++) {
      cudaGauge->X[d] = param->X[d];
	cudaGauge->volumeCB *= param->X[d];
    }
    //cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaGauge->volumeCB /= 2;    
    cudaGauge->pad = param->ga_pad;
    cudaGauge->stride = cudaGauge->volumeCB + cudaGauge->pad;
    cudaGauge->reconstruct = param->reconstruct;

    allocateGaugeField(cudaGauge, param->reconstruct, param->cuda_prec);
    
    return;
}


template<typename FloatN, typename Float>
static void 
do_loadLinkToGPU(int* X, FloatN *even, FloatN *odd, Float **cpuGauge, Float** ghost_cpuGauge,
		 Float** ghost_cpuGauge_diag, 
		 ReconstructType reconstruct, int bytes, int Vh, int pad, 
		 int Vsh_x, int Vsh_y, int Vsh_z, int Vsh_t,
		 QudaPrecision prec) 
{
  cudaStream_t streams[2];
  for(int i=0;i < 2; i++){
    cudaStreamCreate(&streams[i]);
  }

  
  int Vh_2d_max = MAX(X[0]*X[1]/2, X[0]*X[2]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[0]*X[3]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[1]*X[2]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[1]*X[3]/2);
  Vh_2d_max = MAX(Vh_2d_max, X[2]*X[3]/2);

  int i;
  char* tmp_even;
  char* tmp_odd;
  int len = Vh*gaugeSiteSize*sizeof(Float);

#ifdef MULTI_GPU    
  int glen[4] = {
    Vsh_x*gaugeSiteSize*sizeof(Float),
    Vsh_y*gaugeSiteSize*sizeof(Float),
    Vsh_z*gaugeSiteSize*sizeof(Float),
    Vsh_t*gaugeSiteSize*sizeof(Float)
  };
  
  int ghostV = 2*(Vsh_x+Vsh_y+Vsh_z+Vsh_t)+4*Vh_2d_max;
#else
  int glen[4] = {0,0,0,0};
#endif  

  int glen_sum = ghostV*gaugeSiteSize*sizeof(Float);
  cudaMalloc(&tmp_even, 4*(len+glen_sum)); CUERR;
  cudaMalloc(&tmp_odd, 4*(len+glen_sum)); CUERR;
  
  //even links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_even + i*(len+glen_sum), cpuGauge[i], len, cudaMemcpyHostToDevice, streams[0]); 
#else
    cudaMemcpy(tmp_even + i*(len+glen_sum), cpuGauge[i], len, cudaMemcpyHostToDevice); 
#endif
  
#ifdef MULTI_GPU  
    //dir: the source direction
    char* dest = tmp_even + i*(len+glen_sum)+len;
    for(int dir = 0; dir < 4; dir++){
#if (CUDA_VERSION >=4000)
      cudaMemcpyAsync(dest, ((char*)ghost_cpuGauge[dir])+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice, streams[0]); 
      cudaMemcpyAsync(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice, streams[0]); 	
#else
      cudaMemcpy(dest, ((char*)ghost_cpuGauge[dir])+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); 
      cudaMemcpy(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); 
#endif
      dest += 2*glen[dir];
  }
    //fill in diag 
    //@nu is @i, mu iterats from 0 to 4 and mu != nu
    int nu = i;
    for(int mu = 0; mu < 4; mu++){
      if(nu  == mu ){
	continue;
      }
      int dir1, dir2;
      for(dir1=0; dir1 < 4; dir1 ++){
	if(dir1 != nu && dir1 != mu){
	  break;
	}
      }
      for(dir2=0; dir2 < 4; dir2 ++){
	if(dir2 != nu && dir2 != mu && dir2 != dir1){
	  break;
	}
      }
#if (CUDA_VERSION >=4000)
      cudaMemcpyAsync(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu], 
		      X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice, streams[0]);	
#else	
      cudaMemcpy(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu], 
		 X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice);	
#endif
      
    }
    
#endif
  }    
  
  link_format_cpu_to_gpu((void*)even, (void*)tmp_even,  reconstruct, bytes, Vh, pad, ghostV, prec, streams[0]); CUERR;

  //odd links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_odd + i*(len+glen_sum), cpuGauge[i] + Vh*gaugeSiteSize, len, cudaMemcpyHostToDevice, streams[1]);CUERR;
#else
    cudaMemcpy(tmp_odd + i*(len+glen_sum), cpuGauge[i] + Vh*gaugeSiteSize, len, cudaMemcpyHostToDevice);CUERR;
#endif

#ifdef MULTI_GPU  
      char* dest = tmp_odd + i*(len+glen_sum)+len;
      for(int dir = 0; dir < 4; dir++){
#if (CUDA_VERSION >=4000)
	cudaMemcpyAsync(dest, ((char*)ghost_cpuGauge[dir])+glen[dir] +i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice, streams[1]); CUERR;
	cudaMemcpyAsync(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+glen[dir] +i*2*glen[dir], glen[dir], 
			cudaMemcpyHostToDevice, streams[1]); CUERR;
#else
	cudaMemcpy(dest, ((char*)ghost_cpuGauge[dir])+glen[dir] +i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); CUERR;
	cudaMemcpy(dest + glen[dir], ((char*)ghost_cpuGauge[dir])+8*glen[dir]+glen[dir] +i*2*glen[dir], glen[dir], cudaMemcpyHostToDevice); CUERR;

#endif

	dest += 2*glen[dir];
      }
      //fill in diag 
      //@nu is @i, mu iterats from 0 to 4 and mu != nu
      int nu = i;
      for(int mu = 0; mu < 4; mu++){
	if(nu  == mu ){
	  continue;
	}
	int dir1, dir2;
	for(dir1=0; dir1 < 4; dir1 ++){
	  if(dir1 != nu && dir1 != mu){
	    break;
	  }
	}
	for(dir2=0; dir2 < 4; dir2 ++){
	  if(dir2 != nu && dir2 != mu && dir2 != dir1){
	    break;
	  }
	}
#if (CUDA_VERSION >=4000)
	cudaMemcpyAsync(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu]+X[dir1]*X[dir2]/2*gaugeSiteSize, 
			X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice, streams[1]);	
#else
	cudaMemcpy(dest+ mu *Vh_2d_max*gaugeSiteSize*sizeof(Float),ghost_cpuGauge_diag[nu*4+mu]+X[dir1]*X[dir2]/2*gaugeSiteSize, 
		   X[dir1]*X[dir2]/2*gaugeSiteSize*sizeof(Float), cudaMemcpyHostToDevice );		
#endif
      }
      

#endif
  }
  link_format_cpu_to_gpu((void*)odd, (void*)tmp_odd, reconstruct, bytes, Vh, pad, ghostV, prec, streams[1]); CUERR;
  
  for(int i=0;i < 2;i++){
    cudaStreamSynchronize(streams[i]);
  }

  cudaFree(tmp_even);
  cudaFree(tmp_odd);

  for(int i=0;i < 2;i++){
    cudaStreamDestroy(streams[i]);
  }
  CUERR;
}


void 
loadLinkToGPU(FullGauge cudaGauge, void **cpuGauge, QudaGaugeParam* param)
{

  if (param->cpu_prec  != param->cuda_prec){
    printf("ERROR: cpu precision and cuda precision must be the same in this function %s\n", __FUNCTION__);
    exit(1);
  }
  QudaPrecision prec= param->cpu_prec;

  int* Z = param->X;
  int pad = param->ga_pad;
  int Vsh_x = param->X[1]*param->X[2]*param->X[3]/2;
  int Vsh_y = param->X[0]*param->X[2]*param->X[3]/2;
  int Vsh_z = param->X[0]*param->X[1]*param->X[3]/2;
  int Vsh_t = param->X[0]*param->X[1]*param->X[2]/2;

  int Vs[4] = {2*Vsh_x, 2*Vsh_y, 2*Vsh_z, 2*Vsh_t};

  void* ghost_cpuGauge[4];
  void* ghost_cpuGauge_diag[16];

  for(int i=0;i < 4; i++){
    
#if (CUDA_VERSION >= 4000)
    cudaMallocHost((void**)&ghost_cpuGauge[i], 8*Vs[i]*gaugeSiteSize*prec);
#else
    ghost_cpuGauge[i] = malloc(8*Vs[i]*gaugeSiteSize*prec);
#endif
    if(ghost_cpuGauge[i] == NULL){
      errorQuda("ERROR: malloc failed for ghost_sitelink[%d] \n",i);
    }
  }

  /*
    nu |     |
       |_____|
          mu     
  */

  int ghost_diag_len[16];
  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
      if(nu == mu){
        ghost_cpuGauge_diag[nu*4+mu] = NULL;
      }else{
        //the other directions
        int dir1, dir2;
        for(dir1= 0; dir1 < 4; dir1++){
          if(dir1 !=nu && dir1 != mu){
            break;
          }
        }
        for(dir2=0; dir2 < 4; dir2++){
          if(dir2 != nu && dir2 != mu && dir2 != dir1){
            break;
          }
        }
	//int rc = posix_memalign((void**)&ghost_cpuGauge_diag[nu*4+mu], ALIGNMENT, Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
#if (CUDA_VERSION >= 4000)
	cudaMallocHost((void**)&ghost_cpuGauge_diag[nu*4+mu],  Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
#else
	ghost_cpuGauge_diag[nu*4+mu] = malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
#endif
	if(ghost_cpuGauge_diag[nu*4+mu] == NULL){
          errorQuda("malloc failed for ghost_sitelink_diag\n");
        }
	
        memset(ghost_cpuGauge_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*prec);
	ghost_diag_len[nu*4+mu] = Z[dir1]*Z[dir2]*gaugeSiteSize*prec;
      }

    }
  }

   int optflag=1;
   exchange_cpu_sitelink(param->X, cpuGauge, ghost_cpuGauge, ghost_cpuGauge_diag, prec, optflag);
   
   if (prec == QUDA_DOUBLE_PRECISION) {
     do_loadLinkToGPU(param->X, (double2*)(cudaGauge.even), (double2*)(cudaGauge.odd), (double**)cpuGauge, 
		      (double**)ghost_cpuGauge, (double**)ghost_cpuGauge_diag, 
		      cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
		      Vsh_x, Vsh_y, Vsh_z, Vsh_t, 
		      prec);
   } else if (prec == QUDA_SINGLE_PRECISION) {
     do_loadLinkToGPU(param->X, (float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (float**)cpuGauge, 
		      (float**)ghost_cpuGauge, (float**)ghost_cpuGauge_diag, 
		      cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
		      Vsh_x, Vsh_y, Vsh_z, Vsh_t, 
		      prec);    
   }else{
    printf("ERROR: half precision not supported in this funciton %s\n", __FUNCTION__);
    exit(1);
  }

   
  for(int i=0;i < 4;i++){
#if (CUDA_VERSION >= 4000)
    cudaFreeHost(ghost_cpuGauge[i]);
#else
    free(ghost_cpuGauge[i]);
#endif
  }
  for(int i=0;i <4; i++){ 
    for(int j=0;j <4; j++){
      if (i==j){
        continue;
      }
#if (CUDA_VERSION >= 4000)
      cudaFreeHost(ghost_cpuGauge_diag[i*4+j]);
#else
      free(ghost_cpuGauge_diag[i*4+j]);
#endif
    }
  }


}


template<typename FloatN, typename Float>
static void 
do_loadLinkToGPU_ex(int* X, FloatN *even, FloatN *odd, Float **cpuGauge, 
		    ReconstructType reconstruct, int bytes, int Vh_ex, int pad, 
		    QudaPrecision prec) 
{
  cudaStream_t streams[2];
  for(int i=0;i < 2; i++){
    cudaStreamCreate(&streams[i]);
  }
  

  int i;
  char* tmp_even = NULL;
  char* tmp_odd = NULL;
  int len = Vh_ex*gaugeSiteSize*sizeof(Float);

  cudaMalloc(&tmp_even, 8*len);
  if(tmp_even == NULL){
    errorQuda("Error: cudaMalloc failed\n");
  }
  tmp_odd = tmp_even + 4*len;

  //even links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_even + i*len, cpuGauge[i], len, cudaMemcpyHostToDevice, streams[0]); 
#else
    cudaMemcpy(tmp_even + i*len, cpuGauge[i], len, cudaMemcpyHostToDevice); 
#endif
  
  }
  
  link_format_cpu_to_gpu((void*)even, (void*)tmp_even,  reconstruct, bytes, Vh_ex, pad, 0, prec, streams[0]); 

  //odd links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_odd + i*len, cpuGauge[i] + Vh_ex*gaugeSiteSize, len, cudaMemcpyHostToDevice, streams[1]);
#else
    cudaMemcpy(tmp_odd + i*len, cpuGauge[i] + Vh_ex*gaugeSiteSize, len, cudaMemcpyHostToDevice);
#endif
  }
  link_format_cpu_to_gpu((void*)odd, (void*)tmp_odd, reconstruct, bytes, Vh_ex, pad, 0, prec, streams[1]); 
  
  
  for(int i=0;i < 2;i++){
    cudaStreamSynchronize(streams[i]);
  }
  
  cudaFree(tmp_even);
  
  for(int i=0;i < 2;i++){
    cudaStreamDestroy(streams[i]);
  }
  CUERR;
}


void 
loadLinkToGPU_ex(FullGauge cudaGauge, void **cpuGauge, QudaGaugeParam* param_ex)
{
  
  if (param_ex->cpu_prec  != param_ex->cuda_prec){
    printf("ERROR: cpu precision and cuda precision must be the same in this function %s\n", __FUNCTION__);
    exit(1);
  }
  QudaPrecision prec= param_ex->cpu_prec;
  
  int* E = param_ex->X;
  int pad = param_ex->ga_pad;

   int optflag=1;
   
   if (prec == QUDA_DOUBLE_PRECISION) {
     do_loadLinkToGPU_ex(E, (double2*)(cudaGauge.even), (double2*)(cudaGauge.odd), (double**)cpuGauge, 
			 cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
			 prec);
   } else if (prec == QUDA_SINGLE_PRECISION) {
     do_loadLinkToGPU_ex(E, (float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (float**)cpuGauge, 
			 cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
			 prec);    
   }else{
     printf("ERROR: half precision not supported in this funciton %s\n", __FUNCTION__);
     exit(1);
   }
   
}



template<typename FloatN, typename Float>
static void 
do_loadLinkToGPU_nl(int* L, FloatN *even, FloatN *odd, Float **cpuGauge, 
		    ReconstructType reconstruct, int bytes, int Vh, int pad, 
		    QudaPrecision prec) 
{
  cudaStream_t streams[2];
  for(int i=0;i < 2; i++){
    cudaStreamCreate(&streams[i]);
  }
  int Vh_nl = L[0]*L[1]*L[2]*L[3]/2;
  int E4E3E2 = (L[3]+2)*(L[2]+2)*(L[1]+2);
  int E4E3E1 = (L[3]+2)*(L[2]+2)*(L[0]+2);
  int E4E2E1 = (L[3]+2)*(L[1]+2)*(L[0]+2);
  int E3E2E1 = (L[2]+2)*(L[1]+2)*(L[1]+2);

  int i;
  char* tmp_even;
  char* tmp_odd;
  int len = Vh_nl*gaugeSiteSize*sizeof(Float);
  int ghostV = E4E3E2+E4E3E1+E4E2E1+E3E2E1;
  int glen_sum = ghostV *gaugeSiteSize*sizeof(Float);
  cudaMalloc(&tmp_even, 4*(len+glen_sum)); CUERR;
  cudaMalloc(&tmp_odd, 4*(len+glen_sum)); CUERR;
  
  //even links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_even + i*(len+glen_sum), cpuGauge[i], len+glen_sum, cudaMemcpyHostToDevice, streams[0]);         
#else
    cudaMemcpy(tmp_even + i*(len+glen_sum), cpuGauge[i], len+glen_sum, cudaMemcpyHostToDevice); 
#endif
    
  }
  
  link_format_cpu_to_gpu((void*)even, (void*)tmp_even,  reconstruct, bytes, Vh, pad, ghostV, prec, streams[0]); CUERR;

  //odd links
  for(i=0;i < 4; i++){
#if (CUDA_VERSION >=4000)
    cudaMemcpyAsync(tmp_odd + i*(len+glen_sum), cpuGauge[i] + (Vh+ghostV)*gaugeSiteSize, len+glen_sum, cudaMemcpyHostToDevice, streams[1]);CUERR;
#else
    cudaMemcpy(tmp_odd + i*(len+glen_sum), cpuGauge[i] + (Vh+ghostV)*gaugeSiteSize, len+glen_sum, cudaMemcpyHostToDevice);CUERR;
#endif
  }
  
  link_format_cpu_to_gpu((void*)odd, (void*)tmp_odd, reconstruct, bytes, Vh, pad, ghostV, prec, streams[1]); CUERR;
  
  for(int i=0;i < 2;i++){
    cudaStreamSynchronize(streams[i]);
  }
  
  cudaFree(tmp_even);
  cudaFree(tmp_odd);
  
  for(int i=0;i < 2;i++){
    cudaStreamDestroy(streams[i]);
  }
  CUERR;
}


void 
loadLinkToGPU_nl(FullGauge cudaGauge, void **cpuGauge, QudaGaugeParam* param_nl)
{
  
  /*
  {
    double* data = (double*)cpuGauge[0];
    //data += (cudaGauge.volumeCB+param_nl->ga_pad)*18;
    printf("gauge_quda sitelink is (cudaGauge.volumeCB=%d, pad=%d)\n", cudaGauge.volumeCB, param_nl->ga_pad);
    printf("(%f %f) (%f %f) (%f %f)\n"
	   "(%f %f) (%f %f) (%f %f)\n"
	   "(%f %f) (%f %f) (%f %f)\n",
	   data[0], data[1], data[2], data[3], data[4], data[5], 
	   data[6], data[7], data[8], data[9], data[10], data[11], 
	   data[12], data[13], data[14], data[15], data[16], data[17]); 
  }
  
  */


  if (param_nl->cpu_prec  != param_nl->cuda_prec){
    printf("ERROR: cpu precision and cuda precision must be the same in this function %s\n", __FUNCTION__);
    exit(1);
  }
  QudaPrecision prec= param_nl->cpu_prec;
  
  int* L = param_nl->X;
  int pad = param_nl->ga_pad;
  
  int optflag=1;
  
  if (prec == QUDA_DOUBLE_PRECISION) {
    do_loadLinkToGPU_nl(L, (double2*)(cudaGauge.even), (double2*)(cudaGauge.odd), (double**)cpuGauge, 
			cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
			prec);
  } else if (prec == QUDA_SINGLE_PRECISION) {
    do_loadLinkToGPU_nl(L, (float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (float**)cpuGauge, 
			cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volumeCB, pad, 
			prec);    
  }else{
    printf("ERROR: half precision not supported in this funciton %s\n", __FUNCTION__);
    exit(1);
   }
   
}



template<typename FloatN, typename Float>
static void 
do_storeLinkToCPU(Float* cpuGauge, FloatN *even, FloatN *odd, 
		  int bytes, int Vh, int stride, QudaPrecision prec) 
{  
  cudaStream_t streams[2];
  for(int i=0;i < 2; i++){
    cudaStreamCreate(&streams[i]);
  }
  

  double* unpackedDataEven;
  double* unpackedDataOdd;
  int datalen = 4*Vh*gaugeSiteSize*sizeof(Float);
  cudaMalloc(&unpackedDataEven, datalen); CUERR;
  cudaMalloc(&unpackedDataOdd, datalen); CUERR;
  
  //unpack even data kernel
  link_format_gpu_to_cpu((void*)unpackedDataEven, (void*)even, bytes, Vh, stride, prec, streams[0]);
#if (CUDA_VERSION >= 4000)
  cudaMemcpyAsync(cpuGauge, unpackedDataEven, datalen, cudaMemcpyDeviceToHost, streams[0]);
#else
  cudaMemcpy(cpuGauge, unpackedDataEven, datalen, cudaMemcpyDeviceToHost);
#endif
  
  //unpack odd data kernel
  link_format_gpu_to_cpu((void*)unpackedDataOdd, (void*)odd,  bytes, Vh, stride, prec, streams[1]);
#if (CUDA_VERSION >=4000)
  cudaMemcpyAsync(cpuGauge + 4*Vh*gaugeSiteSize, unpackedDataOdd, datalen, cudaMemcpyDeviceToHost, streams[1]);  
  for(int i=0;i < 2; i++){
    cudaStreamSynchronize(streams[i]);
  }
#else
  cudaMemcpy(cpuGauge + 4*Vh*gaugeSiteSize, unpackedDataOdd, datalen, cudaMemcpyDeviceToHost);  
#endif
  
  cudaFree(unpackedDataEven);
  cudaFree(unpackedDataOdd);
  
  for(int i=0;i < 2;i++){
    cudaStreamDestroy(streams[i]);
  }


  CUERR;
}



void 
storeLinkToCPU(void* cpuGauge, FullGauge *cudaGauge, QudaGaugeParam* param)
{
  
  QudaPrecision cpu_prec = param->cpu_prec;
  QudaPrecision cuda_prec= param->cuda_prec;

  if (cpu_prec  != cuda_prec){
    printf("ERROR: cpu precision and cuda precision must be the same in this function %s\n", __FUNCTION__);
    exit(1);
  }
    
  if (cudaGauge->reconstruct != QUDA_RECONSTRUCT_NO){
    printf("ERROR: it makes no sense to get data back to cpu for 8/12 reconstruct, function %s\n", __FUNCTION__);
    exit(1);
  }
  
  int stride = cudaGauge->volumeCB + cudaGauge->pad;
  
  if (cuda_prec == QUDA_DOUBLE_PRECISION){
    do_storeLinkToCPU( (double*)cpuGauge, (double2*) cudaGauge->even, (double2*)cudaGauge->odd, 
		       cudaGauge->bytes, cudaGauge->volumeCB, stride, cuda_prec);
  }else if (cuda_prec == QUDA_SINGLE_PRECISION){
    do_storeLinkToCPU( (float*)cpuGauge, (float2*) cudaGauge->even, (float2*)cudaGauge->odd, 
		       cudaGauge->bytes, cudaGauge->volumeCB, stride, cuda_prec);
  }else{
    printf("ERROR: half precision not supported in this function %s\n", __FUNCTION__);
    exit(1);
  }
}


#endif

