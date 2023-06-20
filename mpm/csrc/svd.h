// https://github.com/ericjang/svd3/blob/master/svd3_cuda/svd3_cuda.h

#pragma once


/*
#define USE_SCALAR_IMPLEMENTATION
#define COMPUTE_V_AS_MATRIX
#define COMPUTE_U_AS_MATRIX
#include "Singular_Value_Decomposition_Preamble.hpp"
*/

#include "mat3.h"

namespace maniskill {

#define _gamma 5.82842712474619f // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.9238795325112867f // cos(pi/8)
#define _sstar 0.3826834323650897f // sin(p/8)
#define _EPSILON 1e-6

inline CUDA_CALLABLE double accurateSqrt(double x) { return x / sqrt(x); }

inline CUDA_CALLABLE void condSwap(bool c, double &X, double &Y) {
  // used in step 2
  double Z = X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}

inline CUDA_CALLABLE void condNegSwap(bool c, double &X, double &Y) {
  // used in step 2 and 3
  double Z = -X;
  X = c ? Y : X;
  Y = c ? Z : Y;
}

// matrix multiplication M = A * B
inline CUDA_CALLABLE void
multAB(double a11, double a12, double a13, double a21, double a22, double a23,
       double a31, double a32, double a33,
       //
       double b11, double b12, double b13, double b21, double b22, double b23,
       double b31, double b32, double b33,
       //
       double &m11, double &m12, double &m13, double &m21, double &m22, double &m23,
       double &m31, double &m32, double &m33) {

  m11 = a11 * b11 + a12 * b21 + a13 * b31;
  m12 = a11 * b12 + a12 * b22 + a13 * b32;
  m13 = a11 * b13 + a12 * b23 + a13 * b33;
  m21 = a21 * b11 + a22 * b21 + a23 * b31;
  m22 = a21 * b12 + a22 * b22 + a23 * b32;
  m23 = a21 * b13 + a22 * b23 + a23 * b33;
  m31 = a31 * b11 + a32 * b21 + a33 * b31;
  m32 = a31 * b12 + a32 * b22 + a33 * b32;
  m33 = a31 * b13 + a32 * b23 + a33 * b33;
}

// matrix multiplication M = Transpose[A] * B
inline CUDA_CALLABLE void
multAtB(double a11, double a12, double a13, double a21, double a22, double a23,
        double a31, double a32, double a33,
        //
        double b11, double b12, double b13, double b21, double b22, double b23,
        double b31, double b32, double b33,
        //
        double &m11, double &m12, double &m13, double &m21, double &m22, double &m23,
        double &m31, double &m32, double &m33) {
  m11 = a11 * b11 + a21 * b21 + a31 * b31;
  m12 = a11 * b12 + a21 * b22 + a31 * b32;
  m13 = a11 * b13 + a21 * b23 + a31 * b33;
  m21 = a12 * b11 + a22 * b21 + a32 * b31;
  m22 = a12 * b12 + a22 * b22 + a32 * b32;
  m23 = a12 * b13 + a22 * b23 + a32 * b33;
  m31 = a13 * b11 + a23 * b21 + a33 * b31;
  m32 = a13 * b12 + a23 * b22 + a33 * b32;
  m33 = a13 * b13 + a23 * b23 + a33 * b33;
}

inline CUDA_CALLABLE void quatToMat3(const double *qV, double &m11, double &m12,
                                     double &m13, double &m21, double &m22,
                                     double &m23, double &m31, double &m32,
                                     double &m33) {
  double w = qV[3];
  double x = qV[0];
  double y = qV[1];
  double z = qV[2];

  double qxx = x * x;
  double qyy = y * y;
  double qzz = z * z;
  double qxz = x * z;
  double qxy = x * y;
  double qyz = y * z;
  double qwx = w * x;
  double qwy = w * y;
  double qwz = w * z;

  m11 = 1 - 2 * (qyy + qzz);
  m12 = 2 * (qxy - qwz);
  m13 = 2 * (qxz + qwy);
  m21 = 2 * (qxy + qwz);
  m22 = 1 - 2 * (qxx + qzz);
  m23 = 2 * (qyz - qwx);
  m31 = 2 * (qxz - qwy);
  m32 = 2 * (qyz + qwx);
  m33 = 1 - 2 * (qxx + qyy);
}

inline CUDA_CALLABLE void approximateGivensQuaternion(double a11, double a12,
                                                      double a22, double &ch,
                                                      double &sh) {
  /*
   * Given givens angle computed by approximateGivensAngles,
   * compute the corresponding rotation quaternion.
   */
  ch = 2 * (a11 - a22);
  sh = a12;
  bool b = _gamma * sh * sh < ch * ch;
  double w = 1.0 / sqrt(ch * ch + sh * sh);
  ch = b ? w * ch : _cstar;
  sh = b ? w * sh : _sstar;
}

inline CUDA_CALLABLE void jacobiConjugation(const int x, const int y,
                                            const int z, double &s11, double &s21,
                                            double &s22, double &s31, double &s32,
                                            double &s33, double *qV) {
  double ch, sh;
  approximateGivensQuaternion(s11, s21, s22, ch, sh);

  double scale = ch * ch + sh * sh;
  double a = (ch * ch - sh * sh) / scale;
  double b = (2 * sh * ch) / scale;

  // make temp copy of S
  double _s11 = s11;
  double _s21 = s21;
  double _s22 = s22;
  double _s31 = s31;
  double _s32 = s32;
  double _s33 = s33;

  // perform conjugation S = Q'*S*Q
  // Q already implicitly solved from a, b
  s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
  s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
  s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
  s31 = a * _s31 + b * _s32;
  s32 = -b * _s31 + a * _s32;
  s33 = _s33;

  // update cumulative rotation qV
  double tmp[3];
  tmp[0] = qV[0] * sh;
  tmp[1] = qV[1] * sh;
  tmp[2] = qV[2] * sh;
  sh *= qV[3];

  qV[0] *= ch;
  qV[1] *= ch;
  qV[2] *= ch;
  qV[3] *= ch;

  // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
  // for (p,q) = ((0,1),(1,2),(0,2))
  qV[z] += sh;
  qV[3] -= tmp[z]; // w
  qV[x] += tmp[y];
  qV[y] -= tmp[x];

  // re-arrange matrix for next iteration
  _s11 = s22;
  _s21 = s32;
  _s22 = s33;
  _s31 = s21;
  _s32 = s31;
  _s33 = s11;
  s11 = _s11;
  s21 = _s21;
  s22 = _s22;
  s31 = _s31;
  s32 = _s32;
  s33 = _s33;
}

inline CUDA_CALLABLE double dist2(double x, double y, double z) {
  return x * x + y * y + z * z;
}

// finds transformation that diagonalizes a symmetric matrix
inline CUDA_CALLABLE void jacobiEigenanlysis( // symmetric matrix
    double &s11, double &s21, double &s22, double &s31, double &s32, double &s33,
    // quaternion representation of V
    double *qV) {
  qV[3] = 1;
  qV[0] = 0;
  qV[1] = 0;
  qV[2] = 0; // follow same indexing convention as GLM
  for (int i = 0; i < 8; i++) {
    // we wish to eliminate the maximum off-diagonal element
    // on every iteration, but cycling over all 3 possible rotations
    // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
    //  asymptotic convergence
    jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
    jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
    jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
  }
}

inline CUDA_CALLABLE void
sortSingularValues( // matrix that we want to decompose
    double &b11, double &b12, double &b13, double &b21, double &b22, double &b23,
    double &b31, double &b32, double &b33,
    // sort V simultaneously
    double &v11, double &v12, double &v13, double &v21, double &v22, double &v23,
    double &v31, double &v32, double &v33) {
  double rho1 = dist2(b11, b21, b31);
  double rho2 = dist2(b12, b22, b32);
  double rho3 = dist2(b13, b23, b33);
  bool c;
  c = rho1 < rho2;
  condNegSwap(c, b11, b12);
  condNegSwap(c, v11, v12);
  condNegSwap(c, b21, b22);
  condNegSwap(c, v21, v22);
  condNegSwap(c, b31, b32);
  condNegSwap(c, v31, v32);
  condSwap(c, rho1, rho2);
  c = rho1 < rho3;
  condNegSwap(c, b11, b13);
  condNegSwap(c, v11, v13);
  condNegSwap(c, b21, b23);
  condNegSwap(c, v21, v23);
  condNegSwap(c, b31, b33);
  condNegSwap(c, v31, v33);
  condSwap(c, rho1, rho3);
  c = rho2 < rho3;
  condNegSwap(c, b12, b13);
  condNegSwap(c, v12, v13);
  condNegSwap(c, b22, b23);
  condNegSwap(c, v22, v23);
  condNegSwap(c, b32, b33);
  condNegSwap(c, v32, v33);
}

inline CUDA_CALLABLE void QRGivensQuaternion(double a1, double a2, double &ch,
                                             double &sh) {
  // a1 = pivot point on diagonal
  // a2 = lower triangular entry we want to annihilate
  double epsilon = _EPSILON;
  double rho = accurateSqrt(a1 * a1 + a2 * a2);

  sh = rho > epsilon ? a2 : 0;
  ch = fabs(a1) + fmax(rho, epsilon);
  bool b = a1 < 0;
  condSwap(b, sh, ch);
  double w = 1.0 / sqrt(ch * ch + sh * sh);
  ch *= w;
  sh *= w;
}

inline CUDA_CALLABLE void QRDecomposition( // matrix that we want to decompose
    double b11, double b12, double b13, double b21, double b22, double b23, double b31,
    double b32, double b33,
    // output Q
    double &q11, double &q12, double &q13, double &q21, double &q22, double &q23,
    double &q31, double &q32, double &q33,
    // output R
    double &r11, double &r12, double &r13, double &r21, double &r22, double &r23,
    double &r31, double &r32, double &r33) {
  double ch1, sh1, ch2, sh2, ch3, sh3;
  double a, b;

  // first givens rotation (ch,0,0,sh)
  QRGivensQuaternion(b11, b21, ch1, sh1);
  a = 1 - 2 * sh1 * sh1;
  b = 2 * ch1 * sh1;
  // apply B = Q' * B
  r11 = a * b11 + b * b21;
  r12 = a * b12 + b * b22;
  r13 = a * b13 + b * b23;
  r21 = -b * b11 + a * b21;
  r22 = -b * b12 + a * b22;
  r23 = -b * b13 + a * b23;
  r31 = b31;
  r32 = b32;
  r33 = b33;

  // second givens rotation (ch,0,-sh,0)
  QRGivensQuaternion(r11, r31, ch2, sh2);
  a = 1 - 2 * sh2 * sh2;
  b = 2 * ch2 * sh2;
  // apply B = Q' * B;
  b11 = a * r11 + b * r31;
  b12 = a * r12 + b * r32;
  b13 = a * r13 + b * r33;
  b21 = r21;
  b22 = r22;
  b23 = r23;
  b31 = -b * r11 + a * r31;
  b32 = -b * r12 + a * r32;
  b33 = -b * r13 + a * r33;

  // third givens rotation (ch,sh,0,0)
  QRGivensQuaternion(b22, b32, ch3, sh3);
  a = 1 - 2 * sh3 * sh3;
  b = 2 * ch3 * sh3;
  // R is now set to desired value
  r11 = b11;
  r12 = b12;
  r13 = b13;
  r21 = a * b21 + b * b31;
  r22 = a * b22 + b * b32;
  r23 = a * b23 + b * b33;
  r31 = -b * b21 + a * b31;
  r32 = -b * b22 + a * b32;
  r33 = -b * b23 + a * b33;

  // construct the cumulative rotation Q=Q1 * Q2 * Q3
  // the number of doubleing point operations for three quaternion
  // multiplications is more or less comparable to the explicit form of the
  // joined matrix. certainly more memory-efficient!
  double sh12 = sh1 * sh1;
  double sh22 = sh2 * sh2;
  double sh32 = sh3 * sh3;

  q11 = (-1 + 2 * sh12) * (-1 + 2 * sh22);
  q12 = 4 * ch2 * ch3 * (-1 + 2 * sh12) * sh2 * sh3 +
        2 * ch1 * sh1 * (-1 + 2 * sh32);
  q13 = 4 * ch1 * ch3 * sh1 * sh3 -
        2 * ch2 * (-1 + 2 * sh12) * sh2 * (-1 + 2 * sh32);

  q21 = 2 * ch1 * sh1 * (1 - 2 * sh22);
  q22 = -8 * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 +
        (-1 + 2 * sh12) * (-1 + 2 * sh32);
  q23 = -2 * ch3 * sh3 +
        4 * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1 + 2 * sh32));

  q31 = 2 * ch2 * sh2;
  q32 = 2 * ch3 * (1 - 2 * sh22) * sh3;
  q33 = (-1 + 2 * sh22) * (-1 + 2 * sh32);
}

inline CUDA_CALLABLE void _svd( // input A
    double a11, double a12, double a13, double a21, double a22, double a23, double a31,
    double a32, double a33,
    // output U
    double &u11, double &u12, double &u13, double &u21, double &u22, double &u23,
    double &u31, double &u32, double &u33,
    // output S
    double &s11, double &s12, double &s13, double &s21, double &s22, double &s23,
    double &s31, double &s32, double &s33,
    // output V
    double &v11, double &v12, double &v13, double &v21, double &v22, double &v23,


    double &v31, double &v32, double &v33) {
  // normal equations matrix
  double ATA11, ATA12, ATA13;
  double ATA21, ATA22, ATA23;
  double ATA31, ATA32, ATA33;

  multAtB(a11, a12, a13, a21, a22, a23, a31, a32, a33, a11, a12, a13, a21, a22,
          a23, a31, a32, a33, ATA11, ATA12, ATA13, ATA21, ATA22, ATA23, ATA31,
          ATA32, ATA33);

  // symmetric eigenalysis
  double qV[4];
  jacobiEigenanlysis(ATA11, ATA21, ATA22, ATA31, ATA32, ATA33, qV);
  quatToMat3(qV, v11, v12, v13, v21, v22, v23, v31, v32, v33);

  double b11, b12, b13;
  double b21, b22, b23;
  double b31, b32, b33;
  multAB(a11, a12, a13, a21, a22, a23, a31, a32, a33, v11, v12, v13, v21, v22,
         v23, v31, v32, v33, b11, b12, b13, b21, b22, b23, b31, b32, b33);

  // sort singular values and find V
  sortSingularValues(b11, b12, b13, b21, b22, b23, b31, b32, b33, v11, v12, v13,
                     v21, v22, v23, v31, v32, v33);

  // QR decomposition
  QRDecomposition(b11, b12, b13, b21, b22, b23, b31, b32, b33, u11, u12, u13,
                  u21, u22, u23, u31, u32, u33, s11, s12, s13, s21, s22, s23,
                  s31, s32, s33);
}

inline CUDA_CALLABLE void svd3(const mat3 &A, mat3 &U, vec3 &sigma, mat3 &V) {
  double s12, s13, s21, s23, s31, s32;

  double U00, U01, U02, U10, U11, U12, U20, U21, U22;
  double V00, V01, V02, V10, V11, V12, V20, V21, V22;
  double sx, sy, sz;
  _svd(A.data[0][0], A.data[0][1], A.data[0][2], A.data[1][0], A.data[1][1],
       A.data[1][2], A.data[2][0], A.data[2][1], A.data[2][2],
       U00, U01, U02, U10, U11, U12, U20, U21, U22,

       //U.data[0][0], U.data[0][1], U.data[0][2], U.data[1][0], U.data[1][1],
       //U.data[1][2], U.data[2][0], U.data[2][1], U.data[2][2],

       sx, s12, s13, s21, sy, s23, s31, s32, sz,

       //V.data[0][0], V.data[0][1], V.data[0][2], V.data[1][0], V.data[1][1],
       //V.data[1][2], V.data[2][0], V.data[2][1], V.data[2][2]);
       V00, V01, V02, V10, V11, V12, V20, V21, V22);
  U = mat3(U00, U01, U02, U10, U11, U12, U20, U21, U22);
  V = mat3(V00, V01, V02, V10, V11, V12, V20, V21, V22);
  sigma.x = sx;
  sigma.y = sy;
  sigma.z = sz;
}

/*
__device__ __forceinline__
void svd3_v2(const mat3 &A, mat3 &U, vec3 &sigma, mat3 &V) {
  //A.show();

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

		ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A.data[0][0];) ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A.data[1][0];) ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A.data[2][0];)
		ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A.data[0][1];) ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A.data[1][1];) ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A.data[2][1];)
		ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A.data[0][2];) ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A.data[1][2];) ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A.data[2][2];)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp" 

  U=mat3(Su11.f, Su12.f, Su13.f, Su21.f, Su22.f, Su23.f, Su31.f, Su32.f, Su33.f);
  V=mat3(Sv11.f, Sv12.f, Sv13.f, Sv21.f, Sv22.f, Sv23.f, Sv31.f, Sv32.f, Sv33.f);
  sigma.x = Sa11.f;
  sigma.y = Sa22.f;
  sigma.z = Sa33.f;

  }
*/

} // namespace maniskill
