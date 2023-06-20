#pragma once

#include "vec3.h"
#include <cuda_runtime.h>

namespace maniskill {

struct mat3 {
  inline CUDA_CALLABLE mat3() = default;

  inline CUDA_CALLABLE mat3(vec3 c0, vec3 c1, vec3 c2) {
    data[0][0] = c0.x;
    data[1][0] = c0.y;
    data[2][0] = c0.z;
    data[0][1] = c1.x;
    data[1][1] = c1.y;
    data[2][1] = c1.z;
    data[0][2] = c2.x;
    data[1][2] = c2.y;
    data[2][2] = c2.z;
  }

  inline CUDA_CALLABLE mat3(float m00, float m01, float m02, float m10,
                            float m11, float m12, float m20, float m21,
                            float m22) {
    data[0][0] = m00;
    data[0][1] = m01;
    data[0][2] = m02;
    data[1][0] = m10;
    data[1][1] = m11;
    data[1][2] = m12;
    data[2][0] = m20;
    data[2][1] = m21;
    data[2][2] = m22;
  }

  inline CUDA_CALLABLE explicit mat3(vec3 diag)
      : mat3(diag.x, 0.f, 0.f, 0.f, diag.y, 0.f, 0.f, 0.f, diag.z) {}

  inline CUDA_CALLABLE explicit mat3(float d)
      : mat3(d, 0.f, 0.f, 0.f, d, 0.f, 0.f, 0.f, d) {}

  inline CUDA_CALLABLE vec3 row(int index) const { return (vec3 &)data[index]; }
  inline CUDA_CALLABLE vec3 diag() const { return vec3(data[0][0], data[1][1], data[2][2]); }
  inline CUDA_CALLABLE void show() const {
	printf("%f %f %f\n%f %f %f\n%f %f %f\n",
			data[0][0], data[0][1], data[0][2],
			data[1][0], data[1][1], data[1][2],
			data[2][0], data[2][1], data[2][2]);
   }
  inline CUDA_CALLABLE vec3 col(int index) const {
    return vec3(data[0][index], data[1][index], data[2][index]);
  }

  inline CUDA_CALLABLE float operator()(int x, int y) const {
    return data[x][y];
  }

  inline CUDA_CALLABLE mat3 transpose() const {
    mat3 r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r.data[i][j] = data[j][i];
      }
    }
    return r;
  }

  inline CUDA_CALLABLE float sum() const {
    float out = 0.;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        out += data[i][j];
      }
    }
    return out;
  }

  inline CUDA_CALLABLE float determinant() const {
    return dot(row(0), cross(row(1), row(2)));
  }

  inline CUDA_CALLABLE mat3 mul(mat3 const &b) const {
    mat3 r(0.f);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          r.data[i][j] += data[i][k] * b.data[k][j];
        }
      }
    }
    return r;
  }

  inline CUDA_CALLABLE vec3 mul(vec3 const &b) const {
    vec3 r = col(0) * b.x + col(1) * b.y + col(2) * b.z;
    return r;
  }

  float data[3][3];
};

inline CUDA_CALLABLE mat3 operator*(mat3 const &a, float b) {
  mat3 r;
  r.data[0][0] = a.data[0][0] * b;
  r.data[0][1] = a.data[0][1] * b;
  r.data[0][2] = a.data[0][2] * b;
  r.data[1][0] = a.data[1][0] * b;
  r.data[1][1] = a.data[1][1] * b;
  r.data[1][2] = a.data[1][2] * b;
  r.data[2][0] = a.data[2][0] * b;
  r.data[2][1] = a.data[2][1] * b;
  r.data[2][2] = a.data[2][2] * b;
  return r;
}

inline CUDA_CALLABLE mat3 operator*(mat3 const &a, mat3 const &b) {
  mat3 r;
  r.data[0][0] = a.data[0][0] * b.data[0][0];
  r.data[0][1] = a.data[0][1] * b.data[0][1];
  r.data[0][2] = a.data[0][2] * b.data[0][2];
  r.data[1][0] = a.data[1][0] * b.data[1][0];
  r.data[1][1] = a.data[1][1] * b.data[1][1];
  r.data[1][2] = a.data[1][2] * b.data[1][2];
  r.data[2][0] = a.data[2][0] * b.data[2][0];
  r.data[2][1] = a.data[2][1] * b.data[2][1];
  r.data[2][2] = a.data[2][2] * b.data[2][2];
  return r;
}

inline CUDA_CALLABLE mat3 operator*(float a, mat3 const &b) { return b * a; }

inline CUDA_CALLABLE mat3 operator+(mat3 const &a, mat3 const &b) {
  mat3 r;
  r.data[0][0] = a.data[0][0] + b.data[0][0];
  r.data[0][1] = a.data[0][1] + b.data[0][1];
  r.data[0][2] = a.data[0][2] + b.data[0][2];
  r.data[1][0] = a.data[1][0] + b.data[1][0];
  r.data[1][1] = a.data[1][1] + b.data[1][1];
  r.data[1][2] = a.data[1][2] + b.data[1][2];
  r.data[2][0] = a.data[2][0] + b.data[2][0];
  r.data[2][1] = a.data[2][1] + b.data[2][1];
  r.data[2][2] = a.data[2][2] + b.data[2][2];
  return r;
}


inline CUDA_CALLABLE mat3 operator+=(mat3 &a, mat3 const &b) {
  a.data[0][0] += b.data[0][0];
  a.data[0][1] += b.data[0][1];
  a.data[0][2] += b.data[0][2];
  a.data[1][0] += b.data[1][0];
  a.data[1][1] += b.data[1][1];
  a.data[1][2] += b.data[1][2];
  a.data[2][0] += b.data[2][0];
  a.data[2][1] += b.data[2][1];
  a.data[2][2] += b.data[2][2];
  return a;
}

inline CUDA_CALLABLE mat3 operator-(mat3 const &a, mat3 const &b) {
  mat3 r;
  r.data[0][0] = a.data[0][0] - b.data[0][0];
  r.data[0][1] = a.data[0][1] - b.data[0][1];
  r.data[0][2] = a.data[0][2] - b.data[0][2];
  r.data[1][0] = a.data[1][0] - b.data[1][0];
  r.data[1][1] = a.data[1][1] - b.data[1][1];
  r.data[1][2] = a.data[1][2] - b.data[1][2];
  r.data[2][0] = a.data[2][0] - b.data[2][0];
  r.data[2][1] = a.data[2][1] - b.data[2][1];
  r.data[2][2] = a.data[2][2] - b.data[2][2];
  return r;
}

inline CUDA_CALLABLE mat3 outer(vec3 const &a, vec3 const &b) {
  return mat3(a * b.x, a * b.y, a * b.z);
}

inline CUDA_CALLABLE mat3 out_product(const vec3& a, const vec3& b) {
  return mat3(
    a.x * b.x, a.x * b.y, a.x * b.z,
    a.y * b.x, a.y * b.y, a.y * b.z,
    a.z * b.x, a.z * b.y, a.z * b.z
  );
}


} // namespace maniskill
