#pragma once

#include "common.h"
#include <cuda_runtime.h>

namespace maniskill {

struct vec3 {
  inline CUDA_CALLABLE vec3() = default;
  inline CUDA_CALLABLE vec3(float _x, float _y, float _z)
      : x(_x), y(_y), z(_z) {}
  inline CUDA_CALLABLE explicit vec3(float _x) : vec3(_x, _x, _x) {}
  inline CUDA_CALLABLE vec3(float3 v) : vec3(v.x, v.y, v.z) {}

  float x, y, z;
};

struct ivec3 {
  inline CUDA_CALLABLE ivec3() = default;
  inline CUDA_CALLABLE ivec3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
  inline CUDA_CALLABLE explicit ivec3(int _x) : ivec3(_x, _x, _x) {}

  int x, y, z;
};

inline CUDA_CALLABLE float dot(vec3 const &a, vec3 const &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}


inline CUDA_CALLABLE vec3 cross(vec3 const &a, vec3 const &b) {
  vec3 r;
  r.x = a.y * b.z - a.z * b.y;
  r.y = a.z * b.x - a.x * b.z;
  r.z = a.x * b.y - a.y * b.x;
  return r;
}

inline CUDA_CALLABLE vec3 operator*(vec3 const &a, vec3 const &b) {
  return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline CUDA_CALLABLE vec3 operator/(vec3 const &a, vec3 const &b) {
  return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline CUDA_CALLABLE vec3 operator*(vec3 const &a, float b) {
  return vec3(a.x * b, a.y * b, a.z * b);
}

inline CUDA_CALLABLE vec3 operator/(vec3 const &a, float b) {
  return vec3(a.x / b, a.y / b, a.z / b);
}

inline CUDA_CALLABLE vec3 operator*(float a, vec3 const &b) {
  return vec3(a * b.x, a * b.y, a * b.z);
}

inline CUDA_CALLABLE vec3 operator+(vec3 const &a, vec3 const &b) {
  return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CUDA_CALLABLE vec3 operator+=(vec3 &a, vec3 const &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

inline CUDA_CALLABLE ivec3 operator+(ivec3 const &a, ivec3 const &b) {
  return ivec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CUDA_CALLABLE vec3 operator+(vec3 const &a, float b) {
  return vec3(a.x + b, a.y + b, a.z + b);
}

inline CUDA_CALLABLE vec3 operator+(float b, vec3 const &a) {
  return vec3(a.x + b, a.y + b, a.z + b);
}

inline CUDA_CALLABLE vec3 operator-(vec3 const &a, vec3 const &b) {
  return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline CUDA_CALLABLE ivec3 operator-(ivec3 const &a, ivec3 const &b) {
  return ivec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline CUDA_CALLABLE vec3 operator-(vec3 const &a, float b) {
  return vec3(a.x - b, a.y - b, a.z - b);
}

inline CUDA_CALLABLE vec3 operator-(float a, vec3 const &b) {
  return vec3(a - b.x, a - b.y, a - b.z);
}

inline CUDA_CALLABLE ivec3 cast_int(vec3 const &a) {
  return ivec3((int)floor(a.x), (int)floor(a.y), (int)floor(a.z));
}

inline CUDA_CALLABLE vec3 cast_float(ivec3 const &a) {
  return vec3((float)a.x, (float)a.y, (float)a.z);
}

inline CUDA_CALLABLE vec3 pow2(vec3 const &a) { return a * a; }

inline CUDA_CALLABLE float sum(vec3 const &a) { return a.x+a.y+a.z; }

inline CUDA_CALLABLE vec3 log(vec3 const &a) {
  return vec3(::log(a.x), ::log(a.y), ::log(a.z));
}

inline CUDA_CALLABLE vec3 exp(vec3 const &a) {
  return vec3(::exp(a.x), ::exp(a.y), ::exp(a.z));
}

inline CUDA_CALLABLE vec3 abs(vec3 const &a) {
  return vec3(::abs(a.x), ::abs(a.y), ::abs(a.z));
}

/* min max */

inline CUDA_CALLABLE vec3 max(vec3 const &a, float b) {
  return vec3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

inline CUDA_CALLABLE vec3 min(vec3 const &a, float b) {
  return vec3(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b));
}

inline CUDA_CALLABLE vec3 max(vec3 const &a, vec3 const &b) {
  return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline CUDA_CALLABLE vec3 min(vec3 const &a, vec3 const &b) {
  return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline CUDA_CALLABLE ivec3 max(ivec3 const &a, int b) {
  return ivec3(::max(a.x, b), ::max(a.y, b), ::max(a.z, b));
}

inline CUDA_CALLABLE ivec3 min(ivec3 const &a, int b) {
  return ivec3(::min(a.x, b), ::min(a.y, b), ::min(a.z, b));
}

inline CUDA_CALLABLE ivec3 max(ivec3 const &a, ivec3 const &b) {
  return ivec3(::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z));
}

inline CUDA_CALLABLE ivec3 min(ivec3 const &a, ivec3 const &b) {
  return ivec3(::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z));
}

inline CUDA_CALLABLE vec3 clamp(vec3 const &a, vec3 const &low,
                                vec3 const &high) {
  return min(max(a, low), high);
}

inline CUDA_CALLABLE vec3 clamp(vec3 const &a, float low, float high) {
  return min(max(a, low), high);
}

inline CUDA_CALLABLE ivec3 clamp(ivec3 const &a, ivec3 const &low,
                                 ivec3 const &high) {
  return min(max(a, low), high);
}

inline CUDA_CALLABLE ivec3 clamp(ivec3 const &a, float low, float high) {
  return min(max(a, low), high);
}

/* end */

//inline CUDA_CALLABLE float length(vec3 const &a) { return sqrt(dot(a, a)); }

//inline CUDA_CALLABLE vec3 normalize(vec3 const &a, float eps = 1e-8) {
//  return a / (length(a) + eps);
//}

inline __device__ void atomicMin(ivec3 *a, ivec3 const &b) {
  ::atomicMin(&a->x, b.x);
  ::atomicMin(&a->y, b.y);
  ::atomicMin(&a->z, b.z);
}

inline __device__ void atomicMax(ivec3 *a, ivec3 const &b) {
  ::atomicMax(&a->x, b.x);
  ::atomicMax(&a->y, b.y);
  ::atomicMax(&a->z, b.z);
}


inline __device__ void atomicAdd(vec3 *a, vec3 const &b) {
  ::atomicAdd(&a->x, b.x);
  ::atomicAdd(&a->y, b.y);
  ::atomicAdd(&a->z, b.z);
}

int CUDA_CALLABLE grid_index(int x, int y, int z, int dim_x, int dim_y,
                             int dim_z) {
  (void)dim_x;
  return (x * dim_y + y) * dim_z + z;
}

int CUDA_CALLABLE grid_index(ivec3 const &index, ivec3 const &dim) {
  return grid_index(index.x, index.y, index.z, dim.x, dim.y, dim.z);
}
} // namespace maniskill
