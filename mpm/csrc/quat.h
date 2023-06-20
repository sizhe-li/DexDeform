#pragma once
#include "vec3.h"

namespace maniskill {
struct quat {
  inline CUDA_CALLABLE quat() = default;
  inline CUDA_CALLABLE quat(float _w, float _x, float _y, float _z)
      : w(_w), x(_x), y(_y), z(_z) {}
  float w, x, y, z;

  inline CUDA_CALLABLE quat inv() const { return quat(w, -x, -y, -z); }
};

inline CUDA_CALLABLE vec3 operator*(quat const &q, vec3 const &v) {
  vec3 qvec(q.x, q.y, q.z);
  vec3 uv = cross(qvec, v);
  vec3 uuv = cross(qvec, uv);
  return v + 2.f * (q.w * uv + uuv);
}

inline CUDA_CALLABLE quat operator + (quat const &q, quat const &v) {
  return quat(q.w + v.w, q.x + v.x, q.y + v.y, q.z + v.z);
}

inline CUDA_CALLABLE void qmul_backward_inplace(
  quat const &q, vec3 const &v, vec3 const &grad_out,
  quat &grad_q, vec3 &grad_v) {
  // d(a x b) = dx x b - db x a
  vec3 qvec(q.x, q.y, q.z);
  vec3 uv = cross(qvec, v);
  vec3 uuv = cross(qvec, uv);

  grad_q.w += dot(uv, grad_out) * 2.f;
  vec3 grad_uv = 2.f * q.w * grad_out;
  vec3 grad_uuv = 2.f * grad_out;

  // gradient of cross(inp1, inp2) is cross(inp2, grad_out), cross(grad_out, inp1)
  vec3 grad_qvec = cross(uv, grad_uuv);
  grad_uv = grad_uv + cross(grad_uuv, qvec);

  grad_qvec += cross(v, grad_uv);
  grad_v += grad_out + cross(grad_uv, qvec);

  grad_q.x += grad_qvec.x;
  grad_q.y += grad_qvec.y;
  grad_q.z += grad_qvec.z;
}


inline CUDA_CALLABLE void spatial_transform_backward_inplace(
  vec3 const&p, quat const &q, vec3 const &point, vec3 const &grad_out,
  vec3 &grad_p, quat &grad_q, vec3 &grad_point) {
  grad_p += grad_out;
  qmul_backward_inplace(q, point, grad_out, grad_q, grad_point);

  /*
  quat tmp_q(0.f, 0.f, 0.f, 0.f);
  vec3 tmp_p(0.f);
  qmul_backward_inplace(q, vec3(1.f, 2.f, 3.f), vec3(4.f, 5.f, 6.f), tmp_q, tmp_p);
  printf("%f %f %f %f %f %f %f %f %f %f %f\n", q.w, q.x, q.y, q.z, tmp_q.w, tmp_q.x, tmp_q.y, tmp_q.z, tmp_p.x, tmp_p.y, tmp_p.z);
  */
}

inline CUDA_CALLABLE void inv_spatial_transform_backward_inplace(
  //verified with pytorch! 
  vec3 const&p, quat const &q, vec3 const &point, vec3 const &grad_out,
  vec3 &grad_p, quat &grad_q, vec3& grad_point) {
  quat tmp_q(0.f, 0.f, 0.f, 0.f);
  vec3 tmp_p(0.f);
  //printf("p %f %f %f\n", p.x, p.y, p.z);
  //printf("q %f %f %f %f\n", q.w, q.x, q.y, q.z);
  //printf("point %f %f %f\n", point.x, point.y, point.z);
  qmul_backward_inplace(q.inv(), point-p, grad_out, tmp_q, tmp_p);
  //printf("inv spatial, %f %f %f %f\n", tmp_q.inv().w, tmp_q.inv().x, tmp_q.inv().y, tmp_q.inv().z);
  //printf("inv spatial, %f %f %f\n", -tmp_p.x, -tmp_p.y, -tmp_p.z);
  //printf("grad %f %f %f\n", grad_out.x, grad_out.y, grad_out.z);
  grad_q = grad_q + tmp_q.inv();
  grad_p = grad_p - tmp_p;
  grad_point = grad_point + tmp_p;
}

inline __device__ void atomicAdd(quat *a, quat const &b) {
  ::atomicAdd(&a->w, b.w);
  ::atomicAdd(&a->x, b.x);
  ::atomicAdd(&a->y, b.y);
  ::atomicAdd(&a->z, b.z);
}

inline CUDA_CALLABLE vec3 spatial_transform(vec3 const &p, quat const &q,
                                            vec3 const &point) {
  return p + (q * point);
}

inline CUDA_CALLABLE vec3 spatial_transform_inv(vec3 const &p, quat const &q,
                                                vec3 const &point) {
  return q.inv() * (point - p);
}

} // namespace maniskill
