#pragma once
#include "vec3.h"

namespace maniskill {

template <class T>
inline CUDA_CALLABLE bool volume_query(vec3 const &point, T *__restrict__ values, vec3 const &lower, vec3 const &upper, ivec3 const &dim, T& out) {

  vec3 ratio = (point - lower) / (upper - lower);
  if(!(ratio.x>=0 && ratio.y>=0 && ratio.z>=0 && ratio.x<1. && ratio.y<1. && ratio.z<1.)){
       return false;
  }
  vec3 grid = ratio * cast_float(dim);
  ivec3 xyz0 = clamp(cast_int(grid), ivec3(0), dim - ivec3(1));
  vec3 dxyz = grid - cast_float(xyz0);
  int x0 = xyz0.x;
  int y0 = xyz0.y;
  int z0 = xyz0.z;
  float dx = dxyz.x;
  float dy = dxyz.y;
  float dz = dxyz.z;

  int x0y0z0 = (x0 * dim.y + y0) * dim.z + z0;
  int fz = z0 + 1 < dim.z;
  int fy = y0 + 1 < dim.y;
  int fx = x0 + 1 < dim.x;
  int x0y0z1 = x0y0z0 + fz;
  int x0y1z0 = x0y0z0 + dim.z * fy;
  int x0y1z1 = x0y1z0 + fz;
  int x1y0z0 = x0y0z0 + dim.y * dim.z * fx;
  int x1y0z1 = x1y0z0 + fz;
  int x1y1z0 = x1y0z0 + dim.z * fy;
  int x1y1z1 = x1y1z0 + fz;

  T vx0y0z0 = values[x0y0z0] * (1 - dx) * (1 - dy) * (1 - dz);
  T vx0y0z1 = values[x0y0z1] * (1 - dx) * (1 - dy) * dz;
  T vx0y1z0 = values[x0y1z0] * (1 - dx) * dy * (1 - dz);
  T vx0y1z1 = values[x0y1z1] * (1 - dx) * dy * dz;
  T vx1y0z0 = values[x1y0z0] * dx * (1 - dy) * (1 - dz);
  T vx1y0z1 = values[x1y0z1] * dx * (1 - dy) * dz;
  T vx1y1z0 = values[x1y1z0] * dx * dy * (1 - dz);
  T vx1y1z1 = values[x1y1z1] * dx * dy * dz;
  out = vx0y0z0 + vx0y0z1 + vx0y1z0 + vx0y1z1 + vx1y0z0 + vx1y0z1 + vx1y1z0 + vx1y1z1;

  //if(point.x < 0.3 && point.y > 0.6){
  //  printf("%f %f %f %f %d %d %d %d %d %d\n", point.x, point.y, point.z, values[(x0*dim.y+y0)*dim.z+z0], x0, y0, z0, dim.x, dim.y, dim.z);
  //}
  return true;
}

inline CUDA_CALLABLE float sample_sdf(float *__restrict__ sdf_volume, vec3 *__restrict__ box_min, vec3 *__restrict__ box_max, ivec3 const &dim, const vec3 &point, float sdf_threshold){
  float out = 0.;
  bool inside = volume_query<float>(point, sdf_volume, box_min[0], box_max[0], dim, out);
  if(inside){
    out -= sdf_threshold;
  }
  return out;
}


inline CUDA_CALLABLE vec3 sample_normal(float *__restrict__ sdf_volume, vec3 *__restrict__ box_min, vec3 *__restrict__ box_max, ivec3 const &dim, const vec3 &point){
  float eps = 1e-3;
  float x0 = sample_sdf(sdf_volume, box_min, box_max, dim, point+vec3(-eps, 0.f, 0.f), 0.);
  float x1 = sample_sdf(sdf_volume, box_min, box_max, dim, point+vec3(eps, 0.f, 0.f), 0.);
  float y0 = sample_sdf(sdf_volume, box_min, box_max, dim, point+vec3(0.f, -eps, 0.f), 0.);
  float y1 = sample_sdf(sdf_volume, box_min, box_max, dim, point+vec3(0.f, eps, 0.f), 0.);
  float z0 = sample_sdf(sdf_volume, box_min, box_max, dim, point+vec3(0.f, 0.f, -eps), 0.);
  float z1 = sample_sdf(sdf_volume, box_min, box_max, dim, point+vec3(0.f, 0.f, eps), 0.);
  return vec3(x1-x0, y1-y0, z1-z0) * (0.5/eps);
}

inline CUDA_CALLABLE vec3 sample_color(vec3 *__restrict__ sdf_color, vec3 *__restrict__ box_min, vec3 *__restrict__ box_max, ivec3 const &dim, const vec3 &point){
  vec3 color(0.);
  volume_query<vec3>(point, sdf_color, box_min[0], box_max[0], dim, color);
  return color;
}

} // namespace maniskill
