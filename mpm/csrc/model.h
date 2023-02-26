#pragma once
#include "mat3.h"

namespace maniskill {

  struct Model {
    float *p_m;
    float *p_vol;
    vec3 *p_mu_lam_yield;
    int particle_count;
    ivec3 grid_dim;
    float dx;
    float inv_dx;
    float dt;
  };

  struct State {
    ivec3 *grid_lower;
    vec3 *p_x;
    vec3 *p_v;
    vec3 *p_f;
    mat3 *p_F;
    mat3 *p_C;
    float *g_m;
    vec3 *g_mv;
    vec3 *g_v;
  };

};
