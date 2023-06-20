#include "quat.h"
#include "svd.h"
#include "shape.h"
#include "array.h"
#include <assert.h>
#include <cstdio>

#include <curand.h>
#include <curand_kernel.h>

#define NORM_EPS 1e-8
#define SIG_CLIP_EPS 0.05

using namespace maniskill;

//// TODO
// inline CUDA_CALLABLE void print_mat(mat3 m) {
//	printf("%f %f %f %f %f %f %f %f %f\n",
//			m.data[0][0], m.data[0][1], m.data[0][2],
//			m.data[1][0], m.data[1][1], m.data[1][2],
//			m.data[2][0], m.data[2][1], m.data[2][2]);
//}
//
// inline CUDA_CALLABLE void print_vec(vec3 v) {
//	printf("%f %f %f\n", v.x, v.y, v.z);
//}

inline CUDA_CALLABLE float norm(const vec3 &a, float eps = NORM_EPS)
{
  return sqrt(dot(a, a) + eps);
}

inline CUDA_CALLABLE vec3 dw(mat3 const &weights0, mat3 const &weights1, const int &i, const int &j, const int &k)
{
  // https://github.com/yuanming-hu/ChainQueen/blob/0bdda869d66b483dc85b8966e4d5f2b8200021e9/src/state.cuh#L461
  return vec3(
      weights1.data[0][i] * weights0.data[1][j] * weights0.data[2][k],
      weights0.data[0][i] * weights1.data[1][j] * weights0.data[2][k],
      weights0.data[0][i] * weights0.data[1][j] * weights1.data[2][k]);
}

inline CUDA_CALLABLE float compute_von_mises(mat3 const &F, mat3 const &U,
                                             vec3 const &s, mat3 const &V,
                                             float yield_stress, float mu,
                                             mat3 &out_F)
{
  vec3 s_new = max(s, SIG_CLIP_EPS);
  vec3 epsilon = log(s_new);
  vec3 epsilon_hat = epsilon - (epsilon.x + epsilon.y + epsilon.z) / 3.f;
  float epsilon_hat_norm = norm(epsilon_hat);
  // yield_stress = 0.;
  float delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu);
  // printf("%f %f %f %f %f\n", mu, delta_gamma, epsilon_hat.x, epsilon_hat.y, epsilon_hat.z);
  if (delta_gamma > 0.f)
  {
    epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat;

    vec3 exp_eps = exp(epsilon);
    out_F = U.mul(mat3(exp_eps)).mul(V.transpose());
    return exp_eps.x * exp_eps.y * exp_eps.z;
  }
  else
  {
    out_F = F;
    return s.x * s.y * s.z;
  }
}


__global__ void compute_grid_lower(vec3 *__restrict__ particle_x, float dx,
                                   float inv_dx, ivec3 *__restrict__ grid_lower,
                                   int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    vec3 x = particle_x[p];
    // compute dynamic grid lower left corner
    ivec3 x2 = cast_int((x - dx * 10.f) * inv_dx);
    atomicMin(&grid_lower[0], x2);
  }
}

__global__ void compute_svd(
    mat3 *__restrict__ F, mat3 *__restrict__ C, mat3 *__restrict__ newF, mat3 *__restrict__ U, mat3 *__restrict__ V, vec3 *__restrict__ sig, float dt, int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    mat3 _F = (mat3(1.f) + C[p] * dt).mul(F[p]);
    mat3 _U(0.f);
    mat3 _V(0.f);
    vec3 _sigma(0.f);
    svd3(_F, _U, _sigma, _V); // TODO: maybe we can remove the temporay variables
    newF[p] = _F;
    U[p] = _U;
    V[p] = _V;
    sig[p] = _sigma;
  }
}

inline CUDA_CALLABLE float clamp(float a, float eps = 1e-6)
{
  if (a >= 0.)
    return max(a, eps);
  else
    return min(a, -eps);
}

__global__ void compute_svd_grad(
    mat3 *__restrict__ F,
    mat3 *__restrict__ C,

    mat3 *__restrict__ U,
    mat3 *__restrict__ V,
    vec3 *__restrict__ sig,

    mat3 *__restrict__ newF_grad,
    mat3 *__restrict__ U_grad,
    mat3 *__restrict__ V_grad,
    vec3 *__restrict__ sig_grad,

    mat3 *__restrict__ F_grad,
    mat3 *__restrict__ C_grad,
    float dt,
    int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    mat3 u = U[p], v = V[p], gu = U_grad[p], gv = V_grad[p];
    vec3 sigma = sig[p];
    vec3 gsigma_ = sig_grad[p];

    mat3 sig = mat3(sigma);
    mat3 gsigma = mat3(gsigma_);

    mat3 vt = v.transpose();
    mat3 ut = u.transpose();
    mat3 sigma_term = u.mul(gsigma).mul(vt);

    // vec3 sigma2 = pow2(sigma);
    // float s0 = pow2(sigma.x);
    // float s1 = pow2(sigma.y);
    // float s2 = pow2(sigma.z);
    double s0 = sigma.x, s1 = sigma.y, s2 = sigma.z;
    s0 = s0 * s0;
    s1 = s1 * s1;
    s2 = s2 * s2;
    mat3 FF(
        0., 1.0 / clamp(s1 - s0), 1.0 / clamp(s2 - s0),
        1.0 / clamp(s0 - s1), 0., 1.0 / clamp(s2 - s1),
        1.0 / clamp(s0 - s2), 1.0 / clamp(s1 - s2), 0.);


    mat3 u_term = u.mul((FF * (ut.mul(gu) - gu.transpose().mul(u))).mul(sig)).mul(vt);
    mat3 v_term = u.mul(sig.mul((FF * (vt.mul(gv) - gv.transpose().mul(v))).mul(vt)));

    mat3 F_tmp_grad = newF_grad[p] + u_term + sigma_term + v_term;

    /*
    if(p==190){
      FF.show();
      printf("ShowFF\n");

      newF_grad[p].show();
      printf("before\n");
      u_term.show();
      printf("u\n");
      v_term.show();
      printf("v\n");
      sigma_term.show();
      printf("sig\n");
      F_tmp_grad.show();
      printf("F_tmp_grad\n");
    }
    */

    newF_grad[p] = F_tmp_grad; //We do not care about newF's grad anymore
    // Y=XW -> dL/dX = dL/dY W^T
    // (mat3(1.f) + C[p] * dt).mul(particle_F[p])
    C_grad[p] += dt * F_tmp_grad.mul(F[p].transpose());
    // Y^T=W^TX^T -> dL/d(W^T) = (dL/dY)^TX  -> dL/dW = X^T(dL/dY)
    F_grad[p] += (mat3(1.f) + dt * C[p]).transpose().mul(F_tmp_grad);
  }
}

__global__ void compute_dist(
    vec3 *__restrict__ particle_x,
    vec3 *__restrict__ body_pos,
    quat *__restrict__ body_rot,
    quat *__restrict__ body_type_friction_softness_round,
    quat *__restrict__ body_args, // describe geometry shape besides round ..
    float *__restrict__ dist,
    int n_bodies,
    vec3 *__restrict__ particle_x_grad,
    vec3 *__restrict__ body_pos_grad,
    quat *__restrict__ body_rot_grad,
    float *__restrict__ dist_grad,
    int compute_grad,
    int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    for (int body_id = 0; body_id < n_bodies; ++body_id)
    {
      vec3 bx = body_pos[body_id];
      quat bq = body_rot[body_id];
      quat type_friction_softness_round = body_type_friction_softness_round[body_id];
      quat shape_args = body_args[body_id];
      vec3 gx_b = spatial_transform_inv(bx, bq, particle_x[p]); // grid x in body frame
      if (compute_grad == 0)
      {
        dist[p * n_bodies + body_id] = shape_sdf(type_friction_softness_round, shape_args, gx_b);
      }
      else
      {
        //printf("gradient is not checked!\n");
        float grad_dist = dist_grad[p * n_bodies + body_id];
        vec3 unnormed_normal = shape_grad(type_friction_softness_round, shape_args, gx_b);
        vec3 grad_gx_b = unnormed_normal * grad_dist;

        vec3 grad_bx(0.f), grad_gx(0.f);
        quat grad_bq(0.f, 0.f, 0.f, 0.f);
        inv_spatial_transform_backward_inplace(bx, bq, particle_x[p],
                                               grad_gx_b, grad_bx, grad_bq, grad_gx);
        //if(dot(unnormed_normal, unnormed_normal) > 1e-5){
        //  printf("cuda %d %f %f %f %f\n", p, grad_dist, grad_gx.x, grad_gx.y, grad_gx.z);
        //}
        particle_x_grad[p] += grad_gx;
        atomicAdd(&body_pos_grad[body_id], grad_bx);
        atomicAdd(&body_rot_grad[body_id], grad_bq);
      }
    }
  }
}

__global__ void
particle2mass(
    vec3 *__restrict__ particle_x,
    float *__restrict__ particle_m,

    ivec3 *__restrict__ grid_lower,
    ivec3 grid_dim,
    float dx,
    float inv_dx,
    float *__restrict__ out_grid_m,

    float *__restrict__ out_grid_m_grad,
    vec3 *__restrict__ particle_x_grad,

    int *__restrict__ particle_ids,
    int id,

    int compute_grad,
    int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    if (id != -1 && particle_ids[p] != id)
      return;

    vec3 x = particle_x[p];
    x = x - cast_float(grid_lower[0]) * dx;
    ivec3 base = cast_int(x * inv_dx - 0.5);
    vec3 fx = x * inv_dx - cast_float(base);
    mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f),
                  0.5f * pow2(fx - 0.5f));
    if (compute_grad)
    {
      //printf("Gradient of compute M is not implemented!");
      mat3 w1 = mat3(-inv_dx * (1.5f - fx), inv_dx * ((-2.f) * fx + 2.0f), -inv_dx * (fx * (-1.f) + 0.5f));
      vec3 grad_x = vec3(0.);
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          for (int k = 0; k < 3; ++k)
          {
            ivec3 i3 = base + ivec3(i, j, k);
            int index = grid_index(i3, grid_dim);
            auto grad_m = out_grid_m_grad[index];
            auto grad_N = dw(w, w1, i, j, k);          // magic function from chainqueen's code
            grad_x += grad_m * particle_m[p] * grad_N; // mN term
          }
        }
      }
      particle_x_grad[p] += grad_x;
    }
    else
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          for (int k = 0; k < 3; ++k)
          {
            float weight = w(0, i) * w(1, j) * w(2, k);
            vec3 dpos = (vec3(i, j, k) - fx) * dx;
            ivec3 i3 = base + ivec3(i, j, k);
            int index = grid_index(i3, grid_dim);
            atomicAdd(&out_grid_m[index], particle_m[p] * weight);
          }
        }
      }
    }
  }
}

__global__ void
p2g(
    vec3 *__restrict__ particle_x,
    vec3 *__restrict__ particle_v,
    float *__restrict__ particle_m,
    float *__restrict__ particle_vol,
    mat3 *__restrict__ particle_F,

    mat3 *__restrict__ particle_U,
    vec3 *__restrict__ particle_sig,
    mat3 *__restrict__ particle_V,
    mat3 *__restrict__ particle_C,
    vec3 *__restrict__ particle_mu_lam_yield,

    ivec3 *__restrict__ grid_lower,

    ivec3 grid_dim,
    float dx,
    float inv_dx,
    float dt,
    mat3 *__restrict__ out_particle_F,
    vec3 *__restrict__ out_grid_mv,
    float *__restrict__ out_grid_m,
    int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    vec3 x = particle_x[p];
    x = x - cast_float(grid_lower[0]) * dx;

    mat3 U = particle_U[p];
    mat3 V = particle_V[p];
    vec3 sigma = particle_sig[p];
    mat3 F_tmp = particle_F[p];

    ivec3 base = cast_int(x * inv_dx - 0.5);
    vec3 fx = x * inv_dx - cast_float(base);

    mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f),
                  0.5f * pow2(fx - 0.5f));

    mat3 new_F;

    float J = compute_von_mises(
        F_tmp, U, sigma, V,
        particle_mu_lam_yield[p].z,
        particle_mu_lam_yield[p].x,
        new_F);
    out_particle_F[p] = new_F;

    // float J = new_F.determinant();
    mat3 r = U.mul(V.transpose());
    // printf("det from svd and determinant %f %f\n", J, J2);

    float mu = particle_mu_lam_yield[p].x;
    float lam = particle_mu_lam_yield[p].y;
    mat3 stress = 2.f * mu * (new_F - r).mul(new_F.transpose()) + mat3(lam * J * (J - 1));

    stress = -dt * particle_vol[p] * 4.f * inv_dx * inv_dx * stress;

    mat3 affine = stress + particle_m[p] * particle_C[p];

    vec3 mv = particle_m[p] * particle_v[p];

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          float weight = w(0, i) * w(1, j) * w(2, k);
          vec3 dpos = (vec3(i, j, k) - fx) * dx;
          ivec3 i3 = base + ivec3(i, j, k);
          int index = grid_index(i3, grid_dim);

          atomicAdd(&out_grid_m[index], particle_m[p] * weight);
          atomicAdd(&out_grid_mv[index], (mv + affine.mul(dpos)) * weight);
        }
      }
    }
  }
}

__global__ void
p2g_grad(
    vec3 *__restrict__ particle_x,
    vec3 *__restrict__ particle_v,
    float *__restrict__ particle_m,
    float *__restrict__ particle_vol,
    mat3 *__restrict__ particle_F,

    mat3 *__restrict__ particle_U,
    vec3 *__restrict__ particle_sig,
    mat3 *__restrict__ particle_V,
    mat3 *__restrict__ particle_C,
    vec3 *__restrict__ particle_mu_lam_yield,

    ivec3 *__restrict__ grid_lower,

    ivec3 grid_dim,
    float dx,
    float inv_dx,
    float dt,

    mat3 *__restrict__ out_particle_F,
    vec3 *__restrict__ out_grid_v,
    float *__restrict__ out_grid_m,

    vec3 *__restrict__ particle_x_grad,
    vec3 *__restrict__ particle_v_grad,
    mat3 *__restrict__ particle_F_grad,
    mat3 *__restrict__ particle_C_grad,

    mat3 *__restrict__ particle_U_grad,
    vec3 *__restrict__ particle_sig_grad,
    mat3 *__restrict__ particle_V_grad,

    mat3 *__restrict__ out_particle_F_grad,
    vec3 *__restrict__ out_grid_v_grad,
    float *__restrict__ out_grid_m_grad,

    int dim)
{
  // https://github.com/yuanming-hu/ChainQueen/blob/master/src/backward.cu
  int p = get_tid();
  if (p < dim)
  {
    mat3 U = particle_U[p];
    mat3 V = particle_V[p];
    vec3 sigma = particle_sig[p];

    float mu = particle_mu_lam_yield[p].x;
    float lam = particle_mu_lam_yield[p].y;
    float yield_stress = particle_mu_lam_yield[p].z;

    // float J = compute_von_mises(F_tmp, U, sigma, V, particle_mu_lam_yield[p].z,
    //                   particle_mu_lam_yield[p].x, new_F);
    float J;
    mat3 new_F;

    // It's possible to reduce the forward passing here. But I assume the time complexity here is negligible compared with AtomicAdd.
    vec3 s_new = max(sigma, SIG_CLIP_EPS);
    vec3 epsilon = log(s_new);
    vec3 epsilon_hat = epsilon - (epsilon.x + epsilon.y + epsilon.z) / 3.f;
    float epsilon_hat_norm = norm(epsilon_hat);
    float delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu);
    if (delta_gamma > 0.f)
    {
      vec3 exp_eps = exp(epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat);
      new_F = U.mul(mat3(exp_eps)).mul(V.transpose());
      J = exp_eps.x * exp_eps.y * exp_eps.z;
    }
    else
    {
      new_F = particle_F[p];
      J = sigma.x * sigma.y * sigma.z;
    }

    mat3 r = U.mul(V.transpose());

    float grad_stress_scale = -dt * inv_dx * particle_vol[p] * 4.f * inv_dx;
    mat3 stress = 2.f * mu * (new_F - r).mul(new_F.transpose()) + mat3(lam * J * (J - 1));
    mat3 affine = grad_stress_scale * stress + particle_m[p] * particle_C[p];

    // let's start here
    // (A) and (B) we do it in g2p part, ignore it now.

    // (C) compute grad P
    float m_p = particle_m[p];
    vec3 v_p = particle_v[p];

    vec3 x = particle_x[p];
    x = x - cast_float(grid_lower[0]) * dx;

    ivec3 base = cast_int(x * inv_dx - 0.5);
    vec3 fx = x * inv_dx - cast_float(base);

    // https://github.com/yuanming-hu/ChainQueen/blob/0bdda869d66b483dc85b8966e4d5f2b8200021e9/src/state.cuh#L429
    // note that their fx is the inverse fx
    mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f), 0.5f * pow2(fx - 0.5f));
    mat3 w1 = mat3(-inv_dx * (1.5f - fx), inv_dx * ((-2.f) * fx + 2.0f), -inv_dx * (fx * (-1.f) + 0.5f));

    mat3 grad_stress = mat3(0.), grad_C = mat3(0.);
    vec3 grad_x = vec3(0.), grad_v = vec3(0.);

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          ivec3 i3 = base + ivec3(i, j, k);
          int index = grid_index(i3, grid_dim);

          float N = w(0, i) * w(1, j) * w(2, k); // N in chainqueen's code
          vec3 dpos = (vec3(i, j, k) - fx) * dx;
          auto out_grad_v = out_grid_v_grad[index]; // grad velocity grad, I guess

          auto grad_N = dw(w, w1, i, j, k); // magic function from chainqueen's code
          auto grad_dpos = -1;              // -fx * dx =  - x * inv_dx * dx = -1

          // atomicAdd(&out_grid_m[index], particle_m[p] * N);
          //  affine = grad_P_scale * P + m * C
          // atomicAdd(&out_grid_mv[index], (mv + affine.mul(dpos)) * N);
          // mat3 tmp = outer(out_grad_v, dpos);
          mat3 tmp = outer(out_grad_v, dpos); // grad dpos^T
          grad_stress += N * grad_stress_scale * tmp;
          grad_C += N * m_p * tmp;

          auto grad_m = out_grid_m_grad[index];
          // mv * N
          grad_v += N * m_p * out_grad_v;
          // m_p * N; dN/dx d(m_p*N)/dN dgrad_m/d(m_p*N)
          grad_x += grad_m * m_p * grad_N;               // mN term
          grad_x += dot(v_p, out_grad_v) * m_p * grad_N; // mvN term
          grad_x += grad_dpos * N * affine.transpose().mul(out_grad_v) + dot(affine.mul(dpos), out_grad_v) * grad_N;
        }
      }
    }

    particle_x_grad[p] += grad_x;
    particle_v_grad[p] += grad_v;
    particle_C_grad[p] += grad_C;

    // NOTE that we do not compute grad to affine, instead, we follow the method in chainqueue, to copute the grad to C, stress (before scaling) direclty.
    //  mat3 stress =2.f * mu * (new_F - r).mul(new_F.transpose()) + mat3(lam * J * (J - 1));
    //  gradient to new_F, J anad r
    //  d(AA^T)/dA = A dA^T + A dA? right?
    mat3 grad_r = -2.f * mu * grad_stress.mul(new_F);
    mat3 grad_U = grad_r.mul(V);             //.mul(grad_r);
    mat3 grad_V = grad_r.transpose().mul(U); // UV^T
    mat3 grad_new_F = out_particle_F_grad[p] + 2.f * mu * (grad_stress.transpose().mul(new_F - r) + grad_stress.mul(new_F));
    //if(p==402){
    //  printf("grad_newF\n");
    //  grad_new_F.show();
    //}

    float grad_J = ((2 * J - 1) * lam) * (grad_stress(0, 0) + grad_stress(1, 1) + grad_stress(2, 2));

    vec3 grad_sigma(0.);
    mat3 grad_F(0.);
    // von mises grad
    /*
    vec3 s_new = max(sigma, 0.05);
    vec3 epsilon = log(s_new);
    vec3 epsilon_hat = epsilon - (epsilon.x + epsilon.y + epsilon.z) / 3.f;
    float epsilon_hat_norm = norm(epsilon_hat);
    float delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu);
    */
    if (delta_gamma > 0.f)
    {
      // recompute it; should be fine.
      vec3 exp_eps = exp(epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat);
      mat3 exp_eps_mat3 = mat3(exp_eps);
      // new_F = U.mul(mat3(exp_eps)).mul(V.transpose());
      // J = exp_eps.x * exp_eps.y * exp_eps.z;
      grad_U += grad_new_F.mul(V).mul(exp_eps_mat3);
      grad_V += grad_new_F.transpose().mul(U).mul(exp_eps_mat3);
      // mat3 xxx = U.transpose().mul(grad_new_F).mul(V);
      vec3 Fpart = U.transpose().mul(grad_new_F).mul(V).diag(); // dL/d(xV^T) = U^T dL/dF   dL/dx=U^TdL/dFV
      // if(p==14955){U.show();V.show();printf("grad J%f\n", grad_J);grad_new_F.show();}
      vec3 Jpart(
          grad_J * exp_eps.y * exp_eps.z,
          grad_J * exp_eps.x * exp_eps.z,
          grad_J * exp_eps.x * exp_eps.y);
      // vec3 grad_epsilon = exp_eps * (Jpart + Fpart); //gradient of exp_eps
      vec3 grad_epsilon = exp_eps * (Jpart + Fpart); // gradient of exp_eps

      // the part below is verified by pytorch!
      // -(1.-yield_stress/(2*mu)/epsilon_hat_norm) * epsilon_hat
      vec3 grad_epsilon_hat = -delta_gamma / epsilon_hat_norm * grad_epsilon;

      double grad_epsilon_hat_norm = -dot(epsilon_hat / epsilon_hat_norm, grad_epsilon) * (yield_stress / (2 * mu)) / epsilon_hat_norm;
      grad_epsilon_hat += epsilon_hat / epsilon_hat_norm * grad_epsilon_hat_norm;

      grad_epsilon += grad_epsilon_hat - sum(grad_epsilon_hat) / 3.;
      /*
      if(p==14955){
        printf("in kernel: %f %f %f\n", grad_epsilon.x, grad_epsilon.y, grad_epsilon.z);
        printf("sigma: %f %f %f\n", sigma.x, sigma.y, sigma.z);
        printf("in kernel: %f %f %f\n", exp_eps.x, exp_eps.y, exp_eps.z);
      }
      */
      if (sigma.x >= SIG_CLIP_EPS)
      {
        grad_sigma.x += grad_epsilon.x / sigma.x;
      }
      if (sigma.y >= SIG_CLIP_EPS)
      {
        grad_sigma.y += grad_epsilon.y / sigma.y;
      }
      if (sigma.z >= SIG_CLIP_EPS)
      {
        grad_sigma.z += grad_epsilon.z / sigma.z;
      }
    }
    else
    {
      grad_sigma += vec3(grad_J * sigma.y * sigma.z, grad_J * sigma.x * sigma.z, grad_J * sigma.x * sigma.y);

      /*
      if(p==402){
        printf("cuda: %d %f %f %f %f %f\n", p, delta_gamma, grad_J*100, grad_J * sigma.y * sigma.z, grad_J * sigma.x * sigma.y, grad_J * sigma.x * sigma.y);
      }
      */
      // printf("%f %f %f\n", grad_J * sigma.y * sigma.z, grad_J * sigma.x * sigma.y, grad_J * sigma.x * sigma.y);
      grad_F += grad_new_F;
    }

    particle_U_grad[p] += grad_U;
    particle_V_grad[p] += grad_V;
    particle_sig_grad[p] += grad_sigma;
    particle_F_grad[p] += grad_F;
  }
}

/*
__global__ void compute_bbox(
  vec3 *__restrict__ particle_x,
  ivec3 *__restrict__ box_min,
  ivec3 *__restrict__ box_max,
  float inv_dx,
  int dim) {
  int p = get_tid();
  if (p < dim) {
    // compute dynamic grid lower left corner
    vec3 x = particle_x[p];
    ivec3 y = cast_int(x * inv_dx - 6.0);
    atomicMin(&box_min[0], y);
    atomicMax(&box_max[0], y);
  }
}
*/

__global__ void grid_op_v2(
    float *__restrict__ grid_m,
    vec3 *__restrict__ grid_v_in,
    vec3 *__restrict__ grid_body_v_in,
    ivec3 *grid_lower,
    vec3 *gravity,
    // rigid body
    vec3 *__restrict__ body_pos,
    quat *__restrict__ body_rot,

    vec3 *__restrict__ next_body_pos,
    quat *__restrict__ next_body_rot,

    quat *__restrict__ body_type_friction_softness_round,
    quat *__restrict__ body_args, // describe geometry shape besides round ..

    float dx, float inv_dx, float dt,
    float ground_friction, // ground friction ..

    vec3 *__restrict__ out_grid_v,
    ivec3 grid_dim, int n_bodies,
    int dim)
{
  int tid = get_tid();
  if (tid < dim && grid_m[tid] > 1e-12)
  {
    int grid_x = tid / grid_dim.z / grid_dim.y;
    int grid_y = (tid / grid_dim.z) % grid_dim.y;
    int grid_z = tid % grid_dim.z;

    // TODO: here linear layout assumed...
    float m = grid_m[tid];
    vec3 mv = grid_v_in[tid];
    vec3 v_out = mv * (1.f / m);
    v_out += dt * gravity[0];

    vec3 lower = cast_float(grid_lower[0]) * dx;
    vec3 gx = lower + vec3(grid_x, grid_y, grid_z) * dx;

    for (int body_id = 0; body_id < n_bodies; ++body_id)
    {
      grid_body_v_in[tid * (n_bodies + 1) + body_id] = v_out;

      //if(grid_x == 34 && grid_y == 2 && grid_z == 29){
      //  printf("%d %f %f %f\n", body_id, v_out.x, v_out.y, v_out.z);
      //}

      vec3 bx = body_pos[body_id];
      quat bq = body_rot[body_id];
      quat type_friction_softness_round = body_type_friction_softness_round[body_id];
      quat shape_args = body_args[body_id];

      // if(tid ==0) printf("bx %f %f %f %f %f %f %f\n", bx.x, bx.y, bx.z, bq.w, bq.x, bq.y, bq.z);

      float friction = type_friction_softness_round.x;
      float softness = type_friction_softness_round.y;
      // if(tid ==0) printf("fric %f soft %f round %f\n", friction, softness, type_friction_softness_round.z);

      vec3 gx_b = spatial_transform_inv(bx, bq, gx); // grid x in body frame

      float dist = shape_sdf(type_friction_softness_round, shape_args, gx_b);
      float influence = min(exp(-dist * softness), 1.);

      if ((softness > 0 && influence > 0.1) || dist <= 0)
      {

        //printf("forward collision %d %f %f %f\n", body_id, v_out.x, v_out.y, v_out.z);
        vec3 normal = normalized(shape_grad(type_friction_softness_round, shape_args, gx_b));
        normal = bq * normal; // normal in world frame
        vec3 bv = (spatial_transform(next_body_pos[body_id], next_body_rot[body_id], gx_b) - gx) / dt;
        // rel_v is the input v
        vec3 rel_v = v_out - bv;

        float normal_component = dot(rel_v, normal);
        vec3 grid_v_t = rel_v - min(normal_component, 0.f) * normal;

        if (normal_component < 0. && dot(grid_v_t, grid_v_t) > 1e-30)
        {
          // apply friction
          float grid_v_t_norm = length(grid_v_t);
          grid_v_t = grid_v_t * (1.f / grid_v_t_norm) * max(0.f, grid_v_t_norm + normal_component * friction);
        }
        v_out = bv + rel_v * (1 - influence) + grid_v_t * influence;
      }
    }
    grid_body_v_in[tid * (n_bodies + 1) + n_bodies] = v_out;

    // boundary condition
    const int bound = 3;

    if (grid_x < bound && v_out.x < 0)
    {
      v_out.x = 0;
    }
    if (grid_x > grid_dim.x - bound && v_out.x > 0)
    {
      v_out.x = 0;
    }
    if (grid_y < bound && v_out.y < 0)
    {
      if (ground_friction > 0.f)
      {
        if (ground_friction < 99.f)
        {
          float lin = v_out.y;
          vec3 vit(v_out.x, 0., v_out.z);
          float lit = norm(vit);
          v_out = vit * fmaxf(1. + ground_friction * lin / lit, 0.f);
        }
        else
        {
          v_out = vec3(0.);
        }
      }
      v_out.y = 0;
    }
    if (grid_y > grid_dim.y - bound && v_out.y > 0)
    {
      v_out.y = 0;
    }
    if (grid_z < bound && v_out.z < 0)
    {
      v_out.z = 0;
    }
    if (grid_z > grid_dim.z - bound && v_out.z > 0)
    {
      v_out.z = 0;
    }
    out_grid_v[tid] = v_out;
  }
}

__global__ void grid_op_v2_grad(

    float *__restrict__ grid_m,
    vec3 *__restrict__ grid_v_in,
    vec3 *__restrict__ grid_body_v_in,
    ivec3 *grid_lower,
    vec3 *gravity,
    // rigid body
    vec3 *__restrict__ body_pos,
    quat *__restrict__ body_rot,

    vec3 *__restrict__ next_body_pos,
    quat *__restrict__ next_body_rot,

    quat *__restrict__ body_type_friction_softness_round,
    quat *__restrict__ body_args, // describe geometry shape besides round ..

    float *__restrict__ grid_m_grad,
    vec3 *__restrict__ grid_v_in_grad,
    vec3 *__restrict__ body_pos_grad,
    quat *__restrict__ body_rot_grad,

    vec3 *__restrict__ next_body_pos_grad,
    quat *__restrict__ next_body_rot_grad,

    float dx, float inv_dx, float dt,
    float ground_friction, // ground friction ..

    vec3 *__restrict__ out_grid_v,
    vec3 *__restrict__ out_grid_v_grad,
    ivec3 grid_dim,
    int n_bodies,
    int dim)
{
  int tid = get_tid();
  if (tid < dim && grid_m[tid] > 1e-12)
  {
    int grid_x = tid / grid_dim.z / grid_dim.y;
    int grid_y = (tid / grid_dim.z) % grid_dim.y;
    int grid_z = tid % grid_dim.z;

    // TODO: here linear layout assumed...
    // backward start here

    // vec3 v_out = out_grid_v[tid];
    vec3 grad_v = out_grid_v_grad[tid];
    float m = grid_m[tid];
    vec3 mv = grid_v_in[tid];

    vec3 vv = grid_body_v_in[tid * (n_bodies + 1) + n_bodies];
    vec3 v_in = vv;
    const int bound = 3;

    // boundary condition
    if (grid_x > grid_dim.x - bound && vv.x > 0)
    {
      v_in.x = 0;
    }
    if (grid_x < bound && vv.x < 0)
    {
      v_in.x = 0;
    }

    float lin, lit, flag;
    vec3 vit;
    bool hit_ground = grid_y < bound && v_in.y < 0;
    if (hit_ground)
    {
      lin = v_in.y;
      vit = vec3(v_in.x, 0.f, v_in.z);
      lit = norm(vit);
      flag = 1. + ground_friction * lin / lit;
      v_in = vit * fmaxf(flag, 0.f);
    }

    if (grid_z > grid_dim.z - bound && v_in.z > 0)
    {
      grad_v.z = 0;
    }
    if (grid_z < bound && v_in.z < 0)
    {
      grad_v.z = 0;
    }

    if (grid_y > grid_dim.y - bound && v_in.y > 0)
    {
      grad_v.y = 0;
    }

    if (hit_ground)
    {
      grad_v.y = 0;
      float flag = 1. + ground_friction * lin / lit;
      if (flag >= 0.)
      {
        vec3 grad_vit = flag * grad_v;
        float grad_lin = ground_friction / lit * dot(vit, grad_v);
        float grad_lit = -ground_friction * lin / lit / lit * dot(vit, grad_v);
        grad_vit += grad_lit * (vit / lit);
        grad_v = vec3(grad_vit.x, grad_lin, grad_vit.z);
      }
      else
      {
        grad_v = vec3(0.); // v_in becomes zero directly, there is no gradient.
      }
    }

    if (grid_x > grid_dim.x - bound && vv.x > 0)
    {
      grad_v.x = 0;
    }
    if (grid_x < bound && vv.x < 0)
    {
      grad_v.x = 0;
    }

    vec3 lower = cast_float(grid_lower[0]) * dx;
    vec3 gx = lower + vec3(grid_x, grid_y, grid_z) * dx; // no gradient for gx

    for (int body_id = n_bodies - 1; body_id >= 0; --body_id)
    {
      // grad_v is the output grad
      // input
      vec3 v_out = grid_body_v_in[tid * (n_bodies + 1) + body_id];

      vec3 bx = body_pos[body_id];
      quat bq = body_rot[body_id];
      quat type_friction_softness_round = body_type_friction_softness_round[body_id];
      quat shape_args = body_args[body_id];

      float friction = type_friction_softness_round.x;
      float softness = type_friction_softness_round.y;

      vec3 gx_b = spatial_transform_inv(bx, bq, gx); // grid x in body frame
      float dist = shape_sdf(type_friction_softness_round, shape_args, gx_b);
      float influence = min(exp(-dist * softness), 1.);

      if ((softness > 0 && influence > 0.1) || dist <= 0)
      {

        //printf("backward collision %d %f %f %f\n", body_id, v_out.x, v_out.y, v_out.z);
        //printf("collision!! %f %f %f\n", bx.x, bx.y, bx.z);
        vec3 grad_gx_b(0.f), grad_bx(0.f);
        quat grad_bq(0.f, 0.f, 0.f, 0.f);
        float grad_influence = 0.;
        vec3 unnormed_normal = shape_grad(type_friction_softness_round, shape_args, gx_b);

        vec3 real_normal = normalized(unnormed_normal);

        vec3 normal = bq * real_normal; // normal in world frame

        vec3 bv = (spatial_transform(next_body_pos[body_id], next_body_rot[body_id], gx_b) - gx) / dt;
        vec3 rel_v = v_out - bv;

        /*
        printf("v_out %f %f %f\n", v_out.x, v_out.y, v_out.z);
        printf("bv %f %f %f\n", bv.x, bv.y, bv.z);
        printf("bq %f %f %f %f\n", bq.w, bq.x, bq.y, bq.z);
        printf("normal %f %f %f\n", normal.x, normal.y, normal.z);
        printf("dist %f\n", dist);
        printf("in grad_v %f %f %f\n", grad_v.x, grad_v.y, grad_v.z);
        */

        float normal_component = dot(rel_v, normal);
        vec3 grid_v_t_in = rel_v - min(normal_component, 0.f) * normal;

        bool has_friction = normal_component < 0 && dot(grid_v_t_in, grid_v_t_in) > 1e-30;

        vec3 grid_v_t = grid_v_t_in;
        float grid_v_t_norm = length(grid_v_t_in);
        if (has_friction)
        {
          // printf("has friction %f\n", normal_component);
          // apply friction
          grid_v_t = grid_v_t_in * (1.f / grid_v_t_norm) * max(grid_v_t_norm + normal_component * friction, 0.f);
        }
        v_out = bv + rel_v * (1 - influence) + grid_v_t * influence;

        float grad_normal_component = 0.;
        vec3 grad_bv = grad_v, grad_rel_v = grad_v * (1 - influence), grad_grid_v_t = grad_v * influence;
        grad_influence += dot(grid_v_t - rel_v, grad_v); // let's assume it is correct
        if (has_friction)
        {
          float bf = grid_v_t_norm + normal_component * friction;
          if (bf > 0.)
          {
            // (1. + normal_component*f/grid_v_t_norm) * grid_v_t
            // auto tmp = grad_grid_v_t;
            grad_normal_component += dot(grid_v_t_in, grad_grid_v_t) * friction / grid_v_t_norm;
            float grad_grid_v_t_norm = -normal_component * grad_normal_component / grid_v_t_norm;
            grad_grid_v_t = grad_grid_v_t * (1. / grid_v_t_norm) * bf + grad_grid_v_t_norm * grid_v_t_in / grid_v_t_norm;
            // printf("checking friction: %f %f %f %f %f %f %f %f out:%f %f %f %f\n", tmp.x, tmp.y, tmp.z, friction, normal_component, grid_v_t_in.x, grid_v_t_in.y, grid_v_t_in.z, grad_grid_v_t.x, grad_grid_v_t.y, grad_grid_v_t.z, grad_normal_component);
            // printf("%d %d %d\n", grid_x, grid_y, grid_z);
          }
          else
          {
            grad_grid_v_t = vec3(0.f);
          }
        }

        vec3 grad_normal(0.);
        grad_rel_v += grad_grid_v_t;
        // printf("normal component %f\n", grad_normal_component);
        // printf("rel v %f %f %f\n", rel_v.x, rel_v.y,rel_v.z);
        // printf("grid v t %f %f %f\n", grid_v_t.x, grid_v_t.y,grid_v_t.z);
        // printf("normal component %f\n", normal_component);
        if (normal_component < 0.)
        {
          grad_normal_component += -dot(normal, grad_grid_v_t);
          grad_normal += -normal_component * grad_grid_v_t;
          // printf("normal component %f\n", grad_normal_component);
        }
        grad_rel_v += normal * grad_normal_component;
        // printf("grad normal component %f %f\n", grad_normal_component, dot(rel_v, rel_v));
        grad_normal += rel_v * grad_normal_component; // above should be correct obviously

        grad_v = grad_rel_v; // grad of v_out
        grad_bv = grad_bv - grad_rel_v;
        // it seems that the gradient of grad_v_out is finished..
        // let's backward to bx, bq, their next ones, and gx_b
        vec3 grad_next_pos(0.f);
        quat grad_next_rot(0.f, 0.f, 0.f, 0.f);
        /*
        printf("influence %f\n", influence);
        printf("v_out %f %f %f\n", v_out.x, v_out.y, v_out.z);
        printf("rel_v %f %f %f\n", rel_v.x, rel_v.y, rel_v.z);
        printf("grid_v_t %f %f %f\n", grid_v_t.x, grid_v_t.y, grid_v_t.z);

        printf("grad_bv %f %f %f\n", grad_bv.x, grad_bv.y, grad_bv.z);
        printf("grad_v_out %f %f %f\n", grad_v.x, grad_v.y, grad_v.z);
        */

        spatial_transform_backward_inplace(
            next_body_pos[body_id], next_body_rot[body_id], gx_b, grad_bv * (1.f / dt),
            grad_next_pos, grad_next_rot, grad_gx_b); // gx is not differentiable
        atomicAdd(&next_body_pos_grad[body_id], grad_next_pos);
        atomicAdd(&next_body_rot_grad[body_id], grad_next_rot);

        // if(influence < 1.)
        //   printf("%f %f %f %f %f %f %f\n", grad_bv.x, grad_bv.y, grad_bv.z, influence);

        vec3 grad_real_normal(0.f); // before it is rotated ..
        qmul_backward_inplace(bq, real_normal, grad_normal, grad_bq, grad_real_normal);
        /*
        printf("bq grad first: %f %f %f %f\n", grad_bq.w, grad_bq.x, grad_bq.y, grad_bq.z);
        printf("grad normal: %f %f %f\n", grad_normal.x, grad_normal.y, grad_normal.z);
        printf("grad real normal: %f %f %f\n", grad_real_normal.x, grad_real_normal.y, grad_real_normal.z);
        */

        auto found_shape_grad = shape_grad_backward(
            type_friction_softness_round, shape_args, gx_b,
            normalized_backward(unnormed_normal, grad_real_normal));
        // printf("grad_bq_before %f %f %f %f\n", grad_bq.w, grad_bq.x, grad_bq.y, grad_bq.z);
        grad_gx_b += found_shape_grad;

        // gx_b = spatial_transform_inv(bx, bq, gx); // grid x in body frame
        // float influence = min(exp(-dist * softness), 1.);
        float expdist = exp(-dist * softness);
        if (expdist <= 1)
        {
          float grad_dist = -softness * expdist * grad_influence;
          grad_gx_b += unnormed_normal * grad_dist; // unormed_normal = shape_grad(..., gx_b)
        }
        // printf("INFLUENCE grad dist %f %f\n", influence, grad_dist);
        // printf("unnormed %f %f %f\n", unnormed_normal.x * grad_dist, unnormed_normal.y, unnormed_normal.z);
        // printf("bq norm %f\n", bq.w*bq.w+bq.x*bq.x+bq.y*bq.y+bq.z*bq.z);
        vec3 grad_gx_tmp(0.);
        inv_spatial_transform_backward_inplace(bx, bq, gx, grad_gx_b, grad_bx, grad_bq, grad_gx_tmp);

        // only backward when collision happens.
        atomicAdd(&body_pos_grad[body_id], grad_bx);
        atomicAdd(&body_rot_grad[body_id], grad_bq);
      }
    }

    grid_v_in_grad[tid] += 1. / m * grad_v;
    grid_m_grad[tid] += (-1.f / m / m) * dot(mv, grad_v);
  }
}

__global__ void g2p(
    vec3 *__restrict__ particle_x,
    vec3 *__restrict__ grid_v,
    ivec3 *__restrict__ grid_lower,
    float dx, float inv_dx,
    float dt, ivec3 grid_dim,
    vec3 *__restrict__ out_particle_v,
    float ground_height,
    mat3 *__restrict__ out_particle_C,
    vec3 *__restrict__ out_particle_x, int dim)
{
  int p = get_tid();
  if (p < dim)
  {
    vec3 lower = cast_float(grid_lower[0]) * dx;

    vec3 x = particle_x[p] - lower;
    ivec3 base = cast_int(x * inv_dx - 0.5);
    vec3 fx = x * inv_dx - cast_float(base);

    mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f),
                  0.5f * pow2(fx - 0.5f));

    vec3 new_v = vec3(0.f, 0.f, 0.f);
    mat3 new_C = mat3(0.f);
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          float weight = w(0, i) * w(1, j) * w(2, k);
          vec3 dpos = vec3(i, j, k) - fx;
          vec3 v = grid_v[grid_index(base + ivec3(i, j, k), grid_dim)];
          new_v = new_v + v * weight;
          new_C = new_C + outer(v, dpos) * (weight * inv_dx * 4.f);
        }
      }
    }

    // update x, avoid overflow
    vec3 new_x = max(min(x + new_v * dt, (cast_float(grid_dim) - 3.f) * dx), ground_height * dx) + lower;
    // vec3 new_x = x + lower + new_v * dt;
    // if(p==281)
    // printf("cuda: %f %f %f %f %f %f %f %f %f\n", x.x, x.y, x.z, new_v.x, new_v.y, new_v.z, new_x.x, new_x.y, new_x.z);

    out_particle_x[p] = new_x;
    out_particle_v[p] = new_v;
    out_particle_C[p] = new_C;
  }
}

inline CUDA_CALLABLE bool ray_aabb_intersection(vec3 _box_min, vec3 _box_max, vec3 pos, vec3 dir, float &near_int, float &far_int)
{
  float d[3] = {dir.x, dir.y, dir.z};
  float o[3] = {pos.x, pos.y, pos.z};
  float box_min[3] = {_box_min.x, _box_min.y, _box_min.z};
  float box_max[3] = {_box_max.x, _box_max.y, _box_max.z};

  near_int = -1e9;
  far_int = 1e9;

  for (int i = 0; i < 3; ++i)
  {
    if (d[i] == 0)
    {
      if (o[i] < box_min[i] || o[i] > box_max[i])
        return false;
    }
    else
    {
      float i1 = (box_min[i] - o[i]) / d[i];
      float i2 = (box_max[i] - o[i]) / d[i];

      float new_far_int = max(i1, i2);
      float new_near_int = min(i1, i2);
      far_int = min(new_far_int, far_int);
      near_int = max(new_near_int, near_int);
    }
  }
  if (near_int > far_int)
    return false;
  return true;
}

inline CUDA_CALLABLE float next_hit(

    float *__restrict__ sdf_volume,
    vec3 *__restrict__ box_min,
    vec3 *__restrict__ box_max,
    vec3 *__restrict__ color_volume,
    float sdf_threshold,
    ivec3 grid_dim,

    vec3 *__restrict__ body_pos,
    quat *__restrict__ body_rot,

    quat *__restrict__ body_type_friction_softness_round,
    quat *__restrict__ body_args, // describe geometry shape besides round ..
    int n_bodies,
    bool visualize_shape,

    float ground_height,

    vec3 const &pos,
    vec3 const &d,
    float &roughness,
    vec3 &color,
    vec3 &normal,
    int &material,
    float dist_limit = 100.f)
{
  // start hit
  auto o = pos;
  float closest = 1e9;
  vec3 background_color(0.6, 0.7, 0.7);
  if (d.y < 0)
  { // hit ground
    auto ground_dist = (o.y - ground_height) / (-d.y);
    if (ground_dist < dist_limit && ground_dist < closest)
    {
      closest = ground_dist;
      normal = vec3(0., 1., 0.);

      color = vec3(0.3, 0.5, 0.7);
      vec3 p = o + d * closest;
      if (p.x <= 1 && p.x >= 0 && p.z <= 1 && p.z >= 0)
        color = color * (((int(p.x / 0.25) + int(p.z / 0.25)) % 2) * 0.2 + 0.35);
      else
        color = color * 0.4;

      material = 1;
    }
  }
  if (n_bodies > 0)
  {
    int j = 0;
    float dist = 0.;
    int sdf_id = 0;
    float sdf_val = 1e9;
    vec3 pp;
    while (j < 200 && dist < dist_limit && sdf_val > 1e-8)
    {
      // dist limit is 100
      pp = o + d * dist;
      sdf_val = 1e9;
      for (int i = 0; i < n_bodies; ++i)
      {
        auto gx_b = spatial_transform_inv(body_pos[i], body_rot[i], pp);
        float tmp = shape_sdf(body_type_friction_softness_round[i], body_args[i], gx_b);
        if (tmp < sdf_val)
        {
          sdf_val = tmp;
          sdf_id = i;
        }
      }
      dist += sdf_val;
      j += 1;
    }
    if (dist < closest && dist < dist_limit)
    {
      closest = dist;
      auto gx_b = spatial_transform_inv(body_pos[sdf_id], body_rot[sdf_id], pp);
      normal = body_rot[sdf_id] * normalized(shape_grad(body_type_friction_softness_round[sdf_id], body_args[sdf_id], gx_b));
      color = vec3(0.5, 0.5, 0.5);
      roughness = 0.;
      material = 1;
    }
  }

  float tnear = 0., tfar = 0.;
  if (visualize_shape == 1 && ray_aabb_intersection(box_min[0], box_max[0], o, d, tnear, tfar))
  {
    tnear = max(tnear, 0.);
    vec3 pos = o + d * (tnear + 1e-4);
    vec3 step = vec3(0.);

    for (int j = 0; j < 500; ++j)
    {
      float s = sample_sdf(sdf_volume, box_min, box_max, grid_dim, pos, sdf_threshold);
      if (s < 0)
      {
        vec3 back_step = step;
        for (int k = 0; k < 20; ++k)
        {
          back_step = back_step * 0.5;
          if (sample_sdf(sdf_volume, box_min, box_max, grid_dim, pos - back_step, sdf_threshold) < 0)
            pos = pos - back_step;
        }
        float dist = sqrt(dot(o - pos, o - pos));
        if (dist < closest)
        {
          // printf("(%f %f %f) %f\n", pos.x, pos.y, pos.z, s);
          closest = dist;
          normal = normalized(sample_normal(sdf_volume, box_min, box_max, grid_dim, pos));
          color = sample_color(color_volume, box_min, box_max, grid_dim, pos);
          material = 1;
        }
        break;
      }
      else
      {
        step = d * max(s * 0.05, 0.01);
        pos += step;
      }
    }
  }
  return closest;
}

__global__ void render(
    float *__restrict__ sdf_volume,
    vec3 *__restrict__ box_min,
    vec3 *__restrict__ box_max,
    vec3 *__restrict__ color_volume,
    float sdf_threshold,

    vec3 *__restrict__ body_pos,
    quat *__restrict__ body_rot,

    quat *__restrict__ body_type_friction_softness_round,
    quat *__restrict__ body_args, // describe geometry shape besides round ..

    mat3 *__restrict__ camera_rot,
    vec3 *__restrict__ camera_pos,
    mat3 *__restrict__ camera_intrinsic,

    vec3 *__restrict__ color_buffer,
    float *__restrict__ depth_buffer,

    ivec3 grid_dim,
    int n_bodies,
    bool visualize_shape,
    ivec3 image_dim,
    int max_ray_depth,
    int spp,
    int dim, // pixel dim
    curandState *rand_states,

    float ground_height,
    int seed,
    vec3 *__restrict light_direction,
    int use_directional_light = 1,
    float light_direction_noise = 0.1)
{
  float dist_limit = 100.f;
  int p = get_tid();

  if (p < dim)
  {
    curand_init(seed, p, 0, rand_states + p);

    int u = p % image_dim.y;
    int v = p / image_dim.y;
    color_buffer[p] = vec3(0.);
    depth_buffer[p] = 0.f;

    for (int sample_id = 0; sample_id < spp; ++sample_id)
    {
      vec3 dir(u + curand_uniform(rand_states + p), v + curand_uniform(rand_states + p), 1.f);
      // render rgb

      vec3 contrib(0.), throughput(1.);
      int depth = 0, hit_sky = 1, ray_depth = 0;
      float first_depth = 0.;

      dir = camera_intrinsic[0].mul(dir);
      auto d = camera_rot[0].mul(normalized(dir));

      auto pos = camera_pos[0];
      while (depth < max_ray_depth)
      {
        float roughness;
        int material;
        vec3 normal(0.), color(0.);

        float closest = next_hit(sdf_volume, box_min, box_max, color_volume, sdf_threshold, grid_dim, body_pos, body_rot, body_type_friction_softness_round, body_args, n_bodies, visualize_shape, ground_height, pos, d, roughness, color, normal, material);

        if (depth == 0)
          first_depth = closest;
        vec3 hit_pos = pos + closest * d;

        depth += 1;
        ray_depth = depth;
        if (dot(normal, normal) > 1e-10)
        {
          vec3 out_direction(0.);
          if (material == 0)
          {
            out_direction = d - dot(d, normal) * 2 * normal;
          }
          else
          {
            vec3 u(1.0, 0.0, 0.0);
            if (abs(normal.y) < 1 - 1e-3)
              u = normalized(cross(normal, vec3(0.0, 1.0, 0.0)));
            vec3 v = cross(normal, u);
            float phi = 2 * M_PI * curand_uniform(rand_states + p);
            float r = curand_uniform(rand_states + p);
            float ay = sqrt(r);
            float ax = sqrt(1 - r);
            out_direction = ax * (cos(phi) * u + sin(phi) * v) + ay * normal;
          }
          if (roughness > 0.)
            printf("WARNING!!! roughness is not zero");
          vec3 glossy(0.);
          d = normalized(out_direction + glossy);

          pos = hit_pos + 1e-4 * d;
          throughput = throughput * color;

          if (use_directional_light)
          {
            vec3 dir_noise = vec3(
                                 curand_uniform(rand_states + p) - 0.5, curand_uniform(rand_states + p) - 0.5, curand_uniform(rand_states + p) - 0.5) *
                             light_direction_noise;

            vec3 direct = normalized(light_direction[0] + dir_noise);
            float dot_val = dot(direct, normal);
            if (dot_val > 0)
            {
              // printf("%f %f %f %f\n", normal.x, normal.y, normal.z, dot_val);
              float dist = next_hit(sdf_volume, box_min, box_max, color_volume, sdf_threshold, grid_dim, body_pos, body_rot, body_type_friction_softness_round, body_args, n_bodies, visualize_shape, ground_height, pos, direct, roughness, color, normal, material);

              if (dist > dist_limit)
              {
                // vec3(1.) is the light color
                contrib += throughput * dot_val * vec3(0.8f);
              }
            }
          }
        }
        else
        {
          hit_sky = 1;
          break;
        }
      }
      if (hit_sky)
      {
        if (ray_depth != 1)
          ;
        contrib += throughput * vec3(0.8f);
      }
      else
      {
        throughput = vec3(0.);
      }
      vec3 out = contrib;
      if (!use_directional_light)
      {
        float coeff1 = dot(d, vec3(0.8, 0.65, 0.15)) * 0.5 + 0.5;
        coeff1 = max(min(coeff1, 1.), 0.);
        auto light = vec3(0.9, 0.9, 0.9) * coeff1 + vec3(0.7, 0.7, 0.8) * (1. - coeff1);
        out = throughput * light * 10.f;
      }
      color_buffer[p] += out / spp;
      depth_buffer[p] += first_depth/spp; //dot(first_depth * dir, camera_rot[0].mul(vec3(0, 0, 1))) / spp;
    }
    float uu = 1.0 * u / image_dim.y;
    float vv = 1.0 * v / image_dim.x;
    float vignette_strength = 0.9;
    float vignette_center[2] = {0.5, 0.5};
    float vignette_radius = 0.;
    float exposure = 1.5;
    float darken = 1.0 - vignette_strength * max((sqrt(
                                                      (uu - vignette_center[0]) * (uu - vignette_center[0]) +
                                                      (vv - vignette_center[1]) * (vv - vignette_center[1])) -
                                                  vignette_radius),
                                                 0.f);

    vec3 c = color_buffer[p] * darken * exposure;
    c.x = sqrt(c.x);
    c.y = sqrt(c.y);
    c.z = sqrt(c.z);
    color_buffer[p] = c;
  }
}

__global__ void smooth(float *__restrict__ volume, float *__restrict__ volume_out, ivec3 grid_dim, int dim)
{
  int tid = get_tid();
  if (tid < dim)
  {
    int a = grid_dim.x;
    int b = grid_dim.y;
    int c = grid_dim.z;
    ivec3 id(tid / grid_dim.z / grid_dim.y, (tid / grid_dim.z) % grid_dim.y, tid % grid_dim.z);
    if (id.x >= 1 && id.y >= 1 && id.z >= 1 && id.x < a - 1 && id.y < b - 1 && id.z < c - 1)
    {
      float sum = 0.0;
      for (int i = -1; i < 2; ++i)
        for (int j = -1; j < 2; ++j)
          for (int k = -1; k < 2; ++k)
            sum += volume[grid_index(id + ivec3(i, j, k), grid_dim)];
      volume_out[tid] = sum / 27.;
    }
    else
    {
      volume_out[tid] = 1.;
    }
  }
}

__global__ void copy_volume(long long int *__restrict__ volume, float *__restrict__ sdf, vec3 *__restrict__ color, int dim)
{
  int tid = get_tid();
  if (tid < dim)
  {
    long long int c = volume[tid];
    float z = (c & 255) / 255.;
    c = c >> 8;
    float y = (c & 255) / 255.;
    c = c >> 8;
    float x = (c & 255) / 255.;
    c = c >> 8;
    color[tid] = vec3(x, y, z);
    sdf[tid] = (c & 255) / 255.;
  }
}

__global__ void bake_volume(
    long long int *__restrict__ volume,
    vec3 *__restrict__ particle_x,
    int *__restrict__ particle_color,
    vec3 *__restrict__ bbox_min,
    vec3 *__restrict__ bbox_max,
    int bake_size,
    ivec3 grid_dim,
    float inv_dx,
    int dim)
{
  int tid = get_tid();
  if (tid < dim)
  {
    int p = tid;
    int width = bake_size * 2;
    int k = p % width - bake_size;
    p = p / width;
    int j = p % width - bake_size;
    p = p / width;
    int i = p % width - bake_size;
    p = p / width;

    vec3 x = (particle_x[p] - bbox_min[0]) * inv_dx;
    ivec3 coord = cast_int(x);
    /*
    if(p == 1){
      printf("%d %d %d %d %d %d %f %f %f\n", i, j, k, coord.x, coord.y, coord.z,
      particle_x[p].x, particle_x[p].y, particle_x[p].z);
      printf("%f %f %f\n", bbox_min[0].x, bbox_min[0].y, bbox_min[0].z);
    }
    */
    auto idx = coord + ivec3(i, j, k);

    if (idx.x >= 0 && idx.y >= 0 && idx.z >= 0 && idx.x < grid_dim.x && idx.y < grid_dim.y && idx.z < grid_dim.z)
    {
      auto color = particle_color[p];
      auto dist = norm(cast_float(idx) - x, 0.f);
      dist = min(max(0.0f, (255.0f * 0.2f * dist)), 255.0f);
      auto out = (((long long int)(dist)) << 24) + (long long int)(color);
      int gid = grid_index(idx, grid_dim);
      // printf("%d %d %d %f %ld\n", idx.x, idx.y, idx.z, inv_dx, out);
      ::atomicMin(&volume[gid], out);
    }
  }
}

__global__ void g2p_grad(
    vec3 *__restrict__ particle_x,
    vec3 *__restrict__ grid_v,
    ivec3 *__restrict__ grid_lower,
    float dx, float inv_dx,
    float dt, ivec3 grid_dim,
    vec3 *__restrict__ out_particle_v,
    float ground_height,
    mat3 *__restrict__ out_particle_C,
    vec3 *__restrict__ out_particle_x, int dim,

    vec3 *__restrict__ particle_x_grad,
    vec3 *__restrict__ grid_v_grad,

    vec3 *__restrict__ out_particle_v_grad,
    mat3 *__restrict__ out_particle_C_grad,
    vec3 *__restrict__ out_particle_x_grad)
{
  int p = get_tid();
  if (p < dim)
  {
    // update x, avoid overflow
    // vec3 new_x = max(min(x + new_v * dt, (cast_float(grid_dim) - 3.f) * dx), ground_height * dx) + lower;
    // vec3 new_x = x + lower + new_v * dt;
    // out_particle_x[p] = new_x;
    // out_particle_v[p] = new_v;
    // out_particle_C[p] = new_C;
    vec3 lower = cast_float(grid_lower[0]) * dx;
    const vec3 x = particle_x[p] - lower;

    vec3 grad_x = out_particle_x_grad[p];
    vec3 grad_new_v = out_particle_v_grad[p];
    mat3 grad_new_C = out_particle_C_grad[p];

    vec3 new_x = x + out_particle_v[p] * dt;
    vec3 upper = (cast_float(grid_dim) - 3.f) * dx;
    float lower_limit = ground_height * dx;
    if (new_x.x > upper.x || new_x.x < lower_limit)
      grad_x.x = 0;
    if (new_x.y > upper.y || new_x.y < lower_limit)
      grad_x.y = 0;
    if (new_x.z > upper.z || new_x.z < lower_limit)
      grad_x.z = 0;
    grad_new_v += grad_x * dt;

    ivec3 base = cast_int(x * inv_dx - 0.5);
    vec3 fx = x * inv_dx - cast_float(base);

    mat3 w = mat3(0.5f * pow2(1.5f - fx), 0.75f - pow2(fx - 1.f), 0.5f * pow2(fx - 0.5f));
    mat3 w1 = mat3(-inv_dx * (1.5f - fx), inv_dx * ((-2.f) * fx + 2.0f), -inv_dx * (fx * (-1.f) + 0.5f));

    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          float weight = w(0, i) * w(1, j) * w(2, k);

          vec3 dpos = vec3(i, j, k) - fx;
          int tid = grid_index(base + ivec3(i, j, k), grid_dim);

          // auto gg = base + ivec3(i, j, k);
          // if(gg.x>63 || gg.y > 63 || gg.z > 63||gg.x <0 || gg.y<0 || gg.z<0)
          //   printf("%d %d %d\n", gg.x, gg.y, gg.z);

          vec3 v = grid_v[tid];
          // new_v = new_v + v * weight;
          // new_C = new_C + outer(v, dpos) * (weight * inv_dx * 4.f);
          //  vdpos^T  then v's grad is g dpos
          float xx = weight * inv_dx * 4.;
          vec3 grad_grid_v = weight * grad_new_v + grad_new_C.mul(dpos) * xx; // v dpos^T
          atomicAdd(&grid_v_grad[tid], grad_grid_v);

          // grad_dpos which is -fx = -x * inv_dx;  v dpos^T, gradient to dpos is simular to that of V^T
          grad_x += -inv_dx * grad_new_C.transpose().mul(v) * xx;

          float grad_weight = dot(grad_new_v, v) + (inv_dx * 4.f) * (outer(v, dpos) * grad_new_C).sum();
          grad_x += dw(w, w1, i, j, k) * grad_weight; // I guess so..

          // auto tmp = dw(w, w1, i, j, k);printf("%f %f %f %f %d %d %d %f %f %f\n", weight, fx.x, fx.y, fx.z, i, j, k, tmp.x, tmp.y, tmp.z); //checked with PYTORCH!
        }
      }
    }
    // atomicAdd(&particle_x_grad, grad_x);
    particle_x_grad[p] += grad_x;
  }
}

extern "C"
{

  void compute_grid_lower(vec3 *__restrict__ particle_x, float dx, float inv_dx,
                          ivec3 *__restrict__ grid_lower, int dim,
                          cudaStream_t stream)
  {
    launch_kernel(compute_grid_lower, dim, stream,
                  (particle_x, dx, inv_dx, grid_lower, dim));
  }

  void compute_svd(
      mat3 *__restrict__ F,
      mat3 *__restrict__ C,
      mat3 *__restrict__ newF,
      mat3 *__restrict__ U,
      mat3 *__restrict__ V,
      vec3 *__restrict__ sig,
      float dt,
      int dim, cudaStream_t stream)
  {
    launch_kernel(compute_svd, dim, stream, (F, C, newF, U, V, sig, dt, dim));
  }

  void compute_svd_grad(
      mat3 *__restrict__ F,
      mat3 *__restrict__ C,
      mat3 *__restrict__ U,
      mat3 *__restrict__ V,
      vec3 *__restrict__ sig,

      mat3 *__restrict__ newF_grad,
      mat3 *__restrict__ U_grad,
      mat3 *__restrict__ V_grad,
      vec3 *__restrict__ sig_grad,
      mat3 *__restrict__ F_grad,
      mat3 *__restrict__ C_grad,
      float dt,
      int dim, cudaStream_t stream)
  {
    launch_kernel(compute_svd_grad, dim, stream, (F, C, U, V, sig, newF_grad, U_grad, V_grad, sig_grad, F_grad, C_grad, dt, dim));
  }

  void p2g(
      vec3 *__restrict__ particle_x,
      vec3 *__restrict__ particle_v,
      float *__restrict__ particle_m,
      float *__restrict__ particle_vol,

      mat3 *__restrict__ particle_F,
      mat3 *__restrict__ particle_U,
      vec3 *__restrict__ particle_sig,
      mat3 *__restrict__ particle_V,

      mat3 *__restrict__ particle_C,
      vec3 *__restrict__ particle_mu_lam_yield,

      ivec3 *__restrict__ grid_lower,
      ivec3 const &grid_dim,
      float dx,
      float inv_dx,
      float dt,
      mat3 *__restrict__ out_particle_F,
      vec3 *__restrict__ out_grid_mv,
      float *__restrict__ out_grid_m,
      int dim,
      cudaStream_t stream)
  {
    // printf("strat p2g. %d %d %d %d\n", grid_dim->x, grid_dim->y, grid_dim->z, dim);
    launch_kernel(p2g, dim, stream,
                  (particle_x, particle_v, particle_m, particle_vol,
                   particle_F, particle_U, particle_sig, particle_V, particle_C, particle_mu_lam_yield, grid_lower,
                   grid_dim, dx, inv_dx, dt, out_particle_F, out_grid_mv,
                   out_grid_m, dim));
  }

  void p2g_grad(
      vec3 *__restrict__ particle_x,
      vec3 *__restrict__ particle_v,
      float *__restrict__ particle_m,
      float *__restrict__ particle_vol,

      mat3 *__restrict__ particle_F,
      mat3 *__restrict__ particle_U,
      vec3 *__restrict__ particle_sig,
      mat3 *__restrict__ particle_V,

      mat3 *__restrict__ particle_C,
      vec3 *__restrict__ particle_mu_lam_yield,

      ivec3 *__restrict__ grid_lower,
      ivec3 const &grid_dim,
      float dx,
      float inv_dx,
      float dt,
      mat3 *__restrict__ out_particle_F,
      vec3 *__restrict__ out_grid_mv,
      float *__restrict__ out_grid_m,

      vec3 *__restrict__ particle_x_grad,
      vec3 *__restrict__ particle_v_grad,
      mat3 *__restrict__ particle_F_grad,
      mat3 *__restrict__ particle_C_grad,

      mat3 *__restrict__ particle_U_grad,
      vec3 *__restrict__ particle_sig_grad,
      mat3 *__restrict__ particle_V_grad,

      mat3 *__restrict__ out_particle_F_grad,
      vec3 *__restrict__ out_grid_v_grad,
      float *__restrict__ out_grid_m_grad,

      int dim,
      cudaStream_t stream)
  {
    launch_kernel(p2g_grad, dim, stream,
                  (particle_x, particle_v, particle_m, particle_vol,
                   particle_F, particle_U, particle_sig, particle_V, particle_C, particle_mu_lam_yield, grid_lower,
                   grid_dim, dx, inv_dx, dt, out_particle_F, out_grid_mv,
                   out_grid_m,

                   particle_x_grad, particle_v_grad, particle_F_grad, particle_C_grad,
                   particle_U_grad, particle_sig_grad, particle_V_grad,
                   out_particle_F_grad, out_grid_v_grad, out_grid_m_grad,

                   dim));
  }

  /*
  void grid_op(float *__restrict__ grid_m, vec3 *__restrict__ grid_mv,
               vec3 *__restrict__ out_grid_v, ivec3 const &grid_dim,
               cudaStream_t stream) {
    int dim = grid_dim.x * grid_dim.y * grid_dim.z;
    launch_kernel(grid_op, dim, stream,
                  (grid_m, grid_mv, out_grid_v, grid_dim, dim));
  }
  */

  void grid_op_v2(
      float *__restrict__ grid_m,
      vec3 *__restrict__ grid_v_in,
      vec3 *__restrict__ grid_body_v_in,
      ivec3 *grid_lower,
      vec3 *gravity,
      // rigid body
      vec3 *__restrict__ body_pos,
      quat *__restrict__ body_rot,

      vec3 *__restrict__ next_body_pos,
      quat *__restrict__ next_body_rot,

      quat *__restrict__ body_type_friction_softness_round,
      quat *__restrict__ body_args, // describe geometry shape besides round ..

      float dx, float inv_dx, float dt,
      float ground_friction, // ground friction ..

      vec3 *__restrict__ out_grid_v,
      ivec3 const &grid_dim, int n_bodies, cudaStream_t stream)
  {
    int dim = grid_dim.x * grid_dim.y * grid_dim.z;
    launch_kernel(grid_op_v2, dim, stream,
                  (grid_m, grid_v_in, grid_body_v_in, grid_lower, gravity, body_pos, body_rot, next_body_pos, next_body_rot, body_type_friction_softness_round, body_args, dx, inv_dx, dt, ground_friction, out_grid_v, grid_dim, n_bodies, dim));
  }

  void grid_op_v2_grad(
      float *__restrict__ grid_m,
      vec3 *__restrict__ grid_v_in,
      vec3 *__restrict__ grid_body_v_in,
      ivec3 *grid_lower,
      vec3 *gravity,
      // rigid body
      vec3 *__restrict__ body_pos,
      quat *__restrict__ body_rot,

      vec3 *__restrict__ next_body_pos,
      quat *__restrict__ next_body_rot,

      quat *__restrict__ body_type_friction_softness_round,
      quat *__restrict__ body_args, // describe geometry shape besides round ..

      float *__restrict__ grid_m_grad,
      vec3 *__restrict__ grid_v_in_grad,
      vec3 *__restrict__ body_pos_grad,
      quat *__restrict__ body_rot_grad,

      vec3 *__restrict__ next_body_pos_grad,
      quat *__restrict__ next_body_rot_grad,

      float dx, float inv_dx, float dt,
      float ground_friction, // ground friction ..

      vec3 *__restrict__ out_grid_v,
      vec3 *__restrict__ out_grid_v_grad,
      ivec3 const &grid_dim, int n_bodies, cudaStream_t stream)
  {
    int dim = grid_dim.x * grid_dim.y * grid_dim.z;
    launch_kernel(grid_op_v2_grad, dim, stream,
                  (grid_m, grid_v_in, grid_body_v_in, grid_lower, gravity, body_pos, body_rot, next_body_pos, next_body_rot, body_type_friction_softness_round, body_args,

                   grid_m_grad, grid_v_in_grad, body_pos_grad, body_rot_grad, next_body_pos_grad, next_body_rot_grad,

                   dx, inv_dx, dt, ground_friction, out_grid_v, out_grid_v_grad, grid_dim, n_bodies, dim));
  }

  void g2p(
      vec3 *__restrict__ particle_x,
      vec3 *__restrict__ grid_v,
      ivec3 *__restrict__ grid_lower,
      float dx, float inv_dx, float dt,
      ivec3 const &grid_dim,
      vec3 *__restrict__ out_particle_v,
      float ground_height,
      mat3 *__restrict__ out_particle_C, vec3 *__restrict__ out_particle_x,
      int dim, cudaStream_t stream)
  {
    launch_kernel(g2p, dim, stream,
                  (particle_x, grid_v, grid_lower, dx, inv_dx, dt, grid_dim,
                   out_particle_v, ground_height, out_particle_C, out_particle_x, dim));
  }

  void g2p_grad(
      vec3 *__restrict__ particle_x,
      vec3 *__restrict__ grid_v,
      ivec3 *__restrict__ grid_lower,
      float dx, float inv_dx, float dt,
      ivec3 const &grid_dim,
      vec3 *__restrict__ out_particle_v,
      float ground_height,
      mat3 *__restrict__ out_particle_C, vec3 *__restrict__ out_particle_x,
      int dim,

      vec3 *__restrict__ particle_x_grad,
      vec3 *__restrict__ grid_v_grad,

      vec3 *__restrict__ out_particle_v_grad,
      mat3 *__restrict__ out_particle_C_grad,
      vec3 *__restrict__ out_particle_x_grad,

      cudaStream_t stream)
  {
    launch_kernel(g2p_grad, dim, stream,
                  (particle_x, grid_v, grid_lower, dx, inv_dx, dt, grid_dim,
                   out_particle_v, ground_height, out_particle_C, out_particle_x, dim,
                   particle_x_grad, grid_v_grad, out_particle_v_grad, out_particle_C_grad, out_particle_x_grad));
  }

  void render(
      float *__restrict__ sdf_volume,
      vec3 *__restrict__ box_min,
      vec3 *__restrict__ box_max,
      vec3 *__restrict__ color_volume,

      vec3 *__restrict__ body_pos,
      quat *__restrict__ body_rot,

      quat *__restrict__ body_type_friction_softness_round,
      quat *__restrict__ body_args, // describe geometry shape besides round ..

      mat3 *__restrict__ camera_rot,
      vec3 *__restrict__ camera_pos,
      mat3 *__restrict__ camera_intrinsic,

      vec3 *__restrict__ color_buffer,
      float *__restrict__ depth_buffer,
      float sdf_threshold,

      const ivec3 &grid_dim,
      int n_bodies,
      bool visualize_shape,
      const ivec3 &image_dim,
      int max_ray_depth,
      int spp,
      float ground_height,
      int seed,
      vec3 *__restrict light_direction,
      cudaStream_t stream
      // int use_directional_light=1,
      // float light_direction_noise=0.1,
  )
  {
    int dim = image_dim.x * image_dim.y;
    curandState *rand_state;
    cudaMalloc(&rand_state, sizeof(curandState) * (dim + 1));

    launch_kernel(render, dim, stream,
                  (sdf_volume, box_min, box_max, color_volume, sdf_threshold, body_pos, body_rot, body_type_friction_softness_round, body_args, camera_rot, camera_pos, camera_intrinsic, color_buffer, depth_buffer,
                   grid_dim, n_bodies, visualize_shape, image_dim, max_ray_depth, spp, dim, rand_state, ground_height, seed, light_direction));

    cudaFree(rand_state);
  }

  void particle_sdf(
      long long int *__restrict__ volume,
      vec3 *__restrict__ particle_x,
      int *__restrict__ particle_color,
      vec3 *__restrict__ bbox_min,
      vec3 *__restrict__ bbox_max,
      int bake_size, // let the default to be 7
      const ivec3 &grid_dim,
      float inv_dx,
      float *__restrict__ sdf,
      vec3 *__restrict__ color,
      float *__restrict__ sdf_tmp,
      int n_particles,
      cudaStream_t stream)
  {
    // launch_kernel(compute_bbox, n_particles, stream, (particle_x, bbox_min, bbox_max, inv_dx, n_particles));
    // printf("%d %d %d\n", _bbox_min[0].x, _bbox_min[0].y, _bbox_min[0].z);
    int dim_bake = n_particles * bake_size * bake_size * bake_size * 8;
    launch_kernel(bake_volume, dim_bake, stream, (volume, particle_x, particle_color, bbox_min, bbox_max, bake_size, grid_dim, inv_dx, dim_bake));

    int dim = grid_dim.x * grid_dim.y * grid_dim.z;
    launch_kernel(copy_volume, dim, stream, (volume, sdf, color, dim));

    launch_kernel(smooth, dim, stream, (sdf, sdf_tmp, grid_dim, dim));
    launch_kernel(smooth, dim, stream, (sdf_tmp, sdf, grid_dim, dim));
  }

  void compute_dist(
      vec3 *__restrict__ particle_x,
      vec3 *__restrict__ body_pos,
      quat *__restrict__ body_rot,
      quat *__restrict__ body_type_friction_softness_round,
      quat *__restrict__ body_args, // describe geometry shape besides round ..
      float *__restrict__ dist,
      int n_bodies,
      vec3 *__restrict__ particle_x_grad,
      vec3 *__restrict__ body_pos_grad,
      quat *__restrict__ body_rot_grad,
      float *__restrict__ dist_grad,
      int compute_grad,
      int dim,
      cudaStream_t stream)
  {
    /*
    printf("%p %p %p\n", particle_x, body_pos, body_rot);
    printf("%p %p %p\n", body_type_friction_softness_round, body_args, dist);
    printf("%p %p %p\n", particle_x_grad,body_pos_grad, body_rot_grad);
    printf("%p %p\n", dist_grad, stream);
    */
    launch_kernel(compute_dist, dim, stream,
                  (particle_x, body_pos, body_rot, body_type_friction_softness_round, body_args, dist, n_bodies,
                   particle_x_grad, body_pos_grad, body_rot_grad, dist_grad, compute_grad, dim));
  }

  void particle2mass(
      vec3 *__restrict__ particle_x,
      float *__restrict__ particle_m,

      ivec3 *__restrict__ grid_lower,
      const ivec3 &grid_dim,
      float dx,
      float inv_dx,
      float *__restrict__ out_grid_m,

      float *__restrict__ out_grid_m_grad,
      vec3 *__restrict__ particle_x_grad,

      int *__restrict__ particle_ids,
      int id,

      int compute_grad,
      int dim,
      cudaStream_t stream)
  {
    launch_kernel(particle2mass, dim, stream,
                  (particle_x, particle_m, grid_lower, grid_dim, dx, inv_dx, out_grid_m, out_grid_m_grad, particle_x_grad, particle_ids, id, compute_grad, dim));
  }
}

extern "C"
{

  void *cuda_alloc(size_t size)
  {
    void *dst;
    CHECK_CUDA(cudaMalloc(&dst, size));
    return dst;
  }

  void print_memory_info(){
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    //for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
    //    cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        printf("GPU %d/%d memory: free=%.3f, total=%.3f\n", id, num_gpus, free/1024./1024./1024., total/1024./1024./1024.);
    //}
    //cudaSetDevice(device);
  }

  void cuda_free(void *ptr) { CHECK_CUDA(cudaFree(ptr)); }

  cudaStream_t cuda_stream_create()
  {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    return stream;
  }

  void cuda_stream_destroy(cudaStream_t stream)
  {
    CHECK_CUDA(cudaStreamDestroy(stream));
  }

  void cuda_upload(void *device_ptr, void *host_ptr, size_t size)
  {
    CHECK_CUDA(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));
  }

  void cuda_download(void *host_ptr, void *device_ptr, size_t size)
  {
    CHECK_CUDA(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
  }

  void cuda_copy(void *dst, void *src, size_t size)
  {
    CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice))
  }

  void cuda_copy2d(void *dst, size_t dpitch, void *src, size_t spitch,
                   size_t width, size_t height)
  {
    CHECK_CUDA(cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                            cudaMemcpyDeviceToDevice));
  }

  void cuda_zero(void *ptr, size_t size) { CHECK_CUDA(cudaMemset(ptr, 0, size)); }
  void cuda_zero_async(void *ptr, size_t size, cudaStream_t stream) { CHECK_CUDA(cudaMemsetAsync(ptr, 0, size, stream)); }

  void cuda_upload_async(void *device_ptr, void *host_ptr, size_t size,
                         cudaStream_t stream)
  {
    CHECK_CUDA(cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice,
                               stream));
  }

  void cuda_download_async(void *host_ptr, void *device_ptr, size_t size,
                           cudaStream_t stream)
  {
    CHECK_CUDA(cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost,
                               stream));
  }

  void cuda_copy_async(void *dst, void *src, size_t size, cudaStream_t stream)
  {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream))
  }

  void cuda_stream_sync(cudaStream_t stream)
  {
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  struct texture_resources
  {
    cudaArray_t array;
    cudaTextureObject_t texture;
  };

  texture_resources create_volume(float *data, int x, int y, int z)
  {
    // data have shape [z, y, x, 4] and size x*y*z*4*4
    cudaArray_t cuArray = 0;
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    CHECK_CUDA(
        cudaMalloc3DArray(&cuArray, &channelDesc, make_cudaExtent(x, y, z)));

    // copy data
    cudaMemcpy3DParms params;
    memset(&params, 0, sizeof(params));
    params.srcPtr = make_cudaPitchedPtr(data, x * 16, x, y);
    params.dstArray = cuArray;
    params.extent = make_cudaExtent(x, y, z);
    params.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&params);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    texture_resources r;
    r.array = cuArray;
    r.texture = texObj;
    return r;
  }

  void destroy_volume(texture_resources tex)
  {
    CHECK_CUDA(cudaDestroyTextureObject(tex.texture));
    CHECK_CUDA(cudaFreeArray(tex.array));
  }
}
