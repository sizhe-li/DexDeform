#pragma once
#include "vec3.h"
#include "quat.h"
#define NORM_EPS2 1e-30

namespace maniskill {
    inline CUDA_CALLABLE float length(vec3 const &vec){
        return sqrt(dot(vec, vec) + NORM_EPS2);
    }


    // the gradient of length is actuall the normalized function..
    inline CUDA_CALLABLE vec3 normalized(vec3 const &vec){
        return vec/length(vec);
    }

    inline CUDA_CALLABLE vec3 normalized_backward(vec3 const &vec, vec3 const& grad_out){
        //return vec/length(vec);
        //checked with pytorch!
        float doted = dot(vec, vec) + NORM_EPS2;
        auto out = (grad_out - vec * dot(vec/doted, grad_out)) * (1./sqrt(doted));
        //mat3 Jacob = (mat3(doted) - outer(vec, vec))* (1./doted/sqrt(doted));
        //auto out2 = Jacob.mul(grad_out);
        return out;
    }

    inline CUDA_CALLABLE void abs_backward_inplace(vec3 const &gx, vec3 &grad){
        if(gx.x<0) grad.x*=-1;
        if(gx.y<0) grad.y*=-1;
        if(gx.z<0) grad.z*=-1;
    }

    inline CUDA_CALLABLE int get_type(quat const &type_softness_friction_round){
        return (int)(floor(type_softness_friction_round.w + 0.1f));
    }

    inline CUDA_CALLABLE float shape_sdf(
        quat const &type_softness_friction_round,
        quat const &shape_args,
        vec3 const &gx)
    {
        int type = get_type(type_softness_friction_round);
        float sdf = 0.;

        if(type == 0){
            // box
            vec3 q = abs(gx) - vec3(shape_args.w, shape_args.x, shape_args.y);
            sdf = length(max(q, 0.f)) + fminf(fmaxf(fmaxf(q.x, q.y), q.z), 0.f);
        }
        else if(type == 1){
            vec3 p2 = gx;
            float r=shape_args.w, h=shape_args.x;
            p2.y += h / 2;
            p2.y -= fminf(fmaxf(p2.y, 0.f), h) ;
            sdf = length(p2) - r;
        }
        else{
            printf("ERORR: Not specified types");
        }
        return sdf - type_softness_friction_round.z;
    }



    inline CUDA_CALLABLE vec3 shape_grad(
        // unnormalized sdf
        quat const &type_softness_friction_round,
        quat const &shape_args,
        vec3 const &gx) {

        int type = get_type(type_softness_friction_round);
        vec3 grad(0.);

        if(type == 0){
            vec3 q = abs(gx) - vec3(shape_args.w, shape_args.x, shape_args.y);
            float inside = fmaxf(fmaxf(q.x, q.y), q.z);
            if(inside<=0){
                if(q.x == inside)  grad.x += 1;
                if(q.y == inside)  grad.y += 1;
                if(q.z == inside)  grad.z += 1;
            } else {
                vec3 q2 = max(q, 0.f);
                // we do not have to consider the max operator here, because grad must have the same sign with q2. 
                grad = normalized(q2);
            }
            abs_backward_inplace(gx, grad);

        } 
        else if(type == 1){
            vec3 p2 = gx;
            float r=shape_args.w;
            float h=shape_args.x;
            p2.y += h / 2;
            p2.y -= fminf(fmaxf(p2.y, 0.f), h);
            // gradient of length. in case that p2.y is between [0, h] now, it should be zero, thus the gradient is also zero.
            return normalized(p2);
        }
        else{
            printf("ERORR: Not specified types");
        }
        return grad;
    }

    inline CUDA_CALLABLE vec3 shape_grad_backward(
        // unnormalized sdf
        quat const &type_softness_friction_round,
        quat const &shape_args,
        vec3 const &gx,
        vec3 const &grad_out) {

        int type = get_type(type_softness_friction_round);
        vec3 grad_in(0.);

        if(type == 0){
            vec3 q = abs(gx) - vec3(shape_args.w, shape_args.x, shape_args.y);
            //printf("shape args %f %f %f\n", shape_args.w, shape_args.x, shape_args.y);
            float inside = fmaxf(fmaxf(q.x, q.y), q.z);

            grad_in = grad_out;
            abs_backward_inplace(gx, grad_in); // note that this is not really the same ..
            if(inside <=0){
                // there is no gradient here?
                grad_in = vec3(0.f);
            } else {
                vec3 q2 = max(q, 0.f);
                grad_in = normalized_backward(q2, grad_in); //get the gradient of q2
                //not sure if we need this, but let's do it.
                if(q.x < 0) grad_in.x = 0;
                if(q.y < 0) grad_in.y = 0;
                if(q.z < 0) grad_in.z = 0;
                //vec3 y = normalized_backward(vec3(1., 2., 3.), vec3(4., 5., 6.)); printf("%f %f %f\n", y.x, y.y, y.z);
                abs_backward_inplace(gx, grad_in);
            }

        } 
        else if (type == 1){
            vec3 p2 = gx;
            float r = shape_args.w;
            float h = shape_args.x;
            p2.y += h / 2;
            float zero_y_grad = (p2.y>=0.f && p2.y <=h);
            p2.y -= fminf(fmaxf(p2.y, 0.f), h);

            grad_in = normalized_backward(p2, grad_in); //get the gradients of p2
            grad_in.y *= zero_y_grad;
            return grad_in;
        }
        else{
            printf("ERORR: Not specified types");
        }
        return grad_in;
    }
} // namespace maniskill
