/*
 * Projected Spherical Cap Maps
 * Original source code:
 * Copyright (C) 2018 Carlos Ureña and Iliyan Georgiev
 * https://github.com/carlos-urena/psc-sampler
 *
 * Ported to straight C for OpenCL compatibility by Stefan Werner
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

CCL_NAMESPACE_BEGIN

typedef struct PSCMap {
  bool                    // values defining the spherical cap type, and which map is being used
      initialized,        // true if the sampler has been initialized
      fully_visible,      // true iif r <= cz     (sphere fully visible)
      partially_visible,  // true iif -r < cz < r (sphere partially visible)
      center_below_hor,  // true iif -r <= cz < 0 (partially visible and sphere center below horizon)
      invisible,         // true iif cz <= -r    (sphere completely invisible)
      using_radial;      // true when using radial map, false when using horizontal map

  float   // areas (form factors)
      E,  // half of the ellipse area
      L,  // half lune area, (it is 0 in the 'ellipse only' case)
      F;  // total form factor: it is:
          //    0 -> when invisible
          //    2E -> when fully_visible (ellipse only)
          //    2L -> when partially_visible and center_below_hor (lune only)
          //    2(E+L) -> otherwise (partially_visible and not center_below_hor)

  float  // sinus and cosine of beta
      cos_beta,
      cos_beta_sq, sin_beta, sin_beta_abs;

  float    // parameters defining the ellipse
      xe,  // X coord. of ellipse center in the local reference frame
      ax,  // ellipse semi-minor axis (X direction)
      ay;  // ellipse semi-major axis (Y direction)

  float         // some precomputed constants
      axay2,    // == (ax*ay)/2
      ay_sq,    // == ay^2
      xe_sq,    // == xe^2
      r1maysq;  // root of (1-ay^2)

  float          // parameters computed only when partially_visible ( there are tangency points)
      xl,        // X coord. of tangency point p (>0)
      yl,        // Y coord. of tangency point p (>0)
      phi_l,     // == arctan(yl/(xe-xl)), only if 'using_radial' (and 'partially_visible')
      AE_phi_l;  // == A_E(phi_l), , only if 'using_radial' (and 'partially_visible')
} PSCMap;

// initial value for the tolerance of the inverse newton function
#define ini_iN_tolerance 1e-4f
// initial value for the max number of iterations in the inverse newton func.
#define ini_iN_max_iters 20

ccl_device_inline float check_y(float y, float max_y)
{
  return max(0.0f, min(y, max_y));
}

ccl_device_inline float check_theta(float theta, float max_theta)
{
  return max(0.0f, min(theta, max_theta));  // trunc
}

ccl_device_inline float eval_ArE(const PSCMap *maps, float theta)
{
  theta = check_theta(theta, M_PI_F);
  const float tt = fabsf(tanf(theta));

  const float pi2 = 0.5f * M_PI_F;
  float result;
  if (theta <= pi2) {
    result = maps->axay2 * atanf(maps->sin_beta_abs * tt);
  }
  else {
    result = maps->axay2 * (M_PI_F - atanf(maps->sin_beta_abs * tt));
  }
  return result;
}

ccl_device_inline float eval_I(float u, float w)
{
  return 0.5f * (w * u * safe_sqrtf(1.0f - u * u) + safe_asinf(u));  // expresion 17
}

ccl_device_inline float eval_ArC(const PSCMap *maps, float theta)
{
  theta = check_theta(theta, maps->phi_l);
  const float z = sinf(theta),  // z == asin(theta)
      xez = maps->xe * z, z_sq = z * z, xe_sq = maps->xe * maps->xe;

  // this is eq. 25 in the paper:
  // return eval_I( z, xe_sq ) - eval_I( xe*z, T(1.0) );

  // this is equivalent to, but faster than, eq. 25
  // (an 'asin' call is saved)
  return 0.5f * (theta - safe_asinf(xez) + xe_sq * z * safe_sqrtf(1.0f - z_sq) -
                 xez * safe_sqrtf(1.0f - xe_sq * z_sq));
}

ccl_device_inline float eval_ApC(const PSCMap *maps, float y)
{
  y = check_y(y, maps->yl);
  return eval_I(y, 1.0f) - maps->xe * y;  // expresion 16
}

ccl_device_inline float eval_ApE(const PSCMap *maps, float y)
{
  y = check_y(y, maps->ay);
  const float v = max(0.0f, min(1.0f, y / maps->ay));
  return maps->ax * maps->ay * eval_I(v, 1.0f);  // expression 16
}

ccl_device_inline void compute_ELF_xlyl_phi_l(PSCMap *maps)
{
  // initialize areas
  maps->E = 0.0f;
  maps->L = 0.0f;
  maps->F = 0.0f;

  if (maps->partially_visible) {
    maps->xl = maps->r1maysq / maps->cos_beta;
    maps->yl = safe_sqrtf(1.0f - maps->xl * maps->xl);

    // compute L
    if (maps->using_radial) {
      maps->phi_l = atan2(maps->yl, maps->xl - maps->xe);
      maps->AE_phi_l = eval_ArE(maps, maps->phi_l);
      maps->L = eval_ArC(maps, maps->phi_l) - maps->AE_phi_l;
    }
    else {
      maps->L = eval_ApC(maps, maps->yl) - eval_ApE(maps, maps->yl);
    }
    if (maps->L < 0.0f) {
      maps->L = 0.0f;
    }
    // compute E
    if (!maps->center_below_hor) {     // ellipse only or ellipse+lune cases (both maps)
      maps->E = M_PI_F * maps->axay2;  // half ellipse area
    }
  }
  else {
    maps->E = M_PI_F * maps->axay2;  // half ellipse area
  }
  // compute F
  maps->F = 2.0f * (maps->E + maps->L);

  if (maps->F < 1e-6f) {
    maps->invisible = true;
  }
}

ccl_device_inline void PSCMap_initialize(PSCMap *maps, float alpha, float beta, bool p_use_radial)
{
  maps->ay = sinf(alpha);
  maps->ay_sq = maps->ay * maps->ay;
  maps->r1maysq = safe_sqrtf(1.0f - maps->ay_sq);
  maps->sin_beta = sinf(beta);
  maps->sin_beta_abs = fabsf(maps->sin_beta);
  maps->cos_beta_sq = 1.0f - maps->sin_beta * maps->sin_beta;
  maps->cos_beta = safe_sqrtf(maps->cos_beta_sq);  // spherical cap center, X coord.
  maps->xe = maps->cos_beta * maps->r1maysq;       // ellipse center
  maps->ax = maps->ay * maps->sin_beta_abs;        // semi-minor axis length (UNSIGNED)
  maps->axay2 = maps->ax * maps->ay * 0.5f;
  maps->xe_sq = maps->xe * maps->xe;

  maps->using_radial = p_use_radial;
  maps->fully_visible = false;
  maps->partially_visible = false;
  maps->invisible = false;
  maps->center_below_hor = false;

  if (maps->ay <= maps->sin_beta) {
    maps->fully_visible = true;
  }
  else if (-maps->ay < maps->sin_beta) {
    maps->partially_visible = true;
  }
  else {
    maps->invisible = true;
  }

  maps->center_below_hor = false;
  if (maps->sin_beta < 0.0f) {
    maps->center_below_hor = true;
  }

  compute_ELF_xlyl_phi_l(maps);
}

ccl_device_inline float sphere_light_projected_area(float alpha, float beta, bool p_use_radial)
{
  PSCMap maps;
  PSCMap_initialize(&maps, alpha, beta, p_use_radial);

  /* return the area */
  return maps.F;
}

ccl_device_inline float eval_ArE_inverse(const PSCMap *maps, float Ar_value)
{
  const float Ar_max_value = maps->E;

  Ar_value = max(0.0f, min(Ar_value, Ar_max_value));

  const float ang = Ar_value / maps->axay2;
  const float pi2 = 0.5f * M_PI_F;

  if (ang <= pi2) {
    return atanf(tanf(ang) / maps->sin_beta_abs);
  }
  else {
    return M_PI_F + atanf(tanf(ang) / maps->sin_beta_abs);
  }
}

ccl_device_inline float eval_Ar(const PSCMap *maps, float theta)
{

  // ellipse only
  if (maps->fully_visible) {
    return eval_ArE(maps, theta);
  }

  // lune only
  if (maps->center_below_hor) {
    return (theta <= maps->phi_l) ? (eval_ArC(maps, theta) - eval_ArE(maps, theta)) : maps->L;
  }

  // ellipse+lune
  return (theta <= maps->phi_l) ? eval_ArC(maps, theta) : eval_ArE(maps, theta) + maps->L;
}

ccl_device_inline float eval_rCirc(const PSCMap *maps, float theta)  // theta in [0,phi_l]
{
  theta = check_theta(theta, maps->phi_l);
  const float sin_theta = sinf(theta), sin_theta_sq = sin_theta * sin_theta,
              cos_theta = safe_sqrtf(
                  1.0f - sin_theta_sq);  // sign is positive because theta < phi_l < PI/2
  return safe_sqrtf(1.0f - maps->xe_sq * sin_theta_sq) - maps->xe * cos_theta;
}

ccl_device_inline float eval_rEll(const PSCMap *maps, float theta)  // theta in [0,pi/2]
{
  theta = check_theta(theta, M_PI_F);
  const float sin_theta = sinf(theta), sin_theta_sq = sin_theta * sin_theta;
  return maps->ax / safe_sqrtf(1.0f - maps->cos_beta_sq * sin_theta_sq);
}

ccl_device_inline float eval_rad_integrand(const PSCMap *maps, float theta)
{
  // ellipse only, or: ellipse+lune and theta above phi_l
  if (maps->fully_visible || (!maps->center_below_hor && maps->phi_l <= theta)) {
    const float re = eval_rEll(maps, theta);
    return 0.5f * re * re;
  }

  // lune only
  if (maps->center_below_hor) {
    if (theta <= maps->phi_l) {
      const float rc = eval_rCirc(maps, theta), re = eval_rEll(maps, theta);
      return 0.5f * (rc * rc - re * re);
    }
    else
      return 0.0f;
  }

  // ellipse+lune (theta is for sure below phi_l, see above)
  const float r = eval_rCirc(maps, theta);
  return 0.5f * r * r;
}

// normalized versions of Ar(C(\phi)) and the integrand (r_max^2-r_min^2)/2
ccl_device_inline float Ar_func(const PSCMap *maps, float theta, float Ar_max_value)
{
  return eval_Ar(maps, theta) / Ar_max_value;
}
ccl_device_inline float Ar_integrand(const PSCMap *maps, float theta, float Ar_max_value)
{
  return eval_rad_integrand(maps, theta) / Ar_max_value;
}

ccl_device_inline float InverseNSB(const PSCMap *maps,
                                   const float t_max,
                                   const float Aobj,
                                   const float A_max,
                                   const float Ar_max_value)
{
  const float A = max(0.0f, min(Aobj, A_max));  // A is always positive
  float tn = (A / A_max) * t_max,               // current best estimation of result value 't'
      tn_min = 0.0f,                            // current interval: minimum value
      tn_max = t_max;                           // current interval: maximum value

  int num_iters = 0;  // number of iterations so far

  float diff;

  while (true) {
    const float Ftn = Ar_func(maps, tn, Ar_max_value);
    diff = Ftn - A;

    // exit when done
    if (fabsf(diff) <= ini_iN_tolerance) {
      break;
    }

    // compute derivative, we must trunc it so it there are no out-of-range instabilities
    const float ftn = Ar_integrand(maps, tn, Ar_max_value);  // we know f(yn) is never negative
    const float delta = -diff / ftn;
    float tn_next = tn + delta;

    if (!isfinite_safe(tn_next) || (tn_next < tn_min || tn_max < tn_next)) {
      // tn_next out of range

      // update interval
      if (0.0 < diff) {
        // move to the left (current F(yn) is higher than desired)
        tn_max = tn;
      }
      else {
        // move to the right (current F(yn) is smaller than desired )
        tn_min = tn;
      }

      // update 'tn' according to the secant rule
      // 'tn' is in the range [tn_min,tn_max]
      // tn_next = ( tn_min*diff_tn_max - tn_max*diff_tn_min )/( diff_tn_max - diff_tn_min );

      // update tn by using current inteval midpoint
      tn_next = 0.5f * tn_max + 0.5f * tn_min;
    }

    tn = tn_next;
    num_iters++;

    // exit when the max number of iterations is exceeded
    if (ini_iN_max_iters < num_iters) {
      break;
    }
  }

  // done
  return tn;
}

ccl_device_inline float eval_Ar_inverse(const PSCMap *maps, float Ar_value)
{

  const float Ar_max_value = 0.5f * maps->F;

  Ar_value = max(0.0f, min(Ar_value, Ar_max_value));

  // for the ellipse only case, just do analytical inversion of the integral
  // (this in fact is never used as we do sampling in a scaled disk)
  if (maps->fully_visible) {
    return eval_ArE_inverse(maps, Ar_value);
  }

  const float A_frac = Ar_value / Ar_max_value;

  // in the ellipse+lune case, when Ar(phi_l) <= Ar_value, (result angle above phi_l)
  // we can and must do analytical inversion (for efficiency and convergence)
  if (!maps->center_below_hor) {
    if (maps->AE_phi_l + maps->L < Ar_value) {
      return eval_ArE_inverse(maps, Ar_value - maps->L);
    }
  }

  // in the lune only case, for small values of L, use a parabola approximation
  if (maps->center_below_hor) {
    if (maps->L < 1e-5f) {
      return maps->phi_l * (1.0f - safe_sqrtf(1.0f - A_frac));  // inverse parabola
    }
  }

  // -------
  // cases involving the lune: either lune only or ellipse+lune and Ar_value <= Ar_phi_l
  // do numerical iterative inversion:

  const float theta_max = maps->center_below_hor ? maps->phi_l : M_PI_F,
              theta_result = InverseNSB(maps, theta_max, A_frac, 1.0f, Ar_max_value);
  const float result = max(0.0f, min(theta_result, theta_max));

  return result;
}

ccl_device_inline void eval_rmin_rmax(const PSCMap *maps,
                                      const float theta,
                                      float *rmin,
                                      float *rmax)
{
  const float theta_c = min(theta, M_PI_F);

  // ellipse only, or: ellipse+lune and theta above phi_l
  if (maps->fully_visible || (!maps->center_below_hor && maps->phi_l <= theta_c)) {
    *rmin = 0.0f;
    *rmax = eval_rEll(maps, theta_c);
  }
  else if (maps->center_below_hor) {
    *rmin = eval_rEll(maps, theta_c);
    *rmax = eval_rCirc(maps, theta_c);
  }
  else {
    *rmin = 0.0f;
    *rmax = eval_rCirc(maps, theta_c);
  }
}

ccl_device_inline void eval(const PSCMap *maps, float s, float t, float *x, float *y)
{
  // compute 'u' by scaling and translating 't'
  const bool angle_is_neg = t < 0.5f;
  const float u = angle_is_neg ? 1.0f - 2.0f * t : 2.0f * t - 1.0f;

  // compute varphi in [0,1] from U by using inverse of Er, Lr or Ur
  float varphi, rmin, rmax;
  bool scaled = false;

  if (maps->fully_visible) {
    //varphi = eval_Er_inv( u*E );  // ellipse only
    varphi = M_PI_F * u;  // as we are in 'scaled' coord. space, this is simple...
    rmin = 0.0f;
    rmax = 1.0f;
    scaled = true;
  }
  else {
    varphi = max(0.0f, min(M_PI_F, eval_Ar_inverse(maps, u * 0.5f * maps->F)));
    eval_rmin_rmax(maps, varphi, &rmin, &rmax);
  }

  // compute x' and y'

  const float si = angle_is_neg ? -(sinf(varphi)) : sinf(varphi),
              co = safe_sqrtf(1.0f - si * si) *
                   (varphi <= M_PI_F * 0.5f ? 1.0f : -1.0f),  // note sign correction
      rad = safe_sqrtf(s * (rmax * rmax) + (1.0f - s) * (rmin * rmin)), xp = rad * co,
              yp = rad * si;

  // compute x and y
  if (scaled) {
    *x = maps->xe + maps->ax * xp;
    *y = maps->ay * yp;
  }
  else {
    *x = maps->xe + xp;
    *y = yp;
  }
}

ccl_device_inline void rad_map(const PSCMap *maps, float s, float t, float *x, float *y)
{
  // compute 'u' by scaling and translating 't'
  const bool angle_is_neg = t < 0.5f;
  const float u = angle_is_neg ? 1.0f - 2.0f * t : 2.0f * t - 1.0f;

  // compute varphi in [0,1] from U by using inverse of Er, Lr or Ur
  float varphi, rmin, rmax;
  bool scaled = false;

  if (maps->fully_visible) {
    //varphi = eval_Er_inv( u*E );  // ellipse only
    varphi = M_PI_F * u;  // as we are in 'scaled' coord. space, this is simple...
    rmin = 0.0f;
    rmax = 1.0f;
    scaled = true;
  }
  else {
    varphi = max(0.0f, min(M_PI_F, eval_Ar_inverse(maps, u * 0.5f * maps->F)));
    eval_rmin_rmax(maps, varphi, &rmin, &rmax);
  }

  // compute x' and y'

  const float si = angle_is_neg ? -(sinf(varphi)) : sinf(varphi),
              co = safe_sqrtf(1.0f - si * si) *
                   (varphi <= M_PI_F * 0.5f ? 1.0f : -1.0f),  // note sign correction
      rad = safe_sqrtf(s * (rmax * rmax) + (1.0f - s) * (rmin * rmin)), xp = rad * co,
              yp = rad * si;

  // compute x and y
  if (scaled) {
    *x = maps->xe + maps->ax * xp;
    *y = maps->ay * yp;
  }
  else {
    *x = maps->xe + xp;
    *y = yp;
  }
}

ccl_device_inline float eval_xEll(const PSCMap *maps, float y)
{
  y = check_y(y, maps->ay);
  return maps->ax * safe_sqrtf(1.0f - (y * y) / maps->ay_sq);
}
// --------------------------------------------------------------------------

ccl_device_inline float eval_xCir(const PSCMap *maps, float y)
{
  y = check_y(y, maps->yl);
  return safe_sqrtf(1.0f - y * y) - maps->xe;
}

ccl_device_inline void eval_xmin_xmax(const PSCMap *maps, float y, float *xmin, float *xmax)
{
  float yy = y;
  const float xell = eval_xEll(maps, yy);

  if (maps->fully_visible) {
    *xmin = maps->xe - xell;
    *xmax = maps->xe + xell;
  }
  else if (maps->center_below_hor) {
    *xmin = maps->xe + xell;
    *xmax = maps->xe + eval_xCir(maps, yy);
  }
  else {
    *xmin = maps->xe - xell;
    *xmax = (y <= maps->yl) ? maps->xe + eval_xCir(maps, yy) : maps->xe + xell;
  }
}

#if 0
ccl_device_inline float eval_Ap_inverse(const PSCMap *maps, float Ap_value)
{
  const float Ap_max_value = 0.5f*maps->F;

  Ap_value = max(0.0f, min(Ap_value, Ap_max_value));

  const float ymax = center_below_hor ? yl : ay;

  // normalized versions of functions: Ap(C(y)) and xmax(y)-xmin(y)
  auto Ap_func{ [=](float y) { return eval_Ap(y) / Ap_max_value;       } };
  auto Ap_integrand{ [=](float y) { return eval_par_integrand(y) / Ap_max_value; } };

  // do inversion, return clamped value
  const float y_result = InverseNSB<T>(Ap_func, Ap_integrand, ymax, Ap_value / Ap_max_value, T(1.0));
  const float result = max(0.0f, min(y_result, ymax));

  return result;
}

ccl_device_inline void hor_map(const PSCMap *maps, float s, float t, float *x, float *y)
{
  // compute 'u' by scaling and translating 't'
  const bool  y_is_neg = t < 0.5f;
  const float u = y_is_neg ? 1.0f - 2.0f*t : 2.0f*t - 1.0f;

  // compute the 'y' (positive), by inverting Ap function
  const float y_pos = eval_Ap_inverse(maps, u*0.5f*maps->F);

  // compute x's interval
  float xmin, xmax;
  eval_xmin_xmax(y_pos, xmin, xmax);

  // compute x and y
  x = (T(1.0) - s)*xmin + s * xmax;
  y = y_is_neg ? -y_pos : y_pos;
}
#endif

ccl_device_inline float sphere_light_projected_sa_eval(
    float alpha, float beta, float randu, float randv, float *x, float *y, bool p_use_radial)
{
  PSCMap maps;
  PSCMap_initialize(&maps, alpha, beta, p_use_radial);

  //if(maps.using_radial)
  rad_map(&maps, randu, randv, x, y);
  //else
  //  hor_map(&maps, randu, randv, x, y);
  return maps.F;
}

ccl_device_inline float uniform_cone_pdf(float cos_theta_max)
{
  return cos_theta_max < 1.0f ? 1.0f / (2.0f * M_PI_F * (1.0f - cos_theta_max)) : 0.0f;
}

CCL_NAMESPACE_END
