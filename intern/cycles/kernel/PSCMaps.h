// *********************************************************************
// **
// ** Projected Spherical Cap Maps
// ** template class 'PSCMaps' (class and aux funcs declaration)
// **
// ** Copyright (C) 2018 Carlos Ureña and Iliyan Georgiev
// **
// ** Licensed under the Apache License, Version 2.0 (the "License");
// ** you may not use this file except in compliance with the License.
// ** You may obtain a copy of the License at
// **
// **    http://www.apache.org/licenses/LICENSE-2.0
// **
// ** Unless required by applicable law or agreed to in writing, software
// ** distributed under the License is distributed on an "AS IS" BASIS,
// ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// ** See the License for the specific language governing permissions and
// ** limitations under the License.

#ifndef PSCMAPS_H
#define PSCMAPS_H

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>   // std::setprecision
#include <cassert>
#include <algorithm>   // std::max and others
#include <functional>
#include <cmath>
#include <string>
#include <fstream>


namespace PSCM
{

// -----------------------------------------------------------------------------
// types

template< class T > using FuncType = std::function< T(T) > ;

// -----------------------------------------------------------------------------
// constants (evaluated at compile time)

// controls whether  assertions are checked
// (it also prevents tracing from actually happening)
constexpr bool do_checks = true ;

// when do_checks == true, used to control when two values are approximately equal
constexpr auto epsilon = 1e-6 ;

// initial value for the tolerance of the inverse newton function
constexpr double ini_iN_tolerance = 1e-4 ;

// initial value for the max number of iterations in the inverse newton func.
constexpr int ini_iN_max_iters = 20 ;

// -----------------------------------------------------------------------------
// A class for projected spherical cap maps evaluation state

template< typename T>  // T == float, double, long double, etc....
class PSCMaps
{
   public:

   // Creates an uninitialized 'empty' object (not usable)
   PSCMaps();

   // Initializes this maps object
   // p_alpha and p_beta are the angles defining the spherical cap (see paper)
   // it must hold: 0 < p_alpha
   // p_use_radial == true --> use radial map, == false --> use parallel map
   void initialize( const T p_alpha, const T p_beta, const bool p_use_radial );

   // evaluates one of the two maps (according to 'using_radial')
   // (s,t) must be in [0,1]^2
   void eval_map( T s, T t, T &x, T &y ) const ;

   // returns the area of the projected spherical cap (straight inline returns)
   inline T get_area();

   // true iif  the sampler has been initialized via 'initialize'
   // any other method should not be called for uninitialized objects
   inline bool is_initialized() ;

   // true if the radial map is in use
   inline bool is_using_radial();

   // query spherical map status (straight inline returns)
   inline bool is_fully_visible();     // true iif  0 <= alpha <= beta  (sphere fully visible)
   inline bool is_partially_visible(); // true iif  -alpha <= beta <= alpha (sphere partially visible)
   inline bool is_invisible() ;        // true iif  beta <= -alpha <= 0 (sphere fully invisible)
   inline bool is_center_below_hor();  // true iif  beta <= 0 (spherical cap center is below horizon)

   // functions for querying the spherical cap parameters
   // (straight inline returns)

   inline T get_xe() ;    // xe: ellipse center
   inline T get_ax() ;    // ax: ellipse semi-minor axis length (width)
   inline T get_ay() ;    // ay: ellipse semi-major axis length (height)
   inline T get_xl() ;    // xl: X coord. of lune-ellipse tangency points (only when partially visible)
   inline T get_yl() ;    // yl: Y coord. of lune-ellipse tangency points (only when partially visible)
   inline T get_phi_l() ; // phi_l: tangency points angle (only when partially visible and using radial map)

   // functions for evaluating the integrals and their inverses

   // eval the parallel integral (Ap), the integrand and inverse integral (Ap^{-1))
   T eval_Ap( T y ) const;           // evals. Ap (see eq. 15) (y in [0,ey])
   T eval_par_integrand( T y ) const;// evals. integrand of eq. 12 (y in [0,ey])
   T eval_Ap_inverse( T Ap ) const ; // evals. Ap inverse (Ap in [0,area/2]), iteratively

   // evals the radial integral (Ar), the integrand and inverse integral (Ar^{-1))
   T eval_Ar( T theta ) const ;            // evals. Ar (see eq. 23) (theta in [0,PI])
   T eval_rad_integrand( T theta ) const ; // evals. integrand of eq 20 (theta in [0,PI])
   T eval_Ar_inverse( T Ar_value ) const ; // evals. Ar inverse (Ar in [0,area/2]), iteratively (when needed)


   // test the area integrals: compares numerical and analytical integration
   void run_test_integrals(  );

   // prints debug info about this spherical cap status and parameters
   void debug() ;

   // --------------------------------------------------------------------------
   private:

   inline void ensure_initialized();
   inline void ensure_using_radial();
   inline void ensure_using_parallel();

   // aux. methods
   void compute_ELF_xlyl_phi_l();

   // --------------------------------------------------------------------------
   // Horizontal map related methods:

   // checks that the sphere is partially visible
   // and that 'y' is in the range (0,y_limit)
   // truncates 'y' if it is slightly off-range (to within epsilon)
   void check_y( T & y, const T & y_limit  ) const ;

   // eval X coord. of intersection between an horizontal line at 'y' and
   // the ellipse or a circle (see )
   T eval_xEll( T y ) const ;  // hor.line/ellipse intersection
   T eval_xCir( T y ) const;   // hor.line/circle intersection

   // for a given y, compute xmin and xmax
   // y must be in (0,ey), but if center is below horizon, it must be in (0,yl)
   void eval_xmin_xmax( const T y, T & xmin, T & xmax ) const ;

   // compute area of the region of the ellipse/circle under an horizontal
   // line at heigh 'y' (see equation 16 in the paper)
   T eval_ApE( T y ) const; // function ApE, area in the ellipse
   T eval_ApC( T y ) const; // function ApC, area in the circle

   // Evaluates horizontal map: computes (x,y) from (s,t)
   // ('using_radial' must be false, (s,t) must be in [0,1]^2 )
   void hor_map( T s, T t, T &x, T &y ) const; // see equations 18 and 19

   // --------------------------------------------------------------------------
   // Radial map

   // checks that the sphere is partially visible
   // and that 'theta' is in the range (0,max_theta)
   // truncates 'theta' if it is slightly off-range (to within epsilon)
   void check_theta( T & theta, const T & max_theta ) const ;

   // eval rEll and rCir (length of radius at 'theta' within circle/ellipse)
   T eval_rEll( T theta ) const;   // theta in [0,pi]
   T eval_rCirc( T theta ) const;  // theta in [0,phi_l]

   // for a given theta, compute rmin and rmax
   // y must be in (0,PI), but if center is below horizon, it must be in (0,phi_l)
   void eval_rmin_rmax( const T y, T & rmin, T & rmax ) const ;

   // compute area of the region of the ellipse/circle under an ellipse
   // radius at angle 'theta' (see equation 24 and 25 in the paper)
   T eval_ArE( T theta ) const;   // see eq. 24 (theta in [0,pi])
   T eval_ArC( T theta ) const;   // see eq. 25 (theta in [0,phi_l])

   // evaluates ArE inverse, by using an exact analytic expresion
   T eval_ArE_inverse( T Ar_value ) const ;

   // Evaluates the radial map: computes (x,y) from (s,t)
   // ('using_radial' must be true, (s,t) must be in [0,1]^2 )
   void rad_map( T s, T t, T &x, T &y ) const ;

   // --------------------------------------------------------------------------

   bool // values defining the spherical cap type, and which map is being used
      initialized ,      // true if the sampler has been initialized
      fully_visible,     // true iif r <= cz     (sphere fully visible)
      partially_visible, // true iif -r < cz < r (sphere partially visible)
      center_below_hor,  // true iif -r <= cz < 0 (partially visible and sphere center below horizon)
      invisible ,        // true iif cz <= -r    (sphere completely invisible)
      using_radial;      // true when using radial map, false when using horizontal map

   T // areas (form factors)
      E,      // half of the ellipse area
      L ,     // half lune area, (it is 0 in the 'ellipse only' case)
      F ;     // total form factor: it is:
              //    0 -> when invisible
              //    2E -> when fully_visible (ellipse only)
              //    2L -> when partially_visible and center_below_hor (lune only)
              //    2(E+L) -> otherwise (partially_visible and not center_below_hor)

   T  // sinus and cosine of beta
      cos_beta,
      cos_beta_sq,
      sin_beta,
      sin_beta_abs ;

   T  // parameters defining the ellipse
      xe ,    // X coord. of ellipse center in the local reference frame
      ax,     // ellipse semi-minor axis (X direction)
      ay ;    // ellipse semi-major axis (Y direction)

   T // some precomputed constants
      axay2 , // == (ax*ay)/2
      ay_sq , // == ay^2
      xe_sq , // == xe^2
      r1maysq ; // root of (1-ay^2)

   T // parameters computed only when partially_visible ( there are tangency points)
      xl,        // X coord. of tangency point p (>0)
      yl ,       // Y coord. of tangency point p (>0)
      phi_l,     // == arctan(yl/(xe-xl)), only if 'using_radial' (and 'partially_visible')
      AE_phi_l ; // == A_E(phi_l), , only if 'using_radial' (and 'partially_visible')

} ;  // end class PSCMaps

// -----------------------------------------------------------------------------
// class 'global' vars and constants (all members are static)

template< class T > class Vars
{
   public:

   static T
      iN_tolerance ; // tolerance for inverse newton....
   static int
      iN_max_iters ; // max iters. for inv. Newton
   static bool
      trace_newton_inversion ;

   static void print_settings();
} ;

// initialization of global vars

template< class T > bool  Vars<T>::trace_newton_inversion = false ;
template< class T > T     Vars<T>::iN_tolerance           = T(ini_iN_tolerance) ;
template< class T > int   Vars<T>::iN_max_iters           = ini_iN_max_iters ;

// *****************************************************************************
// aux. functions

// --------------------------------------------------------------------------
// eval I function, according to expression 17 in the paper

template< class T >
T eval_I( T u, T w ) ;

// -----------------------------------------------------------------------------
// InverseNSB
//
// evaluate the inverse of a function whose derivative is > 0
//  Uses Newton method for root finding when possible,
//  and a combination of Binary search  and Secant methods
//  (at each step the best option is selected)
//
// returns the value 't' (in the range [0,t_max]) such that F(t) = Aobj
// it is assumed that F(t) in [0,A_max] for t in [0,t_max]
//                    0.0 <= Aobj <= A_max
//                    F(0.0) == 0.0, and F(t_max) == A_max
//                    F' = f
// where
//    F    : function whose inverse value we want to compute
//    f    : derivative of F  (>0 in [0,t_max])
//    t_max: maximum value for the result 't'
//    Aobj : desired value of F(t)
//    A_max: maximum value for A (minimum is 0.0)
//

template< class T >
T InverseNSB( FuncType<T> F, FuncType<T> f,
              const T t_max, const T Aobj, const T A_max ) ;

// ---------------------------------------------------------------------
// numerically integrate a real function on a real interval (x0,x1),
// by using 'n' equispaced samples

template< class T >
T numeric_integral( const std::function<T(T)> & f, T x0, T x1, int n ) ;

//****************************************************************************
// Implementation of all methods

using namespace std ;

// --------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_area()
{
   ensure_initialized();
   return F ;
}
// -------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_xe()
{
   ensure_initialized();
   return xe ;
}
// -------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_ax()
{
   ensure_initialized();
   return ax ;
}
// -------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_ay()
{
   ensure_initialized();
   return ay ;
}
// -------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_xl()
{
   ensure_initialized();
   return xl ;
}
// -------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_yl()
{
   ensure_initialized();
   return yl ;
}
// -------------------------------------------------------------------------

template< class T >
inline T PSCMaps<T>::get_phi_l()
{
   ensure_initialized();
   if ( do_checks )
      assert( partially_visible );
   return phi_l ;
}
// -------------------------------------------------------------------------

template< class T >
inline bool PSCMaps<T>::is_initialized()
{
   return initialized ;
}
// -------------------------------------------------------------------------

template< class T >
inline bool PSCMaps<T>::is_invisible()
{
   ensure_initialized();
   return invisible ;
}
// -------------------------------------------------------------------------

template< class T >
inline bool PSCMaps<T>::is_partially_visible()
{
   ensure_initialized();
   return partially_visible ;
}
// -------------------------------------------------------------------------

template< class T >
inline bool PSCMaps<T>::is_fully_visible()
{
   ensure_initialized();
   return fully_visible ;
}
// -------------------------------------------------------------------------

template< class T >
inline bool PSCMaps<T>::is_center_below_hor()
{
   ensure_initialized();
   return center_below_hor ;
}
// -------------------------------------------------------------------------

template< class T >
inline bool PSCMaps<T>::is_using_radial()
{
   ensure_initialized();
   return using_radial ;
}

// --------------------------------------------------------------------------
// eval I function, according to expression 17 in the paper

template< class T >
T eval_I( T u, T w )
{
   if ( do_checks )
   {
      assert( epsilon < w );
      assert( T(0.0) <= u );
   }
   return 0.5*( w*u*std::sqrt(1.0-u*u) + std::asin(u) ) ; // expresion 17
}
// --------------------------------------------------------------------------
// Creates an uninitialized 'empty' object (not usable)

template< class T >
PSCMaps<T>::PSCMaps( )
{
   initialized = false ;
   E = 0.0 ;
   L = 0.0 ;
   F = 0.0 ;
}
// --------------------------------------------------------------------------
// various checking functions

template< class T >
inline void PSCMaps<T>::ensure_initialized()
{
   if ( do_checks )
      assert( initialized );
}
// --------------------------------------------------------------------------

template< class T >
inline void PSCMaps<T>::ensure_using_radial()
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( using_radial );
   }
}
// --------------------------------------------------------------------------

template< class T >
inline void PSCMaps<T>::ensure_using_parallel()
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( ! using_radial );
   }
}
// --------------------------------------------------------------------------
// initializes the maps object

template< class T >
void PSCMaps<T>::initialize( const T p_alpha, const T p_beta,
                             const bool p_use_radial )
{
   constexpr T tolerance = 1e-5 ,
               pi2       = T(M_PI)*T(0.5) ;

   if ( do_checks )
   {
      if ( pi2+tolerance < p_alpha )
         cout << "p_alpha == " << p_alpha << ", pi2+tolerance == " << pi2+tolerance << endl ;
      assert( p_alpha <= pi2 + tolerance );
      assert( p_beta  <= pi2 + tolerance );
      assert( -tolerance  <= p_alpha );
      assert( -pi2-tolerance  <= p_beta );
   }

   const T alpha = std::max( T(0.0), std::min( pi2, p_alpha ) ),
           beta  = std::max( -pi2,   std::min( pi2, p_beta  ) );

   initialized = false ;

   E = 0.0 ;
   L = 0.0 ;
   F = 0.0 ;

   ay           = std::sin( alpha );
   ay_sq        = ay*ay ;
   r1maysq      = std::sqrt( T(1.0)-ay_sq ) ;
   sin_beta     = std::sin( beta ),
   sin_beta_abs = std::abs( sin_beta );
   cos_beta_sq  = T(1.0)- sin_beta*sin_beta ;
   cos_beta     = std::sqrt( cos_beta_sq );  // spherical cap center, X coord.
   xe           = cos_beta*r1maysq ;     // ellipse center
   ax           = ay*sin_beta_abs ;     // semi-minor axis length (UNSIGNED)
   axay2        = ax*ay*T(0.5);
   xe_sq        = xe*xe ;

   // intialize boolean values
   using_radial      = p_use_radial ;
   fully_visible     = false ;
   partially_visible = false ;
   invisible         = false ;
   center_below_hor  = false ;

   if ( ay <= sin_beta )      fully_visible     = true ;
   else if ( -ay < sin_beta ) partially_visible = true ;
   else                       invisible         = true ;

   center_below_hor = false ;
   if ( sin_beta < T(0.0) )
      center_below_hor = true ;

   // if the sphere is not visible, mark is initialized, and do an early exit
   if ( invisible )
   {
      initialized = true ;
      return ;
   }
   if ( do_checks )
      assert( fully_visible || partially_visible );

   // mark this instance as already initialized (needed to precompute values)
   initialized = true ;

   // pre-compute some values
   compute_ELF_xlyl_phi_l();

   if ( do_checks )
   {
      // check cos_beta is in [0,1], cy == 0, and cos_beta^2+sin_beta^2 == 1

      assert( -epsilon <= cos_beta && cos_beta <= T(1.0)+epsilon );
      assert( std::abs( cos_beta*cos_beta+sin_beta*sin_beta-T(1.0) ) < epsilon );

      // check ax and ay are both positive and smaller than one
      assert( -epsilon <= ax && ax <= T(1.0)+epsilon );
      assert( -epsilon <= ay && ay <= T(1.0)+epsilon );
      assert( ax <= ay );

      if ( partially_visible )
      {
         assert( -epsilon <= xl && xl <= T(1.0)+epsilon );
         assert( -epsilon <= yl && yl <= T(1.0)+epsilon );
      }
   }
}

// --------------------------------------------------------------------------
// compute xl and yl if there area tangency points, and phi_l
// compute areas: E,L and F (they are initialized previously to 0.0)

template< class T > inline
void PSCMaps<T>::compute_ELF_xlyl_phi_l()
{
   // initialize areas
   E = 0.0 ;
   L = 0.0 ;
   F = 0.0 ;
   AE_phi_l = 0.0 ;

   if ( partially_visible )
   {
      xl = r1maysq/cos_beta ;
      yl  = std::sqrt(T(1.0)-xl*xl);

      // compute L
      if ( using_radial )
      {
         phi_l    = std::atan2( yl, xl-xe );
         AE_phi_l = eval_ArE( phi_l );
         L        = eval_ArC( phi_l ) - AE_phi_l ;
      }
      else
         L = eval_ApC( yl ) - eval_ApE( yl );

      if ( L < T(0.0) )
         L = T(0.0) ;

      // compute E
      if ( ! center_below_hor ) // ellipse only or ellipse+lune cases (both maps)
         E = T(M_PI)*axay2 ;  // half ellipse area


   }
   else
      E = T(M_PI)*axay2 ;  // half ellipse area

   // compute F
   F = T(2.0)*(E+L) ;

   if ( F < 1e-6 )
      invisible = true ;
}

// --------------------------------------------------------------------------
// checks that the sphere is partially visible and that 'y' is in the proper range
// it also checks y and truncates if it is slightly off-range (to within epsilon)
// (the range is [0,max_y])

template< class T > inline
void PSCMaps<T>::check_y( T & y, const T & max_y ) const
{
   if ( do_checks )
   {
      assert( initialized ) ;
      assert( !invisible );
      assert( !using_radial );
      assert( -epsilon <= y );
      assert( y <= max_y+epsilon );
   }
   y = std::max( T(0.0), std::min( y, max_y ));
}

// --------------------------------------------------------------------------

// eval X on the ellipse and on the circle, for a given y
// y must be in [0,yl].

template< class T >
T PSCMaps<T>::eval_xEll( T y ) const
{
   if ( do_checks )
      assert( initialized && ! using_radial );

   check_y( y, ay );
   return ax*std::sqrt( T(1.0) - (y*y)/ay_sq ) ;
}
// --------------------------------------------------------------------------

template< class T >
T PSCMaps<T>::eval_xCir( T y ) const
{
   if ( do_checks )
      assert( initialized && ! using_radial && partially_visible );

   check_y( y, yl );
   return std::sqrt( T(1.0)-y*y ) - xe ;
}

// --------------------------------------------------------------------------
// eval ApE according to expression 16 in the paper
// y must be in [0,ay]

template< class T >
T PSCMaps<T>::eval_ApE( T y ) const
{
   if ( do_checks )
      assert( initialized && ! using_radial );

   check_y( y, ay );
   const T v = std::max( T(0.0), std::min( T(1.0), y/ay ));
   return ax*ay*eval_I( v, T(1.0) );  // expression 16
}
// --------------------------------------------------------------------------
// eval AcE according to expression 16 in the paper
// y must be in [0,yl]

template< class T >
T PSCMaps<T>::eval_ApC( T y ) const
{
   if ( do_checks )
      assert( initialized && ! using_radial && partially_visible );

   check_y( y, yl );
   return eval_I( y, T(1.0) ) - xe*y;    // expresion 16
}

// ---------------------------------------------------------------------------
// evals the parallel map area (equivalent to eq.15, but optimized)
// y in [0,ay]

template< class T >
T PSCMaps<T>::eval_Ap( T y ) const
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( ! using_radial );
      assert( T(0.0) <= y );
      assert( y  <= ay+epsilon );
   }

   // ellipse only
   if ( fully_visible )
      return T(2.0)*eval_ApE( y );

   // lune only
   if ( center_below_hor )
      return ( y <= yl )
         ? (eval_ApC( y ) - eval_ApE( y ))
         : L ;

    // ellipse+lune, y below 'yl'
   if  ( y <= yl )
      return eval_ApC( y ) + eval_ApE( y ) ;

   // ellipse + lune, y above 'yl'
   return
      T(2.0)*eval_ApE( y ) + L ;
}

// ---------------------------------------------------------------------------
// evals the parallel integrand (simpler, unified)
// y in [0,ay]

template< class T >
T PSCMaps<T>::eval_par_integrand( T y ) const
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( ! using_radial );
      assert( T(0.0) <= y );
      assert( y  <= ay );
   }

   // ellipse only
   if ( fully_visible )
      return T(2.0)*eval_xEll( y );

   // lune only
   if ( center_below_hor )
      return ( y <= yl )
         ? (eval_xCir( y ) - eval_xEll( y ))
         : T(0.0) ;

    // ellipse+lune
   return  ( y <= yl )
         ? eval_xCir( y ) + eval_xEll( y )
         : T(2.0)*eval_xEll( y )  ;
}

// --------------------------------------------------------------------------
// for a given y, eval x0 and x1
// y must be in (0,ay), but if center is below horizon, it must be in (0,yl)
template< class T >
void PSCMaps<T>::eval_xmin_xmax( const T y, T & xmin, T & xmax ) const
{
   T yy = y ;
   if ( do_checks )
   {
      assert( initialized && !using_radial );
      const T y_limit = center_below_hor ? yl : ay ;
      check_y( yy, y_limit );
   }

   const float xell = eval_xEll( yy );

   if ( fully_visible ) // ellipse only
   {
      xmin = xe - xell ;
      xmax = xe + xell ;
   }
   else if ( center_below_hor )  // lune only
   {
      xmin = xe + xell ;
      xmax = xe + eval_xCir( yy );
   }
   else // ellipse plus lune
   {
      xmin = xe - xell ;
      xmax = ( y <= yl ) ? xe+eval_xCir( yy ) : xe+xell ;
   }
}

// ---------------------------------------------------------------------------
// evals the parallel map inverse integral
// I in [0,F/2]

template< class T >
T PSCMaps<T>::eval_Ap_inverse( T Ap_value ) const
{
   const T Ap_max_value = T(0.5)*F ;

   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( ! using_radial );
      assert( T(0.0)-epsilon <= Ap_value );
      assert( Ap_value <= Ap_max_value+epsilon );
   }
   Ap_value = std::max( T(0.0), std::min( Ap_value, Ap_max_value ));

   const T ymax = center_below_hor ? yl : ay ;

   // normalized versions of functions: Ap(C(y)) and xmax(y)-xmin(y)
   auto Ap_func      { [=]( float y ) { return eval_Ap( y )/Ap_max_value;       } } ;
   auto Ap_integrand { [=]( float y ) { return eval_par_integrand( y )/Ap_max_value ; } } ;

   // do inversion, return clamped value
   const T y_result = InverseNSB<T>( Ap_func, Ap_integrand, ymax, Ap_value/Ap_max_value, T(1.0) );
   const T result = std::max( T(0.0), std::min( y_result, ymax ));

   return result ;
}


// ---------------------------------------------------------------------------
// horizontal map: computes (x,y) from (s,t)
template< class T >
void PSCMaps<T>::hor_map( T s, T t, T &x, T &y ) const
{

   if ( do_checks )
   {
      assert( initialized );
      assert( !invisible );
      assert( !using_radial );
      assert( fully_visible || partially_visible );
      assert( T(0.0) <= s && s <= T(1.0) );
      assert( T(0.0) <= t && t <= T(1.0) );
   }

   // compute 'u' by scaling and translating 't'
   const bool  y_is_neg = t < T(0.5)   ;
   const float u        = y_is_neg ? T(1.0)-T(2.0)*t
                                   : T(2.0)*t - T(1.0) ;

   // compute the 'y' (positive), by inverting Ap function
   const T y_pos = eval_Ap_inverse( u*T(0.5)*F );

   // compute x's interval
   T xmin, xmax ;
   eval_xmin_xmax( y_pos, xmin, xmax );

   // compute x and y
   x = (T(1.0)-s)*xmin + s*xmax ;
   y = y_is_neg ? -y_pos : y_pos ;
}
// ****************************************************************************
// Radial map related methods

// checks that the sphere is partially visible
// and that 'theta' is in the range (0,phi_l) (computes phi_l)
// truncates 'theta' if it is slightly off-range (to within epsilon)

template< class T > inline
void PSCMaps<T>::check_theta( T & theta, const T & max_theta ) const
{
   if ( do_checks )
   {
      assert( initialized );
      assert( using_radial );
      assert( -epsilon  <= theta );
      assert( theta <= max_theta+epsilon );
   }
   theta = std::max( T(0.0), std::min( theta, max_theta )); // trunc
}
// ---------------------------------------------------------------------------
// eval 're(theta)' according to equation 45

template< class T >
T PSCMaps<T>::eval_rEll( T theta ) const   // theta in [0,pi/2]
{
   if ( do_checks )
      assert( initialized && using_radial );

   check_theta( theta, T(M_PI) );

   const T sin_theta    = sin(theta),
           sin_theta_sq = sin_theta*sin_theta ;
   return ax / std::sqrt( T(1.0)-cos_beta_sq* sin_theta_sq );
}

// ---------------------------------------------------------------------------
// eval 'rc(theta)' according to equation 48

template< class T >
T PSCMaps<T>::eval_rCirc( T theta ) const   // theta in [0,phi_l]
{
   if ( do_checks )
      assert( initialized && using_radial && partially_visible );

   check_theta( theta, phi_l );

   const T sin_theta    = sin(theta),
           sin_theta_sq = sin_theta*sin_theta ,
           cos_theta    = std::sqrt( T(1.0) - sin_theta_sq ); // sign is positive because theta < phi_l < PI/2
   return std::sqrt( T(1.0)- xe_sq * sin_theta_sq ) - xe*cos_theta ;
}
// ---------------------------------------------------------------------------
// evaluate Re according to expression 50  (sec.4)
// theta in [0,pi]
template< class T >
T PSCMaps<T>::eval_ArE( T theta ) const
{

   if ( do_checks )
      assert( initialized && using_radial );

   check_theta( theta, T(M_PI) );

   const T tt = std::abs(std::tan(theta));

   constexpr T pi2 = T(0.5)*M_PI ;
   T result ;
   if ( theta <= pi2 )
      result = axay2*std::atan( sin_beta_abs*tt );
   else
      result = axay2*( M_PI - std::atan( sin_beta_abs*tt ));

   return result ;
}

// ---------------------------------------------------------------------------
// evaluate ArC according to equation 25
// theta in [0,phi_l]

template< class T >
T PSCMaps<T>::eval_ArC( T theta ) const
{
   if ( do_checks )
      assert( initialized && using_radial && partially_visible );

   check_theta( theta, phi_l );

   if ( do_checks )
      assert( T(0.0) <= theta && theta <= T(M_PI)*T(0.5) );

   const T z      = std::sin( theta ), // z == asin(theta)
           xez    = xe*z,
           z_sq   = z*z,
           xe_sq  = xe*xe ;


   // this is eq. 25 in the paper:
   // return eval_I( z, xe_sq ) - eval_I( xe*z, T(1.0) );

   // this is equivalent to, but faster than, eq. 25
   // (an 'asin' call is saved)
   return T(0.5)*
           ( theta
             - std::asin(xez)
             + xe_sq*z*std::sqrt( T(1.0)-z_sq )
             - xez*std::sqrt( T(1.0)- xe_sq*z_sq)
           );

}

// ---------------------------------------------------------------------------
// evals the radial integral, an optimized version of equation 25
// theta in [0,pi]

template< class T >
T PSCMaps<T>::eval_Ar( T theta ) const
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( using_radial );
      assert( T(0.0) <= theta );
      assert( theta  <= T(M_PI) );
   }

   // ellipse only
   if ( fully_visible )
      return eval_ArE( theta  );

   // lune only
   if ( center_below_hor )
      return (theta <= phi_l)
         ? (eval_ArC( theta ) - eval_ArE( theta ))
         : L ;

    // ellipse+lune
   return  ( theta <= phi_l )
         ? eval_ArC( theta )
         : eval_ArE( theta ) + L ;
}

// ---------------------------------------------------------------------------
// evals the radial integrand, as defined in equation 20,
//   it is implemented in terms of 'eval_rEll' and 'eval_rCirc'
template< class T >
T PSCMaps<T>::eval_rad_integrand( T theta ) const
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( using_radial );
      assert( T(0.0) <= theta );
      assert( theta  <= T(M_PI) );
   }

   // ellipse only, or: ellipse+lune and theta above phi_l
   if ( fully_visible || ( !center_below_hor && phi_l <= theta ) )
   {
      const T re = eval_rEll( theta ) ;
      return T(0.5)*re*re ;
   }

   // lune only
   if ( center_below_hor )
   {
      if ( theta <= phi_l )
      {
         const T rc = eval_rCirc( theta ),
                 re = eval_rEll( theta );
         return T(0.5)*( rc*rc - re*re );
      }
      else
         return T(0.0) ;
   }

   // ellipse+lune (theta is for sure below phi_l, see above)
   const T r = eval_rCirc( theta ) ;
   return T(0.5)*r*r ;

}
// ---------------------------------------------------------------------------

template< class T >
void PSCMaps<T>::eval_rmin_rmax( const T theta, T & rmin, T & rmax ) const
{
   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( using_radial );
      assert( T(0.0) <= theta );
      if ( T(M_PI) + epsilon < theta )
         cout << "theta-PI == " << theta-T(M_PI) << endl ;
      assert( theta  <= T(M_PI)+epsilon );
   }
   const T theta_c = std::min( theta, T(M_PI) );

   // ellipse only, or: ellipse+lune and theta above phi_l
   if ( fully_visible || ( !center_below_hor && phi_l <= theta_c ) )
   {
      rmin = T(0.0);
      rmax = eval_rEll( theta_c );
   }
   else if ( center_below_hor ) // lune only
   {
      if ( do_checks )
         assert( theta <= phi_l );
      rmin = eval_rEll( theta_c );
      rmax = eval_rCirc( theta_c );
   }
   else // ellipse+lune (theta is for sure below phi_l, see above)
   {
      if ( do_checks )
         assert( theta <= phi_l );
      rmin = T(0.0);
      rmax = eval_rCirc( theta_c ) ;
   }

}
// ---------------------------------------------------------------------------

template< class T >
T PSCMaps<T>::eval_Ar_inverse( T Ar_value ) const
{

   if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
      cout << "eval_Ar_inverse: Ar_value == " << Ar_value << endl ;

   const T Ar_max_value = T(0.5)*F ;

   if ( do_checks )
   {
      assert( initialized );
      assert( ! invisible );
      assert( using_radial );
      assert( T(0.0)-epsilon <= Ar_value );
      assert( Ar_value <= Ar_max_value+epsilon );
   }

   Ar_value = std::max( T(0.0), std::min( Ar_value, Ar_max_value ));

   // for the ellipse only case, just do analytical inversion of the integral
   // (this in fact is never used as we do sampling in a scaled disk)
   if ( fully_visible )
      return eval_ArE_inverse( Ar_value );

   const T A_frac = Ar_value/Ar_max_value ;

   // in the ellipse+lune case, when Ar(phi_l) <= Ar_value, (result angle above phi_l)
   // we can and must do analytical inversion (for efficiency and convergence)
   if ( ! center_below_hor )
   if ( AE_phi_l + L < Ar_value )
      return eval_ArE_inverse( Ar_value - L ) ;


   // in the lune only case, for small values of L, use a parabola approximation
   if ( center_below_hor )  // !fully visible and center below hor., --> lune only
   if ( L < 1e-5 )          // small lune area
   {
      if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
         cout << "eval_Ar_inverse: doing parabola approximation" << endl ;
      return phi_l*( T(1.0)-std::sqrt( T(1.0)-A_frac ) );  // inverse parabola
   }

   // -------
   // cases involving the lune: either lune only or ellipse+lune and Ar_value <= Ar_phi_l
   // do numerical iterative inversion:

   if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
      cout << "eval_Ar_inverse: doing numeric inversion" << endl ;

   // normalized versions of Ar(C(\phi)) and the integrand (r_max^2-r_min^2)/2
   auto Ar_func      { [=]( float theta ) { return eval_Ar( theta )/Ar_max_value;       } } ;
   auto Ar_integrand { [=]( float theta ) { return eval_rad_integrand( theta )/Ar_max_value ; } } ;

   const T
      theta_max    = center_below_hor ? phi_l : T(M_PI) ,
      theta_result = InverseNSB<T>( Ar_func, Ar_integrand,
                                   theta_max, A_frac, T(1.0));
   const T result = std::max( T(0.0), std::min( theta_result, theta_max ));

   return result ;
}

// --------------------------------------------------------------------------
// evaluation of the inverse area integral, for the radial map, in the ellipse
// only (full visible) case
//
template< class T >
T PSCMaps<T>::eval_ArE_inverse( T Ar_value ) const
{
   const T Ar_max_value = E ;

   if ( do_checks )
   {
      assert( initialized );
      assert( !invisible );
      assert( using_radial );
      //assert( fully_visible );
      assert( Ar_value <= Ar_max_value +epsilon );
      assert( T(0.0)-epsilon <= Ar_value );
   }
   Ar_value = std::max( T(0.0), std::min( Ar_value, Ar_max_value ) );

   const T ang = Ar_value/axay2 ;
   constexpr T pi2 = T(0.5)*T(M_PI) ;

   if ( ang <= pi2 )
      return std::atan(std::tan(ang)/sin_beta_abs);
   else
      return T(M_PI) + std::atan(std::tan(ang)/sin_beta_abs);
}
// ---------------------------------------------------------------------------
// radial map: computes (x,y) from (s,t)
template< class T >
void PSCMaps<T>::rad_map( T s, T t, T &x, T &y ) const
{

   if ( do_checks )
   {
      assert( initialized );
      assert( using_radial );
      assert( !invisible );
      assert( fully_visible || partially_visible );
      assert( T(0.0) <= s && s <= T(1.0) );
      assert( T(0.0) <= t && t <= T(1.0) );
   }

   // compute 'u' by scaling and translating 't'
   const bool  angle_is_neg = t < T(0.5)   ;
   const float u            = angle_is_neg ? T(1.0)-T(2.0)*t
                                           : T(2.0)*t - T(1.0) ;

   // compute varphi in [0,1] from U by using inverse of Er, Lr or Ur

   T varphi, rmin, rmax ;
   bool scaled = false ;

   if ( fully_visible )
   {
      //varphi = eval_Er_inv( u*E );  // ellipse only
      varphi = M_PI*u ; // as we are in 'scaled' coord. space, this is simple...
      rmin = T(0.0);
      rmax = T(1.0);
      scaled = true ;
   }
   else // partially visible
   {
      varphi = std::max( T(0.0), std::min( T(M_PI),
                  eval_Ar_inverse( u*T(0.5)*F  ) ));
      eval_rmin_rmax( varphi, rmin, rmax );
   }

   // compute x' and y'

   const T si  = angle_is_neg ? -(std::sin(varphi)) : std::sin( varphi ),
           co  = std::sqrt( T(1.0) - si*si )
                     * ( varphi <= T(M_PI)*T(0.5) ? T(1.0) : T(-1.0) ), // note sign correction
           rad = std::sqrt( s*(rmax*rmax) + (T(1.0)-s)*(rmin*rmin)  ),
           xp  = rad*co ,
           yp  = rad*si ;

   // compute x and y
   if ( scaled )
   {
      x = xe + ax*xp ;
      y = ay*yp ;
   }
   else
   {
      x = xe + xp ;
      y = yp ;
   }
}

// --------------------------------------------------------------------------
// evaluates the map, according to 'using_radial'

template< class T >
void PSCMaps<T>::eval_map( T s, T t, T &x, T &y ) const
{
   if ( do_checks )
      assert( initialized );

   if ( using_radial )
      rad_map( s,t,x,y );
   else
      hor_map( s,t,x,y );
}

// ****************************************************************************

inline std::string b2s( const bool b  )
{
   return b ? "true" : "false" ;
}
// ---------------------------------------------------------------------------
template< class T > void PSCMaps<T>::debug( )
{
   const std::string
      T_descr = std::is_same<T,float>::value ? "float" :
                  std::is_same<T,double>::value ? "double" :
                     "other" ;

   cout << "Spherical cap specific data" << endl
        << "     using radial      == " << b2s(using_radial ) << endl
        << "     fully visible     == " << b2s(fully_visible) << endl
        << "     partially visible == " << b2s(partially_visible) << endl
        << "     center_below_hor  == " << b2s(center_below_hor) << endl
        << "     invisible         == " << b2s(invisible) << endl ;

   if ( invisible )
      return ;

   cout << "     cos_beta          == " << cos_beta << endl
        << "     sin_beta          == " << sin_beta << endl
        << "     ax                == " << ax << endl
        << "     ay                == " << ay << endl ;
   cout << "     E                 == " << E << endl
        << "     L                 == " << L << endl
        << "     F                 == " << F << endl ;

   if ( partially_visible )
   {
      cout << "     xl                == " << xl << endl
           << "     yl                == " << yl << endl ;
      if ( using_radial  )
      {
         cout << "     phi_l             == " << phi_l << endl
              << "     AE(phi_l)         == " << AE_phi_l << endl ;
      }
   }
}

// -----------------------------------------------------------------------------
// function InverseNSB
//
// evaluate the inverse of a function whose derivative is > 0
//  Uses Newton method for root finding when possible,
//  and a combination of Binary search  and Secant methods
//  (at each step the best option is selected)
//
// returns the value 't' (in the range [0,t_max]) such that F(t) = Aobj
// it is assumed that F(t) in [0,A_max] for t in [0,t_max]
//                    0.0 <= Aobj <= A_max
//                    F(0.0) == 0.0, and F(t_max) == A_max
//                    F' = f
// where
//    F    : function whose inverse value we want to compute
//    f    : derivative of F  (>0 in [0,t_max])
//    t_max: maximum value for the result 't'
//    Aobj : desired value of F(t)
//    A_max: maximum value for A (minimum is 0.0)
//


template< class T >
T InverseNSB( FuncType<T> F, FuncType<T> f,
              const T t_max, const T Aobj, const T A_max )
{
   using namespace std ;

   if ( do_checks )
   if ( Vars<T>::trace_newton_inversion )
   {
      // set precision (see: http://en.cppreference.com/w/cpp/io/manip/setprecision)
      cout << fixed << showpoint << showpos << setprecision( 8 );
      cout << endl << "begins InverseNSB, t_max = " << t_max << ", Aobj == " << Aobj << ", A_max == " << A_max << endl << endl ;
   }

   if ( do_checks )
   {
      assert( -epsilon <= Aobj );
      assert( Aobj <= A_max+epsilon );

      // sometimes the initial guess may be out of range, we don't check that
      //assert( ymin-tolerance <= y0 && y0 <= ymax+tolerance );
   }

   const T
      A      = std::max( T(0.0), std::min( Aobj, A_max ) ); // A is always positive
   T
      tn     = (A/A_max)*t_max, // current best estimation of result value 't'
      tn_min = T(0.0) ,         // current interval: minimum value
      tn_max = t_max ,          // current interval: maximum value
      diff_tn_min = T(0.0)-A ,  // == F(tn_min) - A, (always negative) current difference at the left extreme of the interval
      diff_tn_max = A_max-A ;   // == F(tn_max) - A, (always positive) current difference at the right extreme of the interval


   int num_iters = 0 ;  // number of iterations so far


   T diff ;

   while( true )
   {
      if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
      {
         if ( num_iters < 10 ) cout << " " ;
         cout << "  (" << num_iters  << "): tn_min " << tn_min << ", tn " << tn << ", tn_max " << tn_max  ;
      }
      const T Ftn  = F(tn) ;

      if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
         cout << ",  F(tn) " << F(tn)  ;

      diff = Ftn - A ;

      if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
         cout <<", diff " << diff  ;

      // exit when done
      if ( std::abs( diff ) <= Vars<T>::iN_tolerance )
      {
         if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
            cout << endl << endl << "end iterations, diff <= tolerance == " << Vars<T>::iN_tolerance << endl ;
         break ;
      }

      // compute derivative, we must trunc it so it there are no out-of-range instabilities
      const T ftn = f(tn) ;   // we know f(yn) is never negative

      if ( do_checks )
      if ( Vars<T>::trace_newton_inversion )
         cout << ", f(tn) " << ftn  ;

      const T delta = -diff/ftn ;

      if ( do_checks )
      if ( Vars<T>::trace_newton_inversion )
         cout << ", -diff/ftn " << delta ;

      T tn_next = tn + delta ;

      if ( std::isnan(tn_next) || (tn_next < tn_min || tn_max < tn_next) )
      {
         // tn_next out of range

         // update interval
         if ( 0.0 < diff )
         {
            // move to the left (current F(yn) is higher than desired)
            tn_max      = tn ;
            diff_tn_max = diff ;
         }
         else
         {
            // move to the right (current F(yn) is smaller than desired )
            tn_min      = tn ;
            diff_tn_min = diff ;
         }

         // update 'tn' according to the secant rule
         // 'tn' is in the range [tn_min,tn_max]
         // tn_next = ( tn_min*diff_tn_max - tn_max*diff_tn_min )/( diff_tn_max - diff_tn_min );

         // update tn by using current inteval midpoint
         tn_next = T(0.5)*tn_max + T(0.5)*tn_min;

         if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
            cout << ", (interval) delta = " << (tn_next-tn) ;
      }
      else
      {
         if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
            cout << ", (newton) delta = " << delta  ;

      }

      tn = tn_next ;
      num_iters++ ;

      if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
         cout << endl ;

      // exit when the max number of iterations is exceeded
      if ( Vars<T>::iN_max_iters < num_iters   )
      {
         if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
            cout << endl << "end iterations: num_iters exceeded max ==  " << Vars<T>::iN_max_iters << endl ;
         break ;
      }
   }

   if ( do_checks ) if ( Vars<T>::trace_newton_inversion )
   {
      cout << "ends InverseNSB: tn == " << tn << ", diff == " << diff << endl << endl ;
      cout << std::defaultfloat ;
   }
   // done
   return tn ;
}
// ---------------------------------------------------------------------
// numerically integrate a real function on a real interval (x0,x1),
// by using 'n' equally spaced samples

template< class T >
T numeric_integral( const std::function<T(T)> & f, T x0, T x1, int n )
{
   assert( 2 < n );

   T       sum = T(0.0) ;
   const T dx  = (x1-x0)/T(n) ;

   for( int i = 0 ; i < n ; i++ )
   {
      const T xi = x0 + (T(i)+T(0.5))*dx ;
      sum += f( xi );
   }
   return sum*dx ;
}
// -----------------------------------------------------------------
// test the integrals

template< class T >
void PSCMaps<T>::run_test_integrals(  )
{
   using namespace std ;


   assert( initialized );
   if ( invisible )
   {
      cout << "(cannot run test: the sphere is fully invisible)" << endl ;
      return ;
   }

   if ( partially_visible )
   {

      cout << "Partially visible integrals tests " << phi_l << endl ;

      if ( ! using_radial )
      {
         // compute L/2 by using numerical integration of horizontal segments
         auto integrand_h = [&]( T y )
         {
            return eval_xCir(y) - eval_xEll(y) ;
         };
         const T L_par_num = numeric_integral<T>( integrand_h, 0.0, yl, 1024 );

         // get (half) lune area which is already computed in the sampler
         const T L_par_ana = L ;

         // compare results for L/2
         cout << " lune area == " << endl
              << "   parallel, numerical  == " << L_par_num << endl
              << "   parallel, analytical == " << L_par_ana << endl ;
      }
      else
      {
         // compute L/2 by using numerical integration of radiuses
         auto integrand_r = [&]( T theta )
         {
            const T re = eval_rEll(theta),
                    rc = eval_rCirc(theta) ;
            return (rc*rc- re*re)/T(2.0) ;
         };

         const T L2_rad_num = numeric_integral<T>( integrand_r, 0.0, phi_l, 1024 );

         // get (half) lune area which is already computed in the sampler
         const T L2_rad_ana = L ;

         // compare results for L/2
         cout << " lune area == " << endl
              << "   radial, numerical    == " << L2_rad_num << endl
              << "   radial, analytical   == " << L2_rad_ana << endl ;
      }

   }
   else
   {
      cout << "(this case's test have to be implemented)" << endl ;
   }
}


// *****************************************************************************
// debug functions in template class Vars<T>
// -----------------------------------------------------------------------------

template<class T>
void Vars<T>::print_settings()
{
      const std::string
         T_descr = std::is_same<T,float>::value ? "float" :
                     std::is_same<T,double>::value ? "double" :
                        "other" ;
   using namespace std ;
   cout << "" << endl
        << "Global settings " << endl
        << "     T             == " << T_descr << endl
        << "     do_checks     == " << (do_checks ? "true" : "false" ) << endl
        << "     tolerance     == " << iN_tolerance << endl
        << "     max. iters.   == " << iN_max_iters << endl ;
}
// *****************************************************************************

} // ends namespace PSCM

#endif // ends #ifndef PSCMAPS_H
