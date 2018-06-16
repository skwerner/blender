/*
 * Copyright 2011-2016 Blender Foundation
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

#ifndef __KERNEL_CPU_IMAGE_H__
#define __KERNEL_CPU_IMAGE_H__

#include "util/util_sparse_grid.h"

CCL_NAMESPACE_BEGIN

template<typename T> struct TextureInterpolator  {
#define SET_CUBIC_SPLINE_WEIGHTS(u, t) \
	{ \
		u[0] = (((-1.0f/6.0f)* t + 0.5f) * t - 0.5f) * t + (1.0f/6.0f); \
		u[1] =  ((      0.5f * t - 1.0f) * t       ) * t + (2.0f/3.0f); \
		u[2] =  ((     -0.5f * t + 0.5f) * t + 0.5f) * t + (1.0f/6.0f); \
		u[3] = (1.0f / 6.0f) * t * t * t; \
	} (void)0

	static ccl_always_inline int flatten(int x, int y, int z, int width, int height)
	{
		return x + width * (y + z * height);
	}

	static ccl_always_inline float4 read(float4 r)
	{
		return r;
	}

	static ccl_always_inline float4 read(uchar4 r)
	{
		float f = 1.0f/255.0f;
		return make_float4(r.x*f, r.y*f, r.z*f, r.w*f);
	}

	static ccl_always_inline float4 read(uchar r)
	{
		float f = r*(1.0f/255.0f);
		return make_float4(f, f, f, 1.0f);
	}

	static ccl_always_inline float4 read(float r)
	{
		/* TODO(dingto): Optimize this, so interpolation
		 * happens on float instead of float4 */
		return make_float4(r, r, r, 1.0f);
	}

	static ccl_always_inline float4 read(half4 r)
	{
		return half4_to_float4(r);
	}

	static ccl_always_inline float4 read(half r)
	{
		float f = half_to_float(r);
		return make_float4(f, f, f, 1.0f);
	}

	static ccl_always_inline float4 read(const T *data,
	                                     int x, int y,
	                                     int width, int height)
	{
		if(x < 0 || y < 0 || x >= width || y >= height) {
			return make_float4(0.0f);
		}
		return read(data[y * width + x]);
	}

	static ccl_always_inline float4 read(const T *data, const int *grid_info,
	                                     int x, int y, int z,
	                                     int bit_count, int ltw, int lth)
	{
		int tix = x / TILE_SIZE, itix = x % TILE_SIZE,
		    tiy = y / TILE_SIZE, itiy = y % TILE_SIZE,
		    tiz = z / TILE_SIZE, itiz = z % TILE_SIZE;
		int dense_index = compute_morton(tix, tiy, tiz, bit_count) * 2;
		int sparse_index = grid_info[dense_index];
		int dims = grid_info[dense_index + 1];
		if(sparse_index < 0) {
			return make_float4(0.0f);
		}
		int itiw = dims & (1 << ST_SHIFT_TRUNCATE_WIDTH) ? ltw : TILE_SIZE;
		int itih = dims & (1 << ST_SHIFT_TRUNCATE_HEIGHT) ? lth : TILE_SIZE;
		int in_tile_index = flatten(itix, itiy, itiz, itiw, itih);
		return read(data[sparse_index + in_tile_index]);
	}

	static ccl_always_inline float4 read(const T *data, const int *grid_info,
	                                     int index, int width, int height, int /*depth*/,
	                                     int bit_count, int ltw, int lth)
	{
		int x = index % width;
		int y = (index / width) % height;
		int z = index / (width * height);
		return read(data, grid_info, x, y, z, bit_count, ltw, lth);
	}

	static ccl_always_inline int wrap_periodic(int x, int width)
	{
		x %= width;
		if(x < 0)
			x += width;
		return x;
	}

	static ccl_always_inline int wrap_clamp(int x, int width)
	{
		return clamp(x, 0, width-1);
	}

	static ccl_always_inline float frac(float x, int *ix)
	{
		int i = float_to_int(x) - ((x < 0.0f)? 1: 0);
		*ix = i;
		return x - (float)i;
	}

	/* ********  2D interpolation ******** */

	static ccl_always_inline float4 interp_closest(const TextureInfo& info,
	                                               float x, float y)
	{
		const T *data = (const T*)info.data;
		const int width = info.width;
		const int height = info.height;
		int ix, iy;
		frac(x*(float)width, &ix);
		frac(y*(float)height, &iy);
		switch(info.extension) {
			case EXTENSION_REPEAT:
				ix = wrap_periodic(ix, width);
				iy = wrap_periodic(iy, height);
				break;
			case EXTENSION_CLIP:
				if(x < 0.0f || y < 0.0f || x > 1.0f || y > 1.0f) {
					return make_float4(0.0f);
				}
				ATTR_FALLTHROUGH;
			case EXTENSION_EXTEND:
				ix = wrap_clamp(ix, width);
				iy = wrap_clamp(iy, height);
				break;
			default:
				kernel_assert(0);
				return make_float4(0.0f);
		}
		return read(data[ix + iy*width]);
	}

	static ccl_always_inline float4 interp_linear(const TextureInfo& info,
	                                              float x, float y)
	{
		const T *data = (const T*)info.data;
		const int width = info.width;
		const int height = info.height;
		int ix, iy, nix, niy;
		const float tx = frac(x*(float)width - 0.5f, &ix);
		const float ty = frac(y*(float)height - 0.5f, &iy);
		switch(info.extension) {
			case EXTENSION_REPEAT:
				ix = wrap_periodic(ix, width);
				iy = wrap_periodic(iy, height);
				nix = wrap_periodic(ix+1, width);
				niy = wrap_periodic(iy+1, height);
				break;
			case EXTENSION_CLIP:
				nix = ix + 1;
				niy = iy + 1;
				break;
			case EXTENSION_EXTEND:
				nix = wrap_clamp(ix+1, width);
				niy = wrap_clamp(iy+1, height);
				ix = wrap_clamp(ix, width);
				iy = wrap_clamp(iy, height);
				break;
			default:
				kernel_assert(0);
				return make_float4(0.0f);
		}
		return (1.0f - ty) * (1.0f - tx) * read(data, ix, iy, width, height) +
		       (1.0f - ty) * tx * read(data, nix, iy, width, height) +
		       ty * (1.0f - tx) * read(data, ix, niy, width, height) +
		       ty * tx * read(data, nix, niy, width, height);
	}

	static ccl_always_inline float4 interp_cubic(const TextureInfo& info,
	                                             float x, float y)
	{
		const T *data = (const T*)info.data;
		const int width = info.width;
		const int height = info.height;
		int ix, iy, nix, niy;
		const float tx = frac(x*(float)width - 0.5f, &ix);
		const float ty = frac(y*(float)height - 0.5f, &iy);
		int pix, piy, nnix, nniy;
		switch(info.extension) {
			case EXTENSION_REPEAT:
				ix = wrap_periodic(ix, width);
				iy = wrap_periodic(iy, height);
				pix = wrap_periodic(ix-1, width);
				piy = wrap_periodic(iy-1, height);
				nix = wrap_periodic(ix+1, width);
				niy = wrap_periodic(iy+1, height);
				nnix = wrap_periodic(ix+2, width);
				nniy = wrap_periodic(iy+2, height);
				break;
			case EXTENSION_CLIP:
				pix = ix - 1;
				piy = iy - 1;
				nix = ix + 1;
				niy = iy + 1;
				nnix = ix + 2;
				nniy = iy + 2;
				break;
			case EXTENSION_EXTEND:
				pix = wrap_clamp(ix-1, width);
				piy = wrap_clamp(iy-1, height);
				nix = wrap_clamp(ix+1, width);
				niy = wrap_clamp(iy+1, height);
				nnix = wrap_clamp(ix+2, width);
				nniy = wrap_clamp(iy+2, height);
				ix = wrap_clamp(ix, width);
				iy = wrap_clamp(iy, height);
				break;
			default:
				kernel_assert(0);
				return make_float4(0.0f);
		}
		const int xc[4] = {pix, ix, nix, nnix};
		const int yc[4] = {piy, iy, niy, nniy};
		float u[4], v[4];
		/* Some helper macro to keep code reasonable size,
		 * let compiler to inline all the matrix multiplications.
		 */
#define DATA(x, y) (read(data, xc[x], yc[y], width, height))
#define TERM(col) \
		(v[col] * (u[0] * DATA(0, col) + \
		           u[1] * DATA(1, col) + \
		           u[2] * DATA(2, col) + \
		           u[3] * DATA(3, col)))

		SET_CUBIC_SPLINE_WEIGHTS(u, tx);
		SET_CUBIC_SPLINE_WEIGHTS(v, ty);

		/* Actual interpolation. */
		return TERM(0) + TERM(1) + TERM(2) + TERM(3);
#undef TERM
#undef DATA
	}

	static ccl_always_inline float4 interp(const TextureInfo& info,
	                                       float x, float y)
	{
		if(UNLIKELY(!info.data)) {
			return make_float4(0.0f);
		}
		switch(info.interpolation) {
			case INTERPOLATION_CLOSEST:
				return interp_closest(info, x, y);
			case INTERPOLATION_LINEAR:
				return interp_linear(info, x, y);
			default:
				return interp_cubic(info, x, y);
		}
	}

	/* ********  3D interpolation ******** */

	static ccl_always_inline float4 interp_3d_closest(const TextureInfo& info,
	                                                  float x, float y, float z)
	{
		int width = info.width;
		int height = info.height;
		int depth = info.depth;
		int ix, iy, iz;

		frac(x*(float)width, &ix);
		frac(y*(float)height, &iy);
		frac(z*(float)depth, &iz);

		switch(info.extension) {
			case EXTENSION_REPEAT:
				ix = wrap_periodic(ix, width);
				iy = wrap_periodic(iy, height);
				iz = wrap_periodic(iz, depth);
				break;
			case EXTENSION_CLIP:
				if(x < 0.0f || y < 0.0f || z < 0.0f ||
				   x > 1.0f || y > 1.0f || z > 1.0f)
				{
					return make_float4(0.0f);
				}
				ATTR_FALLTHROUGH;
			case EXTENSION_EXTEND:
				ix = wrap_clamp(ix, width);
				iy = wrap_clamp(iy, height);
				iz = wrap_clamp(iz, depth);
				break;
			default:
				kernel_assert(0);
				return make_float4(0.0f);
		}

		const T *data = (const T*)info.data;
		const int *grid_info = (const int*)info.grid_info;

		if(grid_info) {
			return read(data, grid_info, ix, iy, iz, info.bit_count,
			            info.last_tile_width, info.last_tile_height);
		}
		return read(data[flatten(ix, iy, iz, width, height)]);
	}

	static ccl_always_inline float4 interp_3d_linear(const TextureInfo& info,
	                                                 float x, float y, float z)
	{
		int width = info.width;
		int height = info.height;
		int depth = info.depth;
		int ix, iy, iz;
		int nix, niy, niz;

		float tx = frac(x*(float)width - 0.5f, &ix);
		float ty = frac(y*(float)height - 0.5f, &iy);
		float tz = frac(z*(float)depth - 0.5f, &iz);

		switch(info.extension) {
			case EXTENSION_REPEAT:
				ix = wrap_periodic(ix, width);
				iy = wrap_periodic(iy, height);
				iz = wrap_periodic(iz, depth);

				nix = wrap_periodic(ix+1, width);
				niy = wrap_periodic(iy+1, height);
				niz = wrap_periodic(iz+1, depth);
				break;
			case EXTENSION_CLIP:
				if(x < 0.0f || y < 0.0f || z < 0.0f ||
				   x > 1.0f || y > 1.0f || z > 1.0f)
				{
					return make_float4(0.0f);
				}
				ATTR_FALLTHROUGH;
			case EXTENSION_EXTEND:
				nix = wrap_clamp(ix+1, width);
				niy = wrap_clamp(iy+1, height);
				niz = wrap_clamp(iz+1, depth);

				ix = wrap_clamp(ix, width);
				iy = wrap_clamp(iy, height);
				iz = wrap_clamp(iz, depth);
				break;
			default:
				kernel_assert(0);
				return make_float4(0.0f);
		}

		float4 r;
		const T *data = (const T*)info.data;
		const int *gi = (const int*)info.grid_info;

		if(gi) {
			int bc = info.bit_count;
			int ltw = info.last_tile_width;
			int lth = info.last_tile_height;
			r  = (1.0f - tz)*(1.0f - ty)*(1.0f - tx) * read(data, gi, ix,  iy,  iz,  bc, ltw, lth);
			r += (1.0f - tz)*(1.0f - ty)*tx          * read(data, gi, nix, iy,  iz,  bc, ltw, lth);
			r += (1.0f - tz)*ty*(1.0f - tx)          * read(data, gi, ix,  niy, iz,  bc, ltw, lth);
			r += (1.0f - tz)*ty*tx                   * read(data, gi, nix, niy, iz,  bc, ltw, lth);
			r += tz*(1.0f - ty)*(1.0f - tx)          * read(data, gi, ix,  iy,  niz, bc, ltw, lth);
			r += tz*(1.0f - ty)*tx                   * read(data, gi, nix, iy,  niz, bc, ltw, lth);
			r += tz*ty*(1.0f - tx)                   * read(data, gi, ix,  niy, niz, bc, ltw, lth);
			r += tz*ty*tx                            * read(data, gi, nix, niy, niz, bc, ltw, lth);
		}
		else {
			r  = (1.0f - tz)*(1.0f - ty)*(1.0f - tx) * read(data[flatten(ix,  iy,  iz,  width, height)]);
			r += (1.0f - tz)*(1.0f - ty)*tx			 * read(data[flatten(nix, iy,  iz,  width, height)]);
			r += (1.0f - tz)*ty*(1.0f - tx)			 * read(data[flatten(ix,  niy, iz,  width, height)]);
			r += (1.0f - tz)*ty*tx					 * read(data[flatten(nix, niy, iz,  width, height)]);
			r += tz*(1.0f - ty)*(1.0f - tx)			 * read(data[flatten(ix,  iy,  niz, width, height)]);
			r += tz*(1.0f - ty)*tx					 * read(data[flatten(nix, iy,  niz, width, height)]);
			r += tz*ty*(1.0f - tx)					 * read(data[flatten(ix,  niy, niz, width, height)]);
			r += tz*ty*tx							 * read(data[flatten(nix, niy, niz, width, height)]);
		}

		return r;
	}

	/* TODO(sergey): For some unspeakable reason both GCC-6 and Clang-3.9 are
	 * causing stack overflow issue in this function unless it is inlined.
	 *
	 * Only happens for AVX2 kernel and global __KERNEL_SSE__ vectorization
	 * enabled.
	 */
#if defined(__GNUC__) || defined(__clang__)
	static ccl_always_inline
#else
	static ccl_never_inline
#endif
	float4 interp_3d_tricubic(const TextureInfo& info, float x, float y, float z)
	{
		int width = info.width;
		int height = info.height;
		int depth = info.depth;
		int bc = info.bit_count;
		int ltw = info.last_tile_width;
		int lth = info.last_tile_height;
		int ix, iy, iz;
		int nix, niy, niz;
		/* Tricubic b-spline interpolation. */
		const float tx = frac(x*(float)width - 0.5f, &ix);
		const float ty = frac(y*(float)height - 0.5f, &iy);
		const float tz = frac(z*(float)depth - 0.5f, &iz);
		int pix, piy, piz, nnix, nniy, nniz;

		switch(info.extension) {
			case EXTENSION_REPEAT:
				ix = wrap_periodic(ix, width);
				iy = wrap_periodic(iy, height);
				iz = wrap_periodic(iz, depth);

				pix = wrap_periodic(ix-1, width);
				piy = wrap_periodic(iy-1, height);
				piz = wrap_periodic(iz-1, depth);

				nix = wrap_periodic(ix+1, width);
				niy = wrap_periodic(iy+1, height);
				niz = wrap_periodic(iz+1, depth);

				nnix = wrap_periodic(ix+2, width);
				nniy = wrap_periodic(iy+2, height);
				nniz = wrap_periodic(iz+2, depth);
				break;
			case EXTENSION_CLIP:
				if(x < 0.0f || y < 0.0f || z < 0.0f ||
				   x > 1.0f || y > 1.0f || z > 1.0f)
				{
					return make_float4(0.0f);
				}
				ATTR_FALLTHROUGH;
			case EXTENSION_EXTEND:
				pix = wrap_clamp(ix-1, width);
				piy = wrap_clamp(iy-1, height);
				piz = wrap_clamp(iz-1, depth);

				nix = wrap_clamp(ix+1, width);
				niy = wrap_clamp(iy+1, height);
				niz = wrap_clamp(iz+1, depth);

				nnix = wrap_clamp(ix+2, width);
				nniy = wrap_clamp(iy+2, height);
				nniz = wrap_clamp(iz+2, depth);

				ix = wrap_clamp(ix, width);
				iy = wrap_clamp(iy, height);
				iz = wrap_clamp(iz, depth);
				break;
			default:
				kernel_assert(0);
				return make_float4(0.0f);
		}

		const int xc[4] = {pix, ix, nix, nnix};
		const int yc[4] = {width * piy,
		                   width * iy,
		                   width * niy,
		                   width * nniy};
		const int zc[4] = {width * height * piz,
		                   width * height * iz,
		                   width * height * niz,
		                   width * height * nniz};
		float u[4], v[4], w[4];

		/* Some helper macro to keep code reasonable size,
		 * let compiler to inline all the matrix multiplications.
		 */
#define DATA(x, y, z) (gi ? \
		read(data, gi, xc[x] + yc[y] + zc[z], width, height, depth, bc, ltw, lth) : \
		read(data[xc[x] + yc[y] + zc[z]]))
#define COL_TERM(col, row) \
		(v[col] * (u[0] * DATA(0, col, row) + \
		           u[1] * DATA(1, col, row) + \
		           u[2] * DATA(2, col, row) + \
		           u[3] * DATA(3, col, row)))
#define ROW_TERM(row) \
		(w[row] * (COL_TERM(0, row) + \
		           COL_TERM(1, row) + \
		           COL_TERM(2, row) + \
		           COL_TERM(3, row)))

		SET_CUBIC_SPLINE_WEIGHTS(u, tx);
		SET_CUBIC_SPLINE_WEIGHTS(v, ty);
		SET_CUBIC_SPLINE_WEIGHTS(w, tz);

		/* Actual interpolation. */
		const T *data = (const T*)info.data;
		const int *gi = (const int*)info.grid_info;
		return ROW_TERM(0) + ROW_TERM(1) + ROW_TERM(2) + ROW_TERM(3);

#undef COL_TERM
#undef ROW_TERM
#undef DATA
	}

	static ccl_always_inline float4 interp_3d(const TextureInfo& info,
	                                          float x, float y, float z,
	                                          InterpolationType interp)
	{
		if(UNLIKELY(!info.data))
			return make_float4(0.0f);

		switch((interp == INTERPOLATION_NONE)? info.interpolation: interp) {
			case INTERPOLATION_CLOSEST:
				return interp_3d_closest(info, x, y, z);
			case INTERPOLATION_LINEAR:
				return interp_3d_linear(info, x, y, z);
			default:
				return interp_3d_tricubic(info, x, y, z);
		}
	}
#undef SET_CUBIC_SPLINE_WEIGHTS
};

ccl_device float4 kernel_tex_image_interp(KernelGlobals *kg, int id, float x, float y)
{
	const TextureInfo& info = kernel_tex_fetch(__texture_info, id);

	switch(kernel_tex_type(id)) {
		case IMAGE_DATA_TYPE_HALF:
			return TextureInterpolator<half>::interp(info, x, y);
		case IMAGE_DATA_TYPE_BYTE:
			return TextureInterpolator<uchar>::interp(info, x, y);
		case IMAGE_DATA_TYPE_FLOAT:
			return TextureInterpolator<float>::interp(info, x, y);
		case IMAGE_DATA_TYPE_HALF4:
			return TextureInterpolator<half4>::interp(info, x, y);
		case IMAGE_DATA_TYPE_BYTE4:
			return TextureInterpolator<uchar4>::interp(info, x, y);
		case IMAGE_DATA_TYPE_FLOAT4:
		default:
			return TextureInterpolator<float4>::interp(info, x, y);
	}
}

ccl_device float4 kernel_tex_image_interp_3d(KernelGlobals *kg, int id, float x, float y, float z, InterpolationType interp)
{
	const TextureInfo& info = kernel_tex_fetch(__texture_info, id);

	switch(kernel_tex_type(id)) {
		case IMAGE_DATA_TYPE_HALF:
			return TextureInterpolator<half>::interp_3d(info, x, y, z, interp);
		case IMAGE_DATA_TYPE_BYTE:
			return TextureInterpolator<uchar>::interp_3d(info, x, y, z, interp);
		case IMAGE_DATA_TYPE_FLOAT:
			return TextureInterpolator<float>::interp_3d(info, x, y, z, interp);
		case IMAGE_DATA_TYPE_HALF4:
			return TextureInterpolator<half4>::interp_3d(info, x, y, z, interp);
		case IMAGE_DATA_TYPE_BYTE4:
			return TextureInterpolator<uchar4>::interp_3d(info, x, y, z, interp);
		case IMAGE_DATA_TYPE_FLOAT4:
		default:
			return TextureInterpolator<float4>::interp_3d(info, x, y, z, interp);
	}
}

CCL_NAMESPACE_END

#endif // __KERNEL_CPU_IMAGE_H__
