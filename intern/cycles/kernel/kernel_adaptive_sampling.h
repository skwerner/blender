/*
 * Copyright 2019 Blender Foundation
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

#ifndef __KERNEL_ADAPTIVE_SAMPLING_H__
#define __KERNEL_ADAPTIVE_SAMPLING_H__

CCL_NAMESPACE_BEGIN

ccl_device bool kernel_adaptive_filter_x(KernelGlobals *kg, ccl_global float* tile_buffer, int y,
	int tile_x, int tile_y, int tile_w, int tile_h, int tile_offset, int tile_stride)
{
	bool any = false;
	bool prev = false;
	for (int x = tile_x; x < tile_x + tile_w; ++x) {
		int index = tile_offset + x + y * tile_stride;
		ccl_global float *buffer = (ccl_global float*)tile_buffer + index * kernel_data.film.pass_stride;
		ccl_global float4 *minmax = (ccl_global float4*)(buffer + kernel_data.film.pass_adaptive_min_max);
		if (minmax->w == 0.0f) {
			prev = true;
			any = true;
			if (x > tile_x) {
				index = index - 1;
				buffer = (ccl_global float*)tile_buffer + index * kernel_data.film.pass_stride;
				minmax = (ccl_global float4*)(buffer + kernel_data.film.pass_adaptive_min_max);
				minmax->w = 0.0f;
			}
		}
		else {
			if (prev) {
				minmax->w = 0.0f;
			}
			prev = false;
		}
	}
	return any;
}

ccl_device bool kernel_adaptive_filter_y(KernelGlobals *kg, ccl_global float* tile_buffer, int x,
	int tile_x, int tile_y, int tile_w, int tile_h, int tile_offset, int tile_stride)
{
	bool prev = false;
	bool any = false;
	for (int y = tile_y; y < tile_y + tile_h; ++y) {
		int index = tile_offset + x + y * tile_stride;
		ccl_global float *buffer = (ccl_global float*)tile_buffer + index * kernel_data.film.pass_stride;
		ccl_global float4 *minmax = (ccl_global float4*)(buffer + kernel_data.film.pass_adaptive_min_max);
		if (minmax->w == 0.0f) {
			prev = true;
			if (y > tile_y) {
				index = index - tile_stride;
				buffer = (ccl_global float*)tile_buffer + index * kernel_data.film.pass_stride;
				minmax = (ccl_global float4*)(buffer + kernel_data.film.pass_adaptive_min_max);
				minmax->w = 0.0f;
			}
		}
		else {
			if (prev) {
				minmax->w = 0.0f;
			}
			prev = false;
		}
	}
	return any;
}

CCL_NAMESPACE_END

#endif  /* __KERNEL_ADAPTIVE_SAMPLING_H__ */
