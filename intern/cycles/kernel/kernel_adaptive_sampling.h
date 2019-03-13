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

ccl_device void kernel_adaptive_post_adjust(KernelGlobals *kg, float *buffer, float sample_multiplier)
{
	*(float4*)(buffer) *= sample_multiplier;
#ifdef __PASSES__
	if(!kernel_data.film.use_light_pass) {
		int light_flag = kernel_data.film.light_pass_flag;

		if(light_flag & PASSMASK(DIFFUSE_INDIRECT))
			*(float3*)(buffer + kernel_data.film.pass_diffuse_indirect) *= sample_multiplier;
		if(light_flag & PASSMASK(GLOSSY_INDIRECT))
			*(float3*)(buffer + kernel_data.film.pass_glossy_indirect) *= sample_multiplier;
		if(light_flag & PASSMASK(TRANSMISSION_INDIRECT))
			*(float3*)(buffer + kernel_data.film.pass_transmission_indirect) *= sample_multiplier;
		if(light_flag & PASSMASK(SUBSURFACE_INDIRECT))
			*(float3*)(buffer + kernel_data.film.pass_subsurface_indirect) *= sample_multiplier;
		if(light_flag & PASSMASK(VOLUME_INDIRECT))
			*(float3*)(buffer + kernel_data.film.pass_volume_indirect) *= sample_multiplier;
		if(light_flag & PASSMASK(DIFFUSE_DIRECT))
			*(float3*)(buffer + kernel_data.film.pass_diffuse_direct) *= sample_multiplier;
		if(light_flag & PASSMASK(GLOSSY_DIRECT))
			*(float3*)(buffer + kernel_data.film.pass_glossy_direct) *= sample_multiplier;
		if(light_flag & PASSMASK(TRANSMISSION_DIRECT))
			*(float3*)(buffer + kernel_data.film.pass_transmission_direct) *= sample_multiplier;
		if(light_flag & PASSMASK(SUBSURFACE_DIRECT))
			*(float3*)(buffer + kernel_data.film.pass_subsurface_direct) *= sample_multiplier;
		if(light_flag & PASSMASK(VOLUME_DIRECT))
			*(float3*)(buffer + kernel_data.film.pass_volume_direct) *= sample_multiplier;

		if(light_flag & PASSMASK(EMISSION))
			*(float3*)(buffer + kernel_data.film.pass_emission) *= sample_multiplier;
		if(light_flag & PASSMASK(BACKGROUND))
			*(float3*)(buffer + kernel_data.film.pass_background) *= sample_multiplier;
		if(light_flag & PASSMASK(AO))
			*(float3*)(buffer + kernel_data.film.pass_ao) *= sample_multiplier;

		if(light_flag & PASSMASK(DIFFUSE_COLOR))
			*(float3*)(buffer + kernel_data.film.pass_diffuse_color) *= sample_multiplier;
		if(light_flag & PASSMASK(GLOSSY_COLOR))
			*(float3*)(buffer + kernel_data.film.pass_glossy_color) *= sample_multiplier;
		if(light_flag & PASSMASK(TRANSMISSION_COLOR))
			*(float3*)(buffer + kernel_data.film.pass_transmission_color) *= sample_multiplier;
		if(light_flag & PASSMASK(SUBSURFACE_COLOR))
			*(float3*)(buffer + kernel_data.film.pass_subsurface_color) *= sample_multiplier;

		if(light_flag & PASSMASK(SHADOW))
			*(float4*)(buffer + kernel_data.film.pass_shadow) *= sample_multiplier;
		if(light_flag & PASSMASK(MIST))
			*(float*)(buffer + kernel_data.film.pass_mist) *= sample_multiplier;
	}
#endif

#ifdef __DENOISING_FEATURES__

#define scale_float3_variance(buffer, offset, scale) \
	*(buffer + offset) *= scale; \
	*(buffer + offset + 1) *= scale; \
	*(buffer + offset + 2) *= scale; \
	*(buffer + offset + 3) *= scale; \
	*(buffer + offset + 4) *= scale; \
	*(buffer + offset + 5) *= scale;

	if(kernel_data.film.pass_denoising_data) {
		scale_float3_variance(buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_SHADOW_A, sample_multiplier);
		if(kernel_data.film.pass_denoising_clean) {
			scale_float3_variance(buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_COLOR, sample_multiplier);
			*(buffer + kernel_data.film.pass_denoising_clean) *= sample_multiplier;
			*(buffer + kernel_data.film.pass_denoising_clean + 1) *= sample_multiplier;
			*(buffer + kernel_data.film.pass_denoising_clean + 2) *= sample_multiplier;
		}
		else {
			scale_float3_variance(buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_COLOR, sample_multiplier);
		}
		scale_float3_variance(buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_NORMAL, sample_multiplier);
		scale_float3_variance(buffer, kernel_data.film.pass_denoising_data + DENOISING_PASS_ALBEDO, sample_multiplier);
		*(buffer + kernel_data.film.pass_denoising_data + DENOISING_PASS_DEPTH) *= sample_multiplier;
		*(buffer + kernel_data.film.pass_denoising_data + DENOISING_PASS_DEPTH + 1) *= sample_multiplier;
	}
#endif  /* __DENOISING_FEATURES__ */

	if(kernel_data.film.cryptomatte_passes) {
		kernel_assert(0); // TODO: Cryptomatte
	}
}

/* This is a simple box filter in two passes.
 * When a pixel demands more adaptive samples, let its neighboring pixels draw more samples too. */

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
