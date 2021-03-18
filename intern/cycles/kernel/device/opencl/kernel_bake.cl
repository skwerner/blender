#include "kernel/device/opencl/compat.h"
#include "kernel/device/opencl/globals.h"
#include "kernel/device/opencl/image.h"

#include "kernel/kernel_path.h"
#include "kernel/kernel_path_branched.h"

#include "kernel/kernel_bake.h"

__kernel void kernel_ocl_bake(
	ccl_constant KernelData *data,
	ccl_global float *buffer,

	KERNEL_BUFFER_PARAMS,

	int sx, int sy, int sw, int sh, int offset, int stride, int sample)
{
	KernelGlobals kglobals, *kg = &kglobals;

	kg->data = data;

	kernel_set_buffer_pointers(kg, KERNEL_BUFFER_ARGS);
	kernel_set_buffer_info(kg);

	int x = sx + ccl_global_id(0);
	int y = sy + ccl_global_id(1);

	if(x < sx + sw && y < sy + sh) {
#ifndef __NO_BAKING__
		kernel_bake_evaluate(kg, buffer, sample, x, y, offset, stride);
#endif
	}
}
