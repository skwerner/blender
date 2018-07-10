#ifndef __DEVICE_MEMORY_OPENVDB_H__
#define __DEVICE_MEMORY_OPENVDB_H__

/* OpenVDB Memory
 *
 *
 */
#include <openvdb/openvdb.h>
#include "device/device_memory.h"

CCL_NAMESPACE_BEGIN

template<typename GridType, typename DataType>
class device_openvdb : public device_memory
{
public:
	device_openvdb(Device *device,
	               const char *name,
	               MemoryType type,
	               typename GridType::Ptr grid,
	               typename GridType::ConstAccessor accessor,
	               int3 resolution)
	: device_memory(device, name, type),
	  vdb_grid(grid),
	  vdb_acc(accessor)
	{
		using namespace openvdb;

		data_type = device_type_traits<DataType>::data_type;
		data_elements = device_type_traits<DataType>::num_elements;

		assert(data_elements > 0);

		host_pointer = static_cast<void*>(&vdb_acc);

		data_width = real_width = resolution.x;
		data_height = real_height = resolution.y;
		data_depth = real_depth = resolution.z;

		assert((vdb_grid->memUsage() % (data_elements * datatype_size(data_type))) == 0);

		data_size = vdb_grid->memUsage() / (data_elements * datatype_size(data_type));
	}

	typename GridType::Ptr vdb_grid;
	typename GridType::ConstAccessor vdb_acc;

	void copy_to_device()
	{
		device_copy_to();
	}

protected:
	size_t size(size_t width, size_t height, size_t depth)
	{
		return width * ((height == 0)? 1: height) * ((depth == 0)? 1: depth);
	}
};

CCL_NAMESPACE_END

#endif /* __DEVICE_MEMORY_OPENVDB_H__ */
