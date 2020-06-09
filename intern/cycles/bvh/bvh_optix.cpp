/*
 * Copyright 2019, NVIDIA Corporation.
 * Copyright 2019, Blender Foundation.
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

#ifdef WITH_OPTIX

#  include "bvh/bvh_optix.h"
#  include "render/geometry.h"
#  include "render/hair.h"
#  include "render/mesh.h"
#  include "render/object.h"
#  include "util/util_foreach.h"
#  include "util/util_logging.h"
#  include "util/util_progress.h"

CCL_NAMESPACE_BEGIN

BVHOptiX::BVHOptiX(const BVHParams &params_,
                   const vector<Geometry *> &geometry_,
                   const vector<Object *> &objects_)
    : BVHExternal(params_, geometry_, objects_)
{
}

BVHOptiX::~BVHOptiX()
{
}

void BVHOptiX::copy_to_device(Progress &progress, DeviceScene *dscene)
{
  progress.set_status("Updating Scene BVH", "Building OptiX acceleration structure");

  Device *const device = dscene->bvh_nodes.device;
  if (!device->build_optix_bvh(this))
    progress.set_error("Failed to build OptiX acceleration structure");
}

void BVHOptiX::pack_nodes(const BVHNode *)
{
}

void BVHOptiX::refit_nodes()
{
  // TODO(pmours): Implement?
  VLOG(1) << "Refit is not yet implemented for OptiX BVH.";
}

BVHNode *BVHOptiX::widen_children_nodes(const BVHNode *)
{
  return NULL;
}

CCL_NAMESPACE_END

#endif /* WITH_OPTIX */
