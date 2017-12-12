/*
 * Copyright 2017 Blender Foundation
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

#ifndef __VOLUMEMANAGER_H__
#define __VOLUMEMANAGER_H__

#include "attribute.h"

#include "util/util_string.h"
#include "util/util_types.h"

#ifdef WITH_OPENVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/RayIntersector.h>
#endif

CCL_NAMESPACE_BEGIN

class Device;
class DeviceScene;
class Progress;
class Scene;
class Shader;
class OpenVDBTextureBase;

class VolumeManager {
	struct GridDescription {
		string filename;
		string gridname;
		Transform tfm;
		int users;
	};

	vector<GridDescription*> grids;

	void add_grid_description(const string& filename, const string& gridname, int slot);
	int find_existing_slot(const string& filename, const string& gridname, const Transform &tfm);

	bool is_openvdb_file(const string& filename) const;
	int add_openvdb_volume(const string& filename, const string& gridname, const Transform &tfm);

public:
	VolumeManager();
	~VolumeManager();

	int add_volume(const string& filename, const string& grodname, const Transform &tfm);
	void remove_volume(int slot);

	void device_update(Device *device, DeviceScene *dscene, Scene *scene, Progress& progress);
	void device_update_attributes(Device *device, DeviceScene *dscene, Scene *scene, Progress& progress);
	void device_free(Device *device, DeviceScene *dscene);

	void tag_update(Scene *scene);

	bool need_update;
};

CCL_NAMESPACE_END

#endif /* __VOLUMEMANAGER_H__ */
