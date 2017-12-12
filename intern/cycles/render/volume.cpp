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

#include "scene.h"
#include "volume.h"

#include "util/util_foreach.h"
#include "util/util_logging.h"
#include "util/util_progress.h"
#include "util/util_task.h"

#include "kernel/vdb/vdb_globals.h"

CCL_NAMESPACE_BEGIN

/* ------------------------------------------------------------------------- */

VolumeManager::VolumeManager()
{
#ifdef WITH_OPENVDB
	openvdb::initialize();
#endif
	need_update = true;
}

VolumeManager::~VolumeManager()
{
	for(size_t i = 0; i < grids.size(); ++i) {
		if(grids[i]) {
			delete grids[i];
		}
	}
}

static inline void catch_exceptions()
{
#ifdef WITH_OPENVDB
	try {
		throw;
	}
	catch(const openvdb::IoError& e) {
		std::cerr << e.what() << "\n";
	}
#endif
}

int VolumeManager::add_volume(const std::string &filename, const std::string &name, const Transform& tfm)
{
	int slot = -1;

	if((slot = find_existing_slot(filename, name, tfm)) != -1) {
		grids[slot]->users += 1;
	}

	try {
		if(is_openvdb_file(filename)) {
			slot = add_openvdb_volume(filename, name, tfm);
		}
	}
	catch(...) {
		catch_exceptions();
		slot = -1;
	}

	return slot;
}

int VolumeManager::find_existing_slot(const std::string &filename, const std::string &gridname, const Transform &tfm)
{
	for(size_t i = 0; i < grids.size(); ++i) {
		if(grids[i]) {
			if(grids[i]->filename == filename
			   && grids[i]->gridname == gridname
			   && grids[i]->tfm == tfm) {
				return i;
			}
		}
	}

	return -1;
}


bool VolumeManager::is_openvdb_file(const string& filename) const
{
	return string_endswith(filename, ".vdb");
}

int VolumeManager::add_openvdb_volume(const std::string &filename, const std::string &gridname, const Transform &tfm)
{
	int slot = -1;

#ifdef WITH_OPENVDB
	bool file_ok = true;
	openvdb::io::File file(filename);
	try {
		file.open();
		if(!file.hasGrid(gridname)) {
			file_ok = false;
		}
		else {
			openvdb::GridBase::Ptr grid = file.readGridMetadata(gridname);
			if(grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
				file_ok = false;
			}
		}
	}
	catch(...) {
		catch_exceptions();
		file_ok = false;
	}

	if(!file_ok) {
		return -1;
	}

	/* Find a free slot. */
	for(slot = 0; slot < grids.size(); slot++) {
		if(!grids[slot])
			break;
	}

	if(slot == grids.size()) {
		grids.resize(grids.size() + 1);
	}

	GridDescription *grid = new GridDescription();
	grid->filename = filename;
	grid->gridname = gridname;
	grid->tfm = tfm;
	grid->users = 1;

	grids[slot] = grid;
#else
	(void)volume;
	(void)filename;
	(void)name;
#endif

	return slot;
}

void VolumeManager::remove_volume(int slot)
{
	if(slot < grids.size() && grids[slot]) {
		delete grids[slot];
		grids[slot] = NULL;
	}
	need_update = true;
}

void VolumeManager::device_update(Device *device, DeviceScene *dscene, Scene *scene, Progress& progress)
{
	if(!need_update) {
		return;
	}

	device_free(device, dscene);
	progress.set_status("Updating OpenVDB volumes", "Sending volumes to device.");

#ifdef WITH_OPENVDB
	OpenVDBGlobals *vdb = (OpenVDBGlobals*)device->vdb_memory();
	if(!vdb) {
		assert(0);
		return;
	}

	for(int i = 0; i < vdb->grids.size(); ++i) {
		if(vdb->grids[i]) {
			delete vdb->grids[i];
		}
	}

	vdb->grids.resize(grids.size());

	for(int i = 0; i < grids.size(); ++i) {
		if(grids[i] == NULL) {
			vdb->grids[i] = NULL;
			continue;
		}

		progress.set_status("Updating OpenVDB volumes", "Loading " + grids[i]->gridname + " from " + grids[i]->filename);
		openvdb::io::File file(grids[i]->filename);
		try {
			file.open();
			openvdb::GridBase::Ptr base_grid = file.readGrid(grids[i]->gridname);
			if(base_grid) {
				vdb->grids[i] = OpenVDBTextureBase::create_from_grid(base_grid, grids[i]->tfm);
				VLOG(1) << base_grid->getName().c_str() << " memory usage: " << base_grid->memUsage() / 1024.0f << " kilobytes.\n";
			}
		}
		catch(...) {
			catch_exceptions();
			vdb->grids[i] = NULL;
		}
	}
#endif

	if(progress.get_cancel()) {
		return;
	}

	need_update = false;
}

void VolumeManager::device_free(Device *device, DeviceScene *dscene)
{
#ifdef WITH_OPENVDB
	OpenVDBGlobals *vdb = (OpenVDBGlobals*)device->vdb_memory();
	for (size_t i = 0; i < vdb->grids.size(); ++i) {
		if(vdb->grids[i]) {
			delete vdb->grids[i];
			vdb->grids[i] = NULL;
		}
	}
	vdb->grids.clear();
#endif
}

void VolumeManager::tag_update(Scene * /*scene*/)
{
	need_update = true;
}

CCL_NAMESPACE_END
