# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ***** END GPL LICENSE BLOCK *****

# Note the utility apps may use png/tiff/gif system libraries, but the
# library itself does not depend on them, so should give no problems.
if (APPLE)
set(EMBREE_AARCH64_EXTRA_ARGS
  -DEMBREE_ISPC_SUPPORT=OFF
  -DEMBREE_TUTORIALS=OFF
  -DEMBREE_STATIC_LIB=ON
  -DEMBREE_RAY_MASK=ON
  -DEMBREE_FILTER_FUNCTION=ON
  -DEMBREE_BACKFACE_CULLING=OFF
  -DEMBREE_TASKING_SYSTEM=GCD
  -DAS_MAC=ON
  -DEMBREE_ARM=ON
  -DEMBREE_BUILD_VERIFY=OFF
)


ExternalProject_Add(external_embree_aarch64
  GIT_REPOSITORY ${EMBREE_AARCH64_GIT_URI}
  GIT_TAG ${EMBREE_AARCH64_GIT_TAG}
  GIT_SHALLOW TRUE
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  PREFIX ${BUILD_DIR}/embree_aarch64
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/embree_aarch64 ${DEFAULT_CMAKE_FLAGS} ${EMBREE_AARCH64_EXTRA_ARGS}
  INSTALL_DIR ${LIBDIR}/embree_aarch64
)

add_dependencies(
  external_embree_aarch64
  external_tbb
)

endif()
