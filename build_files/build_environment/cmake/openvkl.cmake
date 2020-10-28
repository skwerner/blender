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


set(OPENVKL_EXTRA_ARGS
  -DOPENVKL_APPS=OFF
  -DTBB_ROOT=${LIBDIR}/tbb
  -DBUILD_EXAMPLES=OFF
  -DISPC_EXECUTABLE=${LIBDIR}/ispc/bin/ispc
  -Drkcommon_DIR=${LIBDIR}/rkcommon/lib/cmake/rkcommon-1.5.1
  -Dembree_DIR=${LIBDIR}/embree/lib/cmake/embree-3.10.0

)

if(WIN32)
  set(OPENVKL_EXTRA_ARGS
    ${OPENVKL_EXTRA_ARGS}
    -DTBB_DEBUG_LIBRARY=${LIBDIR}/tbb/lib/tbb.lib
    -DTBB_DEBUG_LIBRARY_MALLOC=${LIBDIR}/tbb/lib/tbbmalloc.lib
  )
else()
endif()

ExternalProject_Add(external_openvkl
  URL ${OPENVKL_URI}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH MD5=${OPENVKL_HASH}
  PREFIX ${BUILD_DIR}/openvkl
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/openvkl ${DEFAULT_CMAKE_FLAGS} ${OPENVKL_EXTRA_ARGS}
  INSTALL_DIR ${LIBDIR}/openvkl
)

add_dependencies(
  external_openvkl
  external_tbb
  external_ispc
  external_embree
  external_rkcommon
)

if(WIN32)
  if(BUILD_MODE STREQUAL Release)
    ExternalProject_Add_Step(external_openvkl after_install
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBDIR}/openvkl/include ${HARVEST_TARGET}/openvkl/include
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/openvkl/lib/openvkl.lib ${HARVEST_TARGET}/openvkl/lib/openvkl.lib
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/openvkl/lib/common.lib ${HARVEST_TARGET}/openvkl/lib/common.lib
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/openvkl/lib/dnnl.lib ${HARVEST_TARGET}/openvkl/lib/dnnl.lib
      DEPENDEES install
    )
  endif()
  if(BUILD_MODE STREQUAL Debug)
    ExternalProject_Add_Step(external_openvkl after_install
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/openvkl/lib/openvkl.lib ${HARVEST_TARGET}/openvkl/lib/openvkl_d.lib
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/openvkl/lib/common.lib ${HARVEST_TARGET}/openvkl/lib/common_d.lib
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/openvkl/lib/dnnl.lib ${HARVEST_TARGET}/openvkl/lib/dnnl_d.lib
      DEPENDEES install
    )
  endif()
endif()
