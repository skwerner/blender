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


set(RKCOMMON_EXTRA_ARGS
  -DTBB_ROOT=${LIBDIR}/tbb
  -DTBB_STATIC_LIB=${TBB_STATIC_LIBRARY}
  -DBUILD_TESTING=OFF
  -DINSTALL_DEPS=OFF
  -DBUILD_SHARED_LIBS=OFF
)

if(WIN32)
  set(RKCOMMON_EXTRA_ARGS
    ${RKCOMMON_EXTRA_ARGS}
    -DTBB_DEBUG_LIBRARY=${LIBDIR}/tbb/lib/tbb.lib
    -DTBB_DEBUG_LIBRARY_MALLOC=${LIBDIR}/tbb/lib/tbbmalloc.lib
  )
else()
  set(RKCOMMON_EXTRA_ARGS
    ${RKCOMMON_EXTRA_ARGS}
    -Dtbb_LIBRARY_RELEASE=${LIBDIR}/tbb/lib/libtbb_static.a
    -Dtbbmalloc_LIBRARY_RELEASE=${LIBDIR}/tbb/lib/libtbbmalloc_static.a
  )
endif()

ExternalProject_Add(external_rkcommon
  URL ${RKCOMMON_URI}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH MD5=${RKCOMMON_HASH}
  PREFIX ${BUILD_DIR}/rkcommon
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/rkcommon ${DEFAULT_CMAKE_FLAGS} ${RKCOMMON_EXTRA_ARGS}
  INSTALL_DIR ${LIBDIR}/rkcommon
)

add_dependencies(
  external_rkcommon
  external_tbb
)
