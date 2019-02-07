/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2013 Blender Foundation.
 * All rights reserved.
 */

#include <stdlib.h>

#include "BLI_utildefines.h"
#include "IMB_imbuf.h"

struct ImBuf;

void IMB_freeImBuf(struct ImBuf *UNUSED(ibuf)) {}
void IMB_colormanagement_display_to_scene_linear_v3(float UNUSED(pixel[3]), struct ColorManagedDisplay *UNUSED(display)) {}
