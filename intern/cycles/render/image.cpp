/*
 * Copyright 2011-2013 Blender Foundation
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

#include "render/image.h"
#include "device/device.h"
#include "render/colorspace.h"
#include "render/scene.h"
#include "render/stats.h"

#include "util/util_foreach.h"
#include "util/util_image_impl.h"
#include "util/util_logging.h"
#include "util/util_path.h"
#include "util/util_progress.h"
#include "util/util_sparse_grid.h"
#include "util/util_texture.h"
#include "util/util_unique_ptr.h"

#ifdef WITH_OSL
#  include <OSL/oslexec.h>
#endif

#ifdef WITH_OPENVDB
#  include "render/openvdb.h"
#endif

CCL_NAMESPACE_BEGIN

namespace {

/* Some helpers to silence warning in templated function. */
bool isfinite(uchar /*value*/)
{
  return true;
}
bool isfinite(half /*value*/)
{
  return true;
}
bool isfinite(uint16_t /*value*/)
{
  return true;
}

/* The lower three bits of a device texture slot number indicate its type.
 * These functions convert the slot ids from ImageManager "images" ones
 * to device ones and vice verse.
 */
int type_index_to_flattened_slot(int slot, ImageDataType type)
{
  return (slot << IMAGE_DATA_TYPE_SHIFT) | (type);
}

int flattened_slot_to_type_index(int flat_slot, ImageDataType *type)
{
  *type = (ImageDataType)(flat_slot & IMAGE_DATA_TYPE_MASK);
  return flat_slot >> IMAGE_DATA_TYPE_SHIFT;
}

const char *name_from_type(ImageDataType type)
{
  switch (type) {
    case IMAGE_DATA_TYPE_FLOAT4:
      return "float4";
    case IMAGE_DATA_TYPE_BYTE4:
      return "byte4";
    case IMAGE_DATA_TYPE_HALF4:
      return "half4";
    case IMAGE_DATA_TYPE_FLOAT:
      return "float";
    case IMAGE_DATA_TYPE_BYTE:
      return "byte";
    case IMAGE_DATA_TYPE_HALF:
      return "half";
    case IMAGE_DATA_TYPE_USHORT4:
      return "ushort4";
    case IMAGE_DATA_TYPE_USHORT:
      return "ushort";
    case IMAGE_DATA_NUM_TYPES:
      assert(!"System enumerator type, should never be used");
      return "";
  }
  assert(!"Unhandled image data type");
  return "";
}

const char *name_from_grid_type(int type)
{
  switch (type) {
    case IMAGE_GRID_TYPE_SPARSE:
      return "sparse";
    case IMAGE_GRID_TYPE_SPARSE_PAD:
      return "padded sparse";
    case IMAGE_GRID_TYPE_OPENVDB:
      return "OpenVDB";
    default:
      return "dense";
  }
}

}  // namespace

ImageManager::ImageManager(const DeviceInfo &info)
{
  need_update = true;
  osl_texture_system = NULL;
  animation_frame = 0;

  /* Set image limits */
  max_num_images = TEX_NUM_MAX;
  has_half_images = info.has_half_images;

  for (size_t type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    tex_num_images[type] = 0;
  }
}

ImageManager::~ImageManager()
{
  for (size_t type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (size_t slot = 0; slot < images[type].size(); slot++)
      assert(!images[type][slot]);
  }
}

void ImageManager::set_osl_texture_system(void *texture_system)
{
  osl_texture_system = texture_system;
}

bool ImageManager::set_animation_frame_update(int frame)
{
  if (frame != animation_frame) {
    animation_frame = frame;

    for (size_t type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
      for (size_t slot = 0; slot < images[type].size(); slot++) {
        if (images[type][slot] && images[type][slot]->animated)
          return true;
      }
    }
  }

  return false;
}

device_memory *ImageManager::image_memory(int flat_slot)
{
  ImageDataType type;
  int slot = flattened_slot_to_type_index(flat_slot, &type);

  Image *img = images[type][slot];

  return img->mem;
}

bool ImageManager::get_image_metadata(int flat_slot, ImageMetaData &metadata)
{
  if (flat_slot == -1) {
    return false;
  }

  ImageDataType type;
  int slot = flattened_slot_to_type_index(flat_slot, &type);

  Image *img = images[type][slot];
  if (img) {
    metadata = img->metadata;
    return true;
  }

  return false;
}

void ImageManager::metadata_detect_colorspace(ImageMetaData &metadata, const char *file_format)
{
  /* Convert used specified color spaces to one we know how to handle. */
  metadata.colorspace = ColorSpaceManager::detect_known_colorspace(
      metadata.colorspace, file_format, metadata.is_float || metadata.is_half);

  if (metadata.colorspace == u_colorspace_raw) {
    /* Nothing to do. */
  }
  else if (metadata.colorspace == u_colorspace_srgb) {
    /* Keep sRGB colorspace stored as sRGB, to save memory and/or loading time
     * for the common case of 8bit sRGB images like PNG. */
    metadata.compress_as_srgb = true;
  }
  else {
    /* Always compress non-raw 8bit images as scene linear + sRGB, as a
     * heuristic to keep memory usage the same without too much data loss
     * due to quantization in common cases. */
    metadata.compress_as_srgb = (metadata.type == IMAGE_DATA_TYPE_BYTE ||
                                 metadata.type == IMAGE_DATA_TYPE_BYTE4);

    /* If colorspace conversion needed, use half instead of short so we can
     * represent HDR values that might result from conversion. */
    if (metadata.type == IMAGE_DATA_TYPE_USHORT) {
      metadata.type = IMAGE_DATA_TYPE_HALF;
    }
    else if (metadata.type == IMAGE_DATA_TYPE_USHORT4) {
      metadata.type = IMAGE_DATA_TYPE_HALF4;
    }
  }
}

bool ImageManager::get_image_metadata(const string &filename,
                                      const string &grid_name,
                                      void *builtin_data,
                                      ustring colorspace,
                                      ImageMetaData &metadata)
{
  metadata = ImageMetaData();
  metadata.colorspace = colorspace;

  if (builtin_data) {
    if (builtin_image_info_cb) {
      builtin_image_info_cb(filename, builtin_data, metadata);
    }
    else {
      return false;
    }

    if (metadata.is_float) {
      metadata.type = (metadata.channels > 1) ? IMAGE_DATA_TYPE_FLOAT4 : IMAGE_DATA_TYPE_FLOAT;
    }
    else {
      metadata.type = (metadata.channels > 1) ? IMAGE_DATA_TYPE_BYTE4 : IMAGE_DATA_TYPE_BYTE;
    }

    metadata_detect_colorspace(metadata, "");

    return true;
  }

  /* Perform preliminary checks, with meaningful logging. */
  if (!path_exists(filename)) {
    VLOG(1) << "File '" << filename << "' does not exist.";
    return false;
  }
  if (path_is_directory(filename)) {
    VLOG(1) << "File '" << filename << "' is a directory, can't use as image.";
    return false;
  }

#ifdef WITH_OPENVDB
  if (string_endswith(filename, ".vdb")) {
    if (!openvdb_has_grid(filename, grid_name)) {
      VLOG(1) << "File '" << filename << "' does not have grid '" << grid_name << "'.";
      return false;
    }
    int3 resolution = openvdb_get_resolution(filename);
    metadata.width = resolution.x;
    metadata.height = resolution.y;
    metadata.depth = resolution.z;
    metadata.is_float = true;
    metadata.is_half = false;

    if (grid_name == Attribute::standard_name(ATTR_STD_VOLUME_COLOR) ||
        grid_name == Attribute::standard_name(ATTR_STD_VOLUME_VELOCITY)) {
      metadata.channels = 4;
      metadata.type = IMAGE_DATA_TYPE_FLOAT4;
    }
    else {
      metadata.channels = 1;
      metadata.type = IMAGE_DATA_TYPE_FLOAT;
    }

    return true;
  }
#endif

  unique_ptr<ImageInput> in(ImageInput::create(filename));

  if (!in) {
    return false;
  }

  ImageSpec spec;
  if (!in->open(filename, spec)) {
    return false;
  }

  metadata.width = spec.width;
  metadata.height = spec.height;
  metadata.depth = spec.depth;
  metadata.compress_as_srgb = false;

  /* Check the main format, and channel formats. */
  size_t channel_size = spec.format.basesize();

  if (spec.format.is_floating_point()) {
    metadata.is_float = true;
  }

  for (size_t channel = 0; channel < spec.channelformats.size(); channel++) {
    channel_size = max(channel_size, spec.channelformats[channel].basesize());
    if (spec.channelformats[channel].is_floating_point()) {
      metadata.is_float = true;
    }
  }

  /* check if it's half float */
  if (spec.format == TypeDesc::HALF) {
    metadata.is_half = true;
  }

  /* set type and channels */
  metadata.channels = spec.nchannels;

  if (metadata.is_half) {
    metadata.type = (metadata.channels > 1) ? IMAGE_DATA_TYPE_HALF4 : IMAGE_DATA_TYPE_HALF;
  }
  else if (metadata.is_float) {
    metadata.type = (metadata.channels > 1) ? IMAGE_DATA_TYPE_FLOAT4 : IMAGE_DATA_TYPE_FLOAT;
  }
  else if (spec.format == TypeDesc::USHORT) {
    metadata.type = (metadata.channels > 1) ? IMAGE_DATA_TYPE_USHORT4 : IMAGE_DATA_TYPE_USHORT;
  }
  else {
    metadata.type = (metadata.channels > 1) ? IMAGE_DATA_TYPE_BYTE4 : IMAGE_DATA_TYPE_BYTE;
  }

  metadata_detect_colorspace(metadata, in->format_name());

  in->close();

  return true;
}

static bool image_equals(ImageManager::Image *image,
                         const string &filename,
                         const string &grid_name,
                         void *builtin_data,
                         InterpolationType interpolation,
                         ExtensionType extension,
                         ImageAlphaType alpha_type,
                         ustring colorspace)
{
  return image->filename == filename && image->grid_name == grid_name &&
         image->builtin_data == builtin_data && image->interpolation == interpolation &&
         image->extension == extension && image->alpha_type == alpha_type &&
         image->colorspace == colorspace;
}

int ImageManager::add_image(const string &filename,
                            const string &grid_name,
                            void *builtin_data,
                            bool animated,
                            float frame,
                            InterpolationType interpolation,
                            ExtensionType extension,
                            ImageAlphaType alpha_type,
                            ustring colorspace,
                            bool is_volume,
                            float isovalue,
                            ImageMetaData &metadata)
{
  Image *img;
  size_t slot;

  get_image_metadata(filename, grid_name, builtin_data, colorspace, metadata);
  ImageDataType type = metadata.type;

  thread_scoped_lock device_lock(device_mutex);

  /* No half textures on OpenCL, use full float instead. */
  if (!has_half_images) {
    if (type == IMAGE_DATA_TYPE_HALF4) {
      type = IMAGE_DATA_TYPE_FLOAT4;
    }
    else if (type == IMAGE_DATA_TYPE_HALF) {
      type = IMAGE_DATA_TYPE_FLOAT;
    }
  }

  /* Find existing image. */
  for (slot = 0; slot < images[type].size(); slot++) {
    img = images[type][slot];

    if (img &&
        image_equals(
            img, grid_name, filename, builtin_data, interpolation, extension, alpha_type, colorspace)) {
      if (img->frame != frame) {
        img->frame = frame;
        img->need_load = true;
      }
      if (img->alpha_type != alpha_type) {
        img->alpha_type = alpha_type;
        img->need_load = true;
      }
      if (img->colorspace != colorspace) {
        img->colorspace = colorspace;
        img->need_load = true;
      }
      if (!(img->metadata == metadata)) {
        img->metadata = metadata;
        img->need_load = true;
      }
      img->users++;
      return type_index_to_flattened_slot(slot, type);
    }
  }

  /* Find free slot. */
  for (slot = 0; slot < images[type].size(); slot++) {
    if (!images[type][slot])
      break;
  }

  /* Count if we're over the limit.
   * Very unlikely, since max_num_images is insanely big. But better safe
   * than sorry.
   */
  int tex_count = 0;
  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    tex_count += tex_num_images[type];
  }
  if (tex_count > max_num_images) {
    printf(
        "ImageManager::add_image: Reached image limit (%d), "
        "skipping '%s'\n",
        max_num_images,
        filename.c_str());
    return -1;
  }

  if (slot == images[type].size()) {
    images[type].resize(images[type].size() + 1);
  }

  /* Add new image. */
  img = new Image();
  img->filename = filename;
  img->grid_name = grid_name;
  img->builtin_data = builtin_data;
  img->metadata = metadata;
  img->need_load = true;
  img->animated = animated;
  img->frame = frame;
  img->interpolation = interpolation;
  img->extension = extension;
  img->users = 1;
  img->alpha_type = alpha_type;
  img->colorspace = colorspace;
  img->is_volume = is_volume;
  img->isovalue = isovalue;
  img->mem = NULL;

  images[type][slot] = img;

  ++tex_num_images[type];

  need_update = true;

  return type_index_to_flattened_slot(slot, type);
}

void ImageManager::add_image_user(int flat_slot)
{
  ImageDataType type;
  int slot = flattened_slot_to_type_index(flat_slot, &type);

  Image *image = images[type][slot];
  assert(image && image->users >= 1);

  image->users++;
}

void ImageManager::remove_image(int flat_slot)
{
  ImageDataType type;
  int slot = flattened_slot_to_type_index(flat_slot, &type);

  Image *image = images[type][slot];
  assert(image && image->users >= 1);

  /* decrement user count */
  image->users--;

  /* don't remove immediately, rather do it all together later on. one of
   * the reasons for this is that on shader changes we add and remove nodes
   * that use them, but we do not want to reload the image all the time. */
  if (image->users == 0)
    need_update = true;
}

void ImageManager::remove_image(const string &filename,
                                const string &grid_name,
                                void *builtin_data,
                                InterpolationType interpolation,
                                ExtensionType extension,
                                ImageAlphaType alpha_type,
                                ustring colorspace)
{
  size_t slot;

  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (slot = 0; slot < images[type].size(); slot++) {
      if (images[type][slot] && image_equals(images[type][slot],
                                             filename,
                                             grid_name,
                                             builtin_data,
                                             interpolation,
                                             extension,
                                             alpha_type,
                                             colorspace)) {
        remove_image(type_index_to_flattened_slot(slot, (ImageDataType)type));
        return;
      }
    }
  }
}

/* TODO(sergey): Deduplicate with the iteration above, but make it pretty,
 * without bunch of arguments passing around making code readability even
 * more cluttered.
 */
void ImageManager::tag_reload_image(const string &filename,
                                    const string &grid_name,
                                    void *builtin_data,
                                    InterpolationType interpolation,
                                    ExtensionType extension,
                                    ImageAlphaType alpha_type,
                                    ustring colorspace)
{
  for (size_t type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (size_t slot = 0; slot < images[type].size(); slot++) {
      if (images[type][slot] && image_equals(images[type][slot],
                                             filename,
                                             grid_name,
                                             builtin_data,
                                             interpolation,
                                             extension,
                                             alpha_type,
                                             colorspace)) {
        images[type][slot]->need_load = true;
        break;
      }
    }
  }
}

bool ImageManager::allocate_grid_info(Device *device,
                                      device_memory *tex_img,
                                      vector<int> *sparse_index)
{
  string info_name = string(tex_img->name) + "_info";
  device_vector<int> *tex_info = new device_vector<int>(device, info_name.c_str(), MEM_TEXTURE);

  int *ti;
  {
    thread_scoped_lock device_lock(device_mutex);
    ti = (int *)tex_info->alloc(sparse_index->size());
  }

  if (ti == NULL) {
    return false;
  }

  memcpy(ti, &(*sparse_index)[0], sparse_index->size() * sizeof(int));
  tex_img->grid_info = static_cast<void *>(tex_info);

  return true;
}

static bool image_associate_alpha(ImageManager::Image *img)
{
  /* For typical RGBA images we let OIIO convert to associated alpha,
   * but some types we want to leave the RGB channels untouched. */
  return !(ColorSpaceManager::colorspace_is_data(img->colorspace) ||
           img->alpha_type == IMAGE_ALPHA_IGNORE || img->alpha_type == IMAGE_ALPHA_CHANNEL_PACKED);
}

bool ImageManager::file_load_image_generic(Image *img, unique_ptr<ImageInput> *in)
{
  if (img->filename == "")
    return false;

  if (img->builtin_data) {
    /* load image using builtin images callbacks */
    if (!builtin_image_info_cb || !builtin_image_pixels_cb)
      return false;
  }
#ifdef WITH_OPENVDB
  else if (string_endswith(img->filename, ".vdb")) {
    /* NOTE: Error logging is done in meta data acquisition. */
    if (!path_exists(img->filename) || path_is_directory(img->filename)) {
      return false;
    }
    if (!openvdb_has_grid(img->filename, img->grid_name)) {
      return false;
    }
  }
#endif
  else {
    /* NOTE: Error logging is done in meta data acquisition. */
    if (!path_exists(img->filename) || path_is_directory(img->filename)) {
      return false;
    }

    /* load image from file through OIIO */
    *in = unique_ptr<ImageInput>(ImageInput::create(img->filename));

    if (!*in)
      return false;

    ImageSpec spec = ImageSpec();
    ImageSpec config = ImageSpec();

    if (!image_associate_alpha(img)) {
      config.attribute("oiio:UnassociatedAlpha", 1);
    }

    if (!(*in)->open(img->filename, spec, config)) {
      return false;
    }
  }

  /* we only handle certain number of components */
  if (!(img->metadata.channels >= 1 && img->metadata.channels <= 4)) {
    if (*in) {
      (*in)->close();
    }
    return false;
  }

  return true;
}

template<typename DeviceType>
void ImageManager::file_load_failed(Image *img,
                                    ImageDataType type,
                                    device_vector<DeviceType> *tex_img)
{
  VLOG(1) << "Failed to load " << path_filename(img->filename) << " (" << img->mem_name << ")";

  /* On failure to load, we set a 1x1 pixels pink image. */
  thread_scoped_lock device_lock(device_mutex);
  DeviceType *device_pixels = tex_img->alloc(1, 1);

  switch (type) {
    case IMAGE_DATA_TYPE_FLOAT4: {
      float4 *pixels = (float4 *)device_pixels;
      pixels[0].x = TEX_IMAGE_MISSING_R;
      pixels[0].y = TEX_IMAGE_MISSING_G;
      pixels[0].z = TEX_IMAGE_MISSING_B;
      pixels[0].w = TEX_IMAGE_MISSING_A;
      break;
    }
    case IMAGE_DATA_TYPE_FLOAT: {
      float *pixels = (float *)device_pixels;
      pixels[0] = TEX_IMAGE_MISSING_R;
      break;
    }
    case IMAGE_DATA_TYPE_BYTE4: {
      uchar4 *pixels = (uchar4 *)device_pixels;
      pixels[0].x = (TEX_IMAGE_MISSING_R * 255);
      pixels[0].y = (TEX_IMAGE_MISSING_G * 255);
      pixels[0].z = (TEX_IMAGE_MISSING_B * 255);
      pixels[0].w = (TEX_IMAGE_MISSING_A * 255);
      break;
    }
    case IMAGE_DATA_TYPE_BYTE: {
      uchar *pixels = (uchar *)device_pixels;
      pixels[0] = (TEX_IMAGE_MISSING_R * 255);
      break;
    }
    case IMAGE_DATA_TYPE_HALF4: {
      half4 *pixels = (half4 *)device_pixels;
      pixels[0].x = TEX_IMAGE_MISSING_R;
      pixels[0].y = TEX_IMAGE_MISSING_G;
      pixels[0].z = TEX_IMAGE_MISSING_B;
      pixels[0].w = TEX_IMAGE_MISSING_A;
      break;
    }
    case IMAGE_DATA_TYPE_HALF: {
      half *pixels = (half *)device_pixels;
      pixels[0] = TEX_IMAGE_MISSING_R;
      break;
    }
    case IMAGE_DATA_TYPE_USHORT4: {
      ushort4 *pixels = (ushort4 *)device_pixels;
      pixels[0].x = (TEX_IMAGE_MISSING_R * 65535);
      pixels[0].y = (TEX_IMAGE_MISSING_G * 65535);
      pixels[0].z = (TEX_IMAGE_MISSING_B * 65535);
      pixels[0].w = (TEX_IMAGE_MISSING_A * 65535);
      break;
    }
    case IMAGE_DATA_TYPE_USHORT: {
      uint16_t *pixels = (uint16_t *)device_pixels;
      pixels[0] = (TEX_IMAGE_MISSING_R * 65535);
      break;
    }
    default:
      assert(0);
  }

  /* Store image. */
  img->mem = tex_img;
  img->mem->interpolation = img->interpolation;
  img->mem->extension = img->extension;
  img->mem->grid_type = IMAGE_GRID_TYPE_DEFAULT;

  tex_img->copy_to_device();
}

#ifdef WITH_OPENVDB
template<typename DeviceType>
void ImageManager::file_load_extern_vdb(Device *device, Image *img, ImageDataType type)
{
  VLOG(1) << "Loading external VDB " << img->filename << ", Grid: " << img->grid_name;

  device_vector<DeviceType> *tex_img = new device_vector<DeviceType>(
      device, img->mem_name.c_str(), MEM_TEXTURE);

  /* Retrieve metadata. */
  if (!file_load_image_generic(img, NULL)) {
    file_load_failed<DeviceType>(img, type, tex_img);
    return;
  }

  const bool use_pad = (device->info.type == DEVICE_CUDA);
  int sparse_size = -1;
  vector<int> sparse_offsets;
  openvdb_load_preprocess(
      img->filename, img->grid_name, img->isovalue, use_pad, &sparse_offsets, sparse_size);

  /* Allocate space for image. */
  float *pixels;
  {
    thread_scoped_lock device_lock(device_mutex);
    if (use_pad && sparse_size > -1) {
      tex_img->grid_type = IMAGE_GRID_TYPE_SPARSE_PAD;
      int width = sparse_size /
                  (PADDED_TILE * PADDED_TILE * (type == IMAGE_DATA_TYPE_FLOAT4 ? 4 : 1));
      pixels = (float *)tex_img->alloc(width, PADDED_TILE, PADDED_TILE);
    }
    else if (sparse_size > -1) {
      tex_img->grid_type = IMAGE_GRID_TYPE_SPARSE;
      pixels = (float *)tex_img->alloc(sparse_size);
    }
    else {
      tex_img->grid_type = IMAGE_GRID_TYPE_DEFAULT;
      pixels = (float *)tex_img->alloc(
          img->metadata.width, img->metadata.height, img->metadata.depth);
    }
  }

  if (!pixels) {
    /* Could be that we've run out of memory. */
    file_load_failed<DeviceType>(img, type, tex_img);
    return;
  }

  /* Load image. */
  openvdb_load_image(img->filename, img->grid_name, &sparse_offsets, sparse_size, use_pad, pixels);

  /* Allocate space for sparse_index if it exists. */
  if (sparse_size > -1) {
    if (!allocate_grid_info(device, (device_memory *)tex_img, &sparse_offsets)) {
      /* Could be that we've run out of memory. */
      file_load_failed<DeviceType>(img, type, tex_img);
      return;
    }
  }

  /* Set metadata and copy. */
  tex_img->dense_width = img->metadata.width;
  tex_img->dense_height = img->metadata.height;
  tex_img->dense_depth = img->metadata.depth;
  tex_img->interpolation = img->interpolation;
  tex_img->extension = img->extension;

  img->mem = tex_img;

  thread_scoped_lock device_lock(device_mutex);
  tex_img->copy_to_device();
}
#endif

template<TypeDesc::BASETYPE FileFormat, typename StorageType, typename DeviceType>
void ImageManager::file_load_image(Device *device,
                                   Image *img,
                                   ImageDataType type,
                                   int texture_limit)
{
  device_vector<DeviceType> *tex_img = new device_vector<DeviceType>(
      device, img->mem_name.c_str(), MEM_TEXTURE);

  tex_img->grid_type = IMAGE_GRID_TYPE_DEFAULT;
  tex_img->interpolation = img->interpolation;
  tex_img->extension = img->extension;

  unique_ptr<ImageInput> in = NULL;
  if (!file_load_image_generic(img, &in)) {
    file_load_failed<DeviceType>(img, type, tex_img);
    return;
  }

  /* Get metadata. */
  int width = img->metadata.width;
  int height = img->metadata.height;
  int depth = img->metadata.depth;
  int components = img->metadata.channels;

  /* Read pixels. */
  vector<StorageType> pixels_storage;
  StorageType *pixels;
  size_t num_pixels = ((size_t)width) * height * depth;
  size_t max_size = max(max(width, height), depth);
  if (max_size == 0) {
    /* Don't bother with empty images. */
    file_load_failed<DeviceType>(img, type, tex_img);
    return;
  }

  /* Allocate memory as needed, may be smaller to resize down. */
  if (texture_limit > 0 && max_size > texture_limit) {
    pixels_storage.resize(num_pixels * 4);
    pixels = &pixels_storage[0];
  }
  else {
    thread_scoped_lock device_lock(device_mutex);
    pixels = (StorageType *)tex_img->alloc(width, height, depth);
  }

  if (pixels == NULL) {
    /* Could be that we've run out of memory. */
    file_load_failed<DeviceType>(img, type, tex_img);
    return;
  }

  bool cmyk = false;
  if (in) {
    /* Read pixels through OpenImageIO. */
    StorageType *readpixels = pixels;
    vector<StorageType> tmppixels;
    if (components > 4) {
      tmppixels.resize(((size_t)width) * height * components);
      readpixels = &tmppixels[0];
    }

    if (depth <= 1) {
      size_t scanlinesize = ((size_t)width) * components * sizeof(StorageType);
      in->read_image(FileFormat,
                     (uchar *)readpixels + (height - 1) * scanlinesize,
                     AutoStride,
                     -scanlinesize,
                     AutoStride);
    }
    else {
      in->read_image(FileFormat, (uchar *)readpixels);
    }

    if (components > 4) {
      size_t dimensions = ((size_t)width) * height;
      for (size_t i = dimensions - 1, pixel = 0; pixel < dimensions; pixel++, i--) {
        pixels[i * 4 + 3] = tmppixels[i * components + 3];
        pixels[i * 4 + 2] = tmppixels[i * components + 2];
        pixels[i * 4 + 1] = tmppixels[i * components + 1];
        pixels[i * 4 + 0] = tmppixels[i * components + 0];
      }
      tmppixels.clear();
    }

    cmyk = strcmp(in->format_name(), "jpeg") == 0 && components == 4;
    in->close();
  }
  else {
    /* Read pixels through callback. */
    if (FileFormat == TypeDesc::FLOAT) {
      builtin_image_float_pixels_cb(img->filename,
                                    img->builtin_data,
                                    (float *)&pixels[0],
                                    num_pixels * components,
                                    image_associate_alpha(img),
                                    img->metadata.builtin_free_cache);
    }
    else if (FileFormat == TypeDesc::UINT8) {
      builtin_image_pixels_cb(img->filename,
                              img->builtin_data,
                              (uchar *)&pixels[0],
                              num_pixels * components,
                              image_associate_alpha(img),
                              img->metadata.builtin_free_cache);
    }
    else {
      /* TODO(dingto): Support half for ImBuf. */
    }
  }

  /* The kernel can handle 1 and 4 channel images. Anything that is not a single
   * channel image is converted to RGBA format. */
  bool is_rgba = (type == IMAGE_DATA_TYPE_FLOAT4 || type == IMAGE_DATA_TYPE_HALF4 ||
                  type == IMAGE_DATA_TYPE_BYTE4 || type == IMAGE_DATA_TYPE_USHORT4);

  if (is_rgba) {
    const StorageType one = util_image_cast_from_float<StorageType>(1.0f);

    if (cmyk) {
      /* CMYK to RGBA. */
      for (size_t i = num_pixels - 1, pixel = 0; pixel < num_pixels; pixel++, i--) {
        float c = util_image_cast_to_float(pixels[i * 4 + 0]);
        float m = util_image_cast_to_float(pixels[i * 4 + 1]);
        float y = util_image_cast_to_float(pixels[i * 4 + 2]);
        float k = util_image_cast_to_float(pixels[i * 4 + 3]);
        pixels[i * 4 + 0] = util_image_cast_from_float<StorageType>((1.0f - c) * (1.0f - k));
        pixels[i * 4 + 1] = util_image_cast_from_float<StorageType>((1.0f - m) * (1.0f - k));
        pixels[i * 4 + 2] = util_image_cast_from_float<StorageType>((1.0f - y) * (1.0f - k));
        pixels[i * 4 + 3] = one;
      }
    }
    else if (components == 2) {
      /* Grayscale + alpha to RGBA. */
      for (size_t i = num_pixels - 1, pixel = 0; pixel < num_pixels; pixel++, i--) {
        pixels[i * 4 + 3] = pixels[i * 2 + 1];
        pixels[i * 4 + 2] = pixels[i * 2 + 0];
        pixels[i * 4 + 1] = pixels[i * 2 + 0];
        pixels[i * 4 + 0] = pixels[i * 2 + 0];
      }
    }
    else if (components == 3) {
      /* RGB to RGBA. */
      for (size_t i = num_pixels - 1, pixel = 0; pixel < num_pixels; pixel++, i--) {
        pixels[i * 4 + 3] = one;
        pixels[i * 4 + 2] = pixels[i * 3 + 2];
        pixels[i * 4 + 1] = pixels[i * 3 + 1];
        pixels[i * 4 + 0] = pixels[i * 3 + 0];
      }
    }
    else if (components == 1) {
      /* Grayscale to RGBA. */
      for (size_t i = num_pixels - 1, pixel = 0; pixel < num_pixels; pixel++, i--) {
        pixels[i * 4 + 3] = one;
        pixels[i * 4 + 2] = pixels[i];
        pixels[i * 4 + 1] = pixels[i];
        pixels[i * 4 + 0] = pixels[i];
      }
    }

    /* Disable alpha if requested by the user. */
    if (img->alpha_type == IMAGE_ALPHA_IGNORE) {
      for (size_t i = num_pixels - 1, pixel = 0; pixel < num_pixels; pixel++, i--) {
        pixels[i * 4 + 3] = one;
      }
    }

    if (img->metadata.colorspace != u_colorspace_raw &&
        img->metadata.colorspace != u_colorspace_srgb) {
      /* Convert to scene linear. */
      ColorSpaceManager::to_scene_linear(
          img->metadata.colorspace, pixels, width, height, depth, img->metadata.compress_as_srgb);
    }
  }

  /* Make sure we don't have buggy values. */
  if (FileFormat == TypeDesc::FLOAT) {
    /* For RGBA buffers we put all channels to 0 if either of them is not
     * finite. This way we avoid possible artifacts caused by fully changed
     * hue. */
    if (is_rgba) {
      for (size_t i = 0; i < num_pixels; i += 4) {
        StorageType *pixel = &pixels[i * 4];
        if (!isfinite(pixel[0]) || !isfinite(pixel[1]) || !isfinite(pixel[2]) ||
            !isfinite(pixel[3])) {
          pixel[0] = 0;
          pixel[1] = 0;
          pixel[2] = 0;
          pixel[3] = 0;
        }
      }
    }
    else {
      for (size_t i = 0; i < num_pixels; ++i) {
        StorageType *pixel = &pixels[i];
        if (!isfinite(pixel[0])) {
          pixel[0] = 0;
        }
      }
    }
  }

  /* Scale image down if needed. */
  if (pixels_storage.size() > 0) {
    float scale_factor = 1.0f;
    while (max_size * scale_factor > texture_limit) {
      scale_factor *= 0.5f;
    }
    VLOG(1) << "Scaling image " << img->filename << " by a factor of " << scale_factor << ".";
    vector<StorageType> scaled_pixels;
    size_t scaled_width, scaled_height, scaled_depth;
    util_image_resize_pixels(pixels_storage,
                             width,
                             height,
                             depth,
                             is_rgba ? 4 : 1,
                             scale_factor,
                             &scaled_pixels,
                             &scaled_width,
                             &scaled_height,
                             &scaled_depth);

    pixels = &scaled_pixels[0];
    width = scaled_width;
    height = scaled_height;
    depth = scaled_depth;
    max_size = max(max(width, height), depth);
    num_pixels = ((size_t)width) * height * depth;
  }

  /* Compress image if needed. */
  int3 sparse_resolution = make_int3(-1, -1, -1);
  if (img->is_volume) {
    vector<StorageType> sparse_pixels;
    vector<int> sparse_offsets;

    if (device->info.type == DEVICE_CUDA) {
      if (create_sparse_grid_pad<StorageType>(pixels,
                                              width,
                                              height,
                                              depth,
                                              components,
                                              img->filename,
                                              img->isovalue,
                                              &sparse_pixels,
                                              &sparse_offsets,
                                              sparse_resolution)) {
        pixels = &sparse_pixels[0];
        if (!allocate_grid_info(device, (device_memory *)tex_img, &sparse_offsets)) {
          /* Could be that we've run out of memory. */
          file_load_failed<DeviceType>(img, type, tex_img);
          return;
        }
        tex_img->grid_type = IMAGE_GRID_TYPE_SPARSE_PAD;
      }
    }
    else {
      if (create_sparse_grid<StorageType>(pixels,
                                          width,
                                          height,
                                          depth,
                                          components,
                                          img->filename,
                                          img->isovalue,
                                          &sparse_pixels,
                                          &sparse_offsets)) {
        pixels = &sparse_pixels[0];
        if (!allocate_grid_info(device, (device_memory *)tex_img, &sparse_offsets)) {
          /* Could be that we've run out of memory. */
          file_load_failed<DeviceType>(img, type, tex_img);
          return;
        }
        tex_img->grid_type = IMAGE_GRID_TYPE_SPARSE;
        sparse_resolution = make_int3(sparse_pixels.size() / components, 1, 1);
      }
    }
  }

  /* Store image. */
  StorageType *texture_pixels = NULL;
  {
    thread_scoped_lock device_lock(device_mutex);
    if (sparse_resolution.x > -1) {
      /* For sparse grids, the dimensions of the image do not match the
       * required storage space. */
      texture_pixels = (StorageType *)tex_img->alloc(
          sparse_resolution.x, sparse_resolution.y, sparse_resolution.z);
    }
    else {
      texture_pixels = (StorageType *)tex_img->alloc(width, height, depth);
    }
  }

  tex_img->dense_width = width;
  tex_img->dense_height = height;
  tex_img->dense_depth = depth;
}

void ImageManager::device_load_image(
    Device *device, Scene *scene, ImageDataType type, int slot, Progress *progress)
{
  if (progress->get_cancel())
    return;

  Image *img = images[type][slot];

  if (osl_texture_system && !img->builtin_data)
    return;

  string filename = path_filename(img->filename);
  progress->set_status("Updating Images", "Loading " + filename);

  /* Slot assignment */
  int flat_slot = type_index_to_flattened_slot(slot, type);
  img->mem_name = string_printf("__tex_image_%s_%03d", name_from_type(type), flat_slot);

  /* Free previous texture(s) in slot. */
  if (img->mem) {
    thread_scoped_lock device_lock(device_mutex);
    if ((img->mem->grid_type == IMAGE_GRID_TYPE_SPARSE ||
         img->mem->grid_type == IMAGE_GRID_TYPE_SPARSE_PAD) &&
        img->mem->grid_info) {
      device_memory *info = (device_memory *)img->mem->grid_info;
      delete info;
      img->mem->grid_info = NULL;
    }
    delete img->mem;
    img->mem = NULL;
  }

  /* Create new texture. */
  const int texture_limit = scene->params.texture_limit;
  const bool is_extern_vdb = string_endswith(img->filename, ".vdb");

  switch (type) {
    case IMAGE_DATA_TYPE_FLOAT4:
#ifdef WITH_OPENVDB
      if (is_extern_vdb)
        file_load_extern_vdb<float4>(device, img, type);
      else
#endif
        file_load_image<TypeDesc::FLOAT, float, float4>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_FLOAT:
#ifdef WITH_OPENVDB
      if (is_extern_vdb)
        file_load_extern_vdb<float>(device, img, type);
      else
#endif
        file_load_image<TypeDesc::FLOAT, float, float>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_BYTE4:
      file_load_image<TypeDesc::UINT8, uchar, uchar4>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_BYTE:
      file_load_image<TypeDesc::UINT8, uchar, uchar>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_HALF4:
      file_load_image<TypeDesc::HALF, half, half4>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_HALF:
      file_load_image<TypeDesc::HALF, half, half>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_USHORT4:
      file_load_image<TypeDesc::USHORT, uint16_t, ushort4>(device, img, type, texture_limit);
      break;
    case IMAGE_DATA_TYPE_USHORT:
      file_load_image<TypeDesc::USHORT, uint16_t, uint16_t>(device, img, type, texture_limit);
      break;
    default:
      assert(0);
  }

  img->need_load = false;

  if (img->mem) {
    VLOG(1) << "Loaded " << img->mem_name << " as " << name_from_grid_type(img->mem->grid_type)
            << " grid.";
  }
}

void ImageManager::device_free_image(Device *, ImageDataType type, int slot)
{
  Image *img = images[type][slot];
  VLOG(1) << "Freeing " << img->mem_name;

  if (img) {
    if (osl_texture_system && !img->builtin_data) {
#ifdef WITH_OSL
      ustring filename(images[type][slot]->filename);
      ((OSL::TextureSystem *)osl_texture_system)->invalidate(filename);
#endif
    }

    if (img->mem) {
      thread_scoped_lock device_lock(device_mutex);
      if ((img->mem->grid_type == IMAGE_GRID_TYPE_SPARSE ||
           img->mem->grid_type == IMAGE_GRID_TYPE_SPARSE_PAD) &&
          img->mem->grid_info) {
        device_memory *info = (device_memory *)img->mem->grid_info;
        delete info;
        img->mem->grid_info = NULL;
      }
      delete img->mem;
    }

    delete img;
    images[type][slot] = NULL;
    --tex_num_images[type];
  }
}

void ImageManager::device_update(Device *device, Scene *scene, Progress &progress)
{
  if (!need_update) {
    return;
  }

  TaskPool pool;
  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (size_t slot = 0; slot < images[type].size(); slot++) {
      if (!images[type][slot])
        continue;

      if (images[type][slot]->users == 0) {
        device_free_image(device, (ImageDataType)type, slot);
      }
      else if (images[type][slot]->need_load) {
        if (!osl_texture_system || images[type][slot]->builtin_data) {
          pool.push(function_bind(&ImageManager::device_load_image,
                                  this,
                                  device,
                                  scene,
                                  (ImageDataType)type,
                                  slot,
                                  &progress));
        }
      }
    }
  }

  pool.wait_work();

  need_update = false;
}

void ImageManager::device_update_slot(Device *device,
                                      Scene *scene,
                                      int flat_slot,
                                      Progress *progress)
{
  ImageDataType type;
  int slot = flattened_slot_to_type_index(flat_slot, &type);

  Image *image = images[type][slot];
  assert(image != NULL);

  if (image->users == 0) {
    device_free_image(device, type, slot);
  }
  else if (image->need_load) {
    if (!osl_texture_system || image->builtin_data) {
      device_load_image(device, scene, type, slot, progress);
    }
  }
}

void ImageManager::device_load_builtin(Device *device, Scene *scene, Progress &progress)
{
  /* Load only builtin images, Blender needs this to load evaluated
   * scene data from depsgraph before it is freed. */
  if (!need_update) {
    return;
  }

  TaskPool pool;
  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (size_t slot = 0; slot < images[type].size(); slot++) {
      if (!images[type][slot])
        continue;

      if (images[type][slot]->need_load) {
        if (images[type][slot]->builtin_data) {
          pool.push(function_bind(&ImageManager::device_load_image,
                                  this,
                                  device,
                                  scene,
                                  (ImageDataType)type,
                                  slot,
                                  &progress));
        }
      }
    }
  }

  pool.wait_work();
}

void ImageManager::device_free_builtin(Device *device)
{
  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (size_t slot = 0; slot < images[type].size(); slot++) {
      if (images[type][slot] && images[type][slot]->builtin_data)
        device_free_image(device, (ImageDataType)type, slot);
    }
  }
}

void ImageManager::device_free(Device *device)
{
  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    for (size_t slot = 0; slot < images[type].size(); slot++) {
      device_free_image(device, (ImageDataType)type, slot);
    }
    images[type].clear();
  }
}

void ImageManager::collect_statistics(RenderStats *stats)
{
  for (int type = 0; type < IMAGE_DATA_NUM_TYPES; type++) {
    foreach (const Image *image, images[type]) {
      stats->image.textures.add_entry(
          NamedSizeEntry(path_filename(image->filename), image->mem->memory_size()));
    }
  }
}

CCL_NAMESPACE_END
