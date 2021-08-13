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

#include "render/tile.h"

#include "render/pass.h"
#include "util/util_algorithm.h"
#include "util/util_foreach.h"
#include "util/util_logging.h"
#include "util/util_path.h"
#include "util/util_string.h"
#include "util/util_system.h"
#include "util/util_types.h"

CCL_NAMESPACE_BEGIN

TileManager::TileManager()
{
  /* Append an unique part to the file name, so that if the temp directory is not set to be a
   * process-specific there is no conflit between different Cycles process instances. Use process
   * ID to separate different processes, and address of the tile manager to identify different
   * Cycles sessions within the same process. */
  const string unique_part = to_string(system_self_process_id()) + "-" +
                             to_string(reinterpret_cast<uintptr_t>(this));
  tile_filepath_ = path_temp_get("cycles-tile-" + unique_part + ".exr");
}

TileManager::~TileManager()
{
}

void TileManager::reset(const BufferParams &params, int2 tile_size)
{
  VLOG(3) << "Using tile size of " << tile_size;

  close_tile_output();

  tile_size_ = tile_size;

  tile_state_.num_tiles_x = divide_up(params.width, tile_size_.x);
  tile_state_.num_tiles_y = divide_up(params.height, tile_size_.y);
  tile_state_.num_tiles = tile_state_.num_tiles_x * tile_state_.num_tiles_y;

  tile_state_.next_tile_index = 0;

  tile_state_.current_tile = Tile();
}

void TileManager::update_passes(const BufferParams &params, const vector<Pass *> &passes)
{
  DCHECK_NE(params.pass_stride, -1);

  buffer_params_ = params;

  configure_image_spec(passes);
}

bool TileManager::done()
{
  return tile_state_.next_tile_index == tile_state_.num_tiles;
}

bool TileManager::next()
{
  if (done()) {
    return false;
  }

  tile_state_.current_tile = get_tile_for_index(tile_state_.next_tile_index);

  ++tile_state_.next_tile_index;

  return true;
}

Tile TileManager::get_tile_for_index(int index) const
{
  /* TODO(sergey): Consider using hilbert spiral, or. maybe, even configurable. Not sure this
   * brings a lot of value since this is only applicable to BIG tiles. */

  const int tile_y = index / tile_state_.num_tiles_x;
  const int tile_x = index - tile_y * tile_state_.num_tiles_x;

  Tile tile;

  tile.x = tile_x * tile_size_.x;
  tile.y = tile_y * tile_size_.y;
  tile.width = tile_size_.x;
  tile.height = tile_size_.y;

  tile.width = min(tile.width, buffer_params_.width - tile.x);
  tile.height = min(tile.height, buffer_params_.height - tile.y);

  return tile;
}

const Tile &TileManager::get_current_tile() const
{
  return tile_state_.current_tile;
}

void TileManager::configure_image_spec(const vector<Pass *> &passes)
{
  static const char *component_suffixes[] = {"R", "G", "B", "A"};

  int pass_index = 0;
  int num_channels = 0;
  std::vector<std::string> channel_names;
  for (const Pass *pass : passes) {
    const PassInfo &pass_info = pass->get_info();
    num_channels += pass_info.num_components;

    /* EXR canonically expects first part of channel names to be sorted alphabetically, which is
     * not guaranteed to be the case with passes names. Assign a prefix based on the pass index
     * with a fixed width to ensure ordering. This makes it possible to dump existing render
     * buffers memory to disk and read it back without doing extra mapping. */
    const string prefix = string_printf("%08d", pass_index);

    const string channel_name_prefix = prefix + string(pass->name) + ".";

    for (int i = 0; i < pass_info.num_components; ++i) {
      channel_names.push_back(channel_name_prefix + component_suffixes[i]);
    }

    ++pass_index;
  }

  write_state_.image_spec = ImageSpec(
      buffer_params_.width, buffer_params_.height, num_channels, TypeDesc::FLOAT);

  write_state_.image_spec.channelnames = move(channel_names);
  write_state_.image_spec.tile_width = tile_size_.x;
  write_state_.image_spec.tile_height = tile_size_.y;
}

bool TileManager::open_tile_output()
{
  write_state_.tile_out = ImageOutput::create(tile_filepath_);
  if (!write_state_.tile_out) {
    LOG(ERROR) << "Error creating image output for " << tile_filepath_;
    return false;
  }

  if (!write_state_.tile_out->supports("tiles")) {
    LOG(ERROR) << "Progress tile file format does not support tiling.";
    return false;
  }

  write_state_.tile_out->open(tile_filepath_, write_state_.image_spec);
  write_state_.num_tiles_written = 0;

  VLOG(3) << "Opened tile file " << tile_filepath_;

  return true;
}

bool TileManager::close_tile_output()
{
  if (!write_state_.tile_out) {
    return true;
  }

  const bool success = write_state_.tile_out->close();
  write_state_.tile_out = nullptr;

  if (!success) {
    LOG(ERROR) << "Error closing tile file.";
    return false;
  }

  VLOG(3) << "Tile output is closed.";

  return true;
}

bool TileManager::write_tile(const RenderBuffers &tile_buffers)
{
  if (!write_state_.tile_out) {
    if (!open_tile_output()) {
      return false;
    }
  }

  DCHECK_EQ(tile_buffers.params.pass_stride, buffer_params_.pass_stride);

  const BufferParams &tile_params = tile_buffers.params;

  vector<float> pixel_storage;
  const float *pixels = tile_buffers.buffer.data();

  /* Tiled writing expects pixels to contain data for an entire tile. Pad the render buffers with
   * empty pixels for tiles which are on the image boundary. */
  if (tile_params.width != tile_size_.x || tile_params.height != tile_size_.y) {
    const int64_t pass_stride = tile_params.pass_stride;
    const int64_t src_row_stride = tile_params.width * pass_stride;

    const int64_t dst_row_stride = tile_size_.x * pass_stride;
    pixel_storage.resize(dst_row_stride * tile_size_.y);

    const float *src = tile_buffers.buffer.data();
    float *dst = pixel_storage.data();
    pixels = dst;

    for (int y = 0; y < tile_params.height; ++y, src += src_row_stride, dst += dst_row_stride) {
      memcpy(dst, src, src_row_stride * sizeof(float));
    }
  }

  const int tile_x = tile_params.full_x - buffer_params_.full_x;
  const int tile_y = tile_params.full_y - buffer_params_.full_y;

  VLOG(3) << "Write tile at " << tile_x << ", " << tile_y;
  if (!write_state_.tile_out->write_tile(tile_x, tile_y, 0, TypeDesc::FLOAT, pixels)) {
    LOG(ERROR) << "Error writing tile " << write_state_.tile_out->geterror();
  }

  ++write_state_.num_tiles_written;

  if (done()) {
    if (!close_tile_output()) {
      return false;
    }
  }

  return true;
}

void TileManager::finish_write_tiles()
{
  if (!write_state_.tile_out) {
    /* None of the tiles were written hence the file was not created.
     * Avoid creation of fully empty file since it is redundant. */
    return;
  }

  vector<float> pixel_storage(tile_size_.x * tile_size_.y * buffer_params_.pass_stride);

  for (int tile_index = write_state_.num_tiles_written; tile_index < tile_state_.num_tiles;
       ++tile_index) {
    const Tile tile = get_tile_for_index(tile_index);

    VLOG(3) << "Write dummy tile at " << tile.x << ", " << tile.y;

    write_state_.tile_out->write_tile(tile.x, tile.y, 0, TypeDesc::FLOAT, pixel_storage.data());
  }

  close_tile_output();
}

static bool check_image_spec_compatible(const ImageSpec &spec, const ImageSpec &expected_spec)
{
  if (spec.width != expected_spec.width || spec.height != expected_spec.height ||
      spec.depth != expected_spec.depth) {
    LOG(ERROR) << "Mismatched image dimension.";
    return false;
  }

  if (spec.format != expected_spec.format) {
    LOG(ERROR) << "Mismatched image format.";
  }

  if (spec.nchannels != expected_spec.nchannels) {
    LOG(ERROR) << "Mismatched number of channels.";
    return false;
  }

  if (spec.channelnames != expected_spec.channelnames) {
    LOG(ERROR) << "Mismatched channel names.";
    return false;
  }

  return true;
}

bool TileManager::read_full_buffer_from_disk(RenderBuffers *buffers)
{
  unique_ptr<ImageInput> in(ImageInput::open(tile_filepath_));
  if (!in) {
    LOG(ERROR) << "Error opening tile file " << tile_filepath_;
    return false;
  }

  const ImageSpec &spec = in->spec();
  if (!check_image_spec_compatible(spec, write_state_.image_spec)) {
    return false;
  }

  buffers->reset(buffer_params_);

  if (!in->read_image(TypeDesc::FLOAT, buffers->buffer.data())) {
    LOG(ERROR) << "Error reading pixels from the tile file " << in->geterror();
    return false;
  }

  if (!in->close()) {
    LOG(ERROR) << "Error closing tile file " << in->geterror();
    return false;
  }

  return true;
}

void TileManager::remove_tile_file() const
{
  VLOG(3) << "Removing tile file " << tile_filepath_;

  path_remove(tile_filepath_);
}

CCL_NAMESPACE_END
