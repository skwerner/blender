/*
 * Copyright 2011-2021 Blender Foundation
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

#pragma once

#include <ostream>  // NOLINT

#include "util/util_string.h"
#include "util/util_vector.h"

#include "kernel/kernel_types.h"

#include "graph/node.h"

CCL_NAMESPACE_BEGIN

const char *pass_type_as_string(const PassType type);

enum class PassMode {
  NOISY,
  DENOISED,
};
const char *pass_mode_as_string(PassMode mode);
std::ostream &operator<<(std::ostream &os, PassMode mode);

struct PassInfo {
  int num_components = -1;
  bool use_filter = false;
  bool use_exposure = false;
  PassType divide_type = PASS_NONE;

  /* Is false when the actual storage of the pass is not aligned to any of boundary.
   * For example, if the pass with 3 components is stored (and written by the kernel) as individual
   * float components. */
  bool is_aligned = true;

  /* Pass access for read can not happen directly and needs some sort of compositing (for example,
   * light passes due to divide_type, or shadow catcher pass. */
  bool use_compositing = false;

  /* Used to disable albedo pass for denoising.
   * Light and shadow catcher passes should not have discontinuity in the denoised result based on
   * the underlying albedo. */
  bool use_denoising_albedo = true;
};

class Pass : public Node {
 public:
  NODE_DECLARE

  PassType type;
  PassMode mode;
  ustring name;

  Pass();

  const PassInfo &get_info() const;

  /* The pass is written by the render pipeline (kernel or denoiser). If the pass is written it
   * will have pixels allocated in a RenderBuffer. Passes which are not written do not have their
   * pixels allocated to save memory. */
  bool is_written() const;

 protected:
  PassInfo info_;

  /* The has been created automatically as a requirement to various rendering functionality (such
   * as adaptive sampling). */
  bool is_auto_;

  /* The pass is written by the render pipeline. */
  bool is_written_;

 public:
  static const NodeEnum *get_type_enum();
  static const NodeEnum *get_mode_enum();

  static PassInfo get_info(PassType type);

  /* Add pass which is written by the kernel and is accessed directly. */
  static void add(vector<Pass> &passes, PassType type, const char *name = nullptr);

  /* Add pass which will be reading denoising result when it is available.
   * This pass is not allocated and is not written by the kernel unless denoiser is used. In this
   * case reading from this pass will fallback to reading noisy corresponding pass. */
  static void add_denoising_read(vector<Pass> &passes, PassType type, const char *name = nullptr);

  /* Add pass which will be written by a denoiser. */
  static void add_denoising_write(vector<Pass> &passes, PassType type, const char *name = nullptr);

  /* Add pass with the given configuration.
   * Note that this is only expected to be used by the Pass implementation and the render pipeline.
   */
  enum {
    FLAG_NONE = 0,

    /* The has been created automatically as a requirement to various rendering functionality (such
     * as adaptive sampling). */
    FLAG_AUTO = (1 << 0),

    /* Pass is created for read request, possibly will not be written by the render pipeline. */
    FLAG_READ_ONLY = (1 << 1),
  };
  static void add_internal(vector<Pass> &passes,
                           PassType type,
                           int flags,
                           const char *name = nullptr);
  static void add_internal(
      vector<Pass> &passes, PassType type, PassMode mode, int flags, const char *name = nullptr);

  static bool contains(const vector<Pass> &passes, PassType type, PassMode mode = PassMode::NOISY);

  /* Remove all passes which were automatically created. */
  static void remove_all_auto(vector<Pass> &passes);

  /* Returns nullptr if there is no pass with the given name or type+mode. */
  static const Pass *find(const vector<Pass> &passes, const string &name);
  static const Pass *find(const vector<Pass> &passes,
                          PassType type,
                          PassMode mode = PassMode::NOISY);

  /* Returns PASS_UNUSED if there is no corresponding pass. */
  static int get_offset(const vector<Pass> &passes, const Pass &pass);
};

std::ostream &operator<<(std::ostream &os, const Pass &pass);

CCL_NAMESPACE_END
