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

#include "util/util_string.h"
#include "util/util_vector.h"

#include "kernel/kernel_types.h"

#include "graph/node.h"

CCL_NAMESPACE_BEGIN

class Pass : public Node {
 public:
  NODE_DECLARE

  Pass();

  PassType type;
  int components;
  bool filter;
  bool exposure;
  PassType divide_type;
  ustring name;

  /* The has been created automatically as a requirement to various rendering functionality (such
   * as adaptive sampling). */
  bool is_auto;

  /* Is true when the actual storage of the pass is not aligned to any of boundary.
   * For example, if the pass with 3 components is stored (and written by the kernel) as individual
   * float components. */
  bool is_unaligned;

  static const NodeEnum *get_type_enum();

  static void add(PassType type,
                  vector<Pass> &passes,
                  const char *name = nullptr,
                  bool is_auto = false);

  /* Check whether two sets of passes are matching exactly. */
  static bool equals_exact(const vector<Pass> &A, const vector<Pass> &B);

  /* Check whether two sets of passes define same set of non-auto passes. */
  static bool equals_no_auto(const vector<Pass> &A, const vector<Pass> &B);

  static bool contains(const vector<Pass> &passes, PassType type);

  /* Remove given pass type if it was automatically created. */
  static void remove_auto(vector<Pass> &passes, PassType type);

  /* Remove all passes which were automatically created. */
  static void remove_all_auto(vector<Pass> &passes);

  /* Returns nullptr if there is no pass with the given name or type. */
  static const Pass *find(const vector<Pass> &passes, const string &name);
  static const Pass *find(const vector<Pass> &passes, PassType type);

  /* Returns PASS_UNUSED if there is no pass with the given type. */
  static int get_offset(const vector<Pass> &passes, PassType type);
};

CCL_NAMESPACE_END
