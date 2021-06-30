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

#include "integrator/work_balancer.h"

#include "util/util_math.h"

CCL_NAMESPACE_BEGIN

void work_balance_do_initial(vector<WorkBalanceInfo> &work_balance_infos)
{
  const int num_infos = work_balance_infos.size();

  if (num_infos == 1) {
    work_balance_infos[0].weight = 1.0;
    return;
  }

  /* There is no statistics available, so start with an equal distribution. */
  const double weight = 1.0 / num_infos;
  for (WorkBalanceInfo &balance_info : work_balance_infos) {
    balance_info.weight = weight;
  }
}

/* Calculate time which takes for every work to complete a unit of work.
 * The result times are normalized so that their sum is 1. */
static vector<double> calculate_normalized_times_per_unit(
    const vector<WorkBalanceInfo> &work_balance_infos)
{
  const int num_infos = work_balance_infos.size();

  vector<double> times_per_unit;
  times_per_unit.reserve(num_infos);

  double total_time_per_unit = 0;
  for (const WorkBalanceInfo &work_balance_info : work_balance_infos) {
    /* The work did `total_work * weight`, and the time per unit is
     * `time_spent / (total_work * weight). The total amount of work is not known here, but it will
     * gets cancelled out during normalization anyway.
     *
     * Note that in some degenerated cases (when amount of work is smaller than amount of workers)
     * it is possible that the time and/or weight of the work is 0. */
    const double time_per_unit = work_balance_info.weight != 0 ?
                                     work_balance_info.time_spent / work_balance_info.weight :
                                     0;
    times_per_unit.push_back(time_per_unit);
    total_time_per_unit += time_per_unit;
  }

  const double total_time_per_unit_inv = 1.0 / total_time_per_unit;
  for (double &time_per_unit : times_per_unit) {
    time_per_unit *= total_time_per_unit_inv;
  }

  return times_per_unit;
}

/* Calculate weights for the more ideal distribution of work.
 * The calculation here is based on an observed performance of every worker: the amount of work
 * scheduler is proportional to the performance of the worker. Performance of the worker is an
 * inverse of the time-per-unit-work. */
static vector<double> calculate_normalized_weights(
    const vector<WorkBalanceInfo> &work_balance_infos)
{
  const int num_infos = work_balance_infos.size();

  const vector<double> times_per_unit = calculate_normalized_times_per_unit(work_balance_infos);

  vector<double> weights;
  weights.reserve(num_infos);

  double total_weight = 0;
  for (double time_per_unit : times_per_unit) {
    /* Note that in some degenerated cases (when amount of work is smaller than amount of workers)
     * it is possible that the time and/or weight of the work is 0. */
    const double weight = time_per_unit != 0 ? 1.0 / time_per_unit : 0;
    total_weight += weight;
    weights.push_back(weight);
  }

  const double total_weight_inv = 1.0 / total_weight;
  for (double &weight : weights) {
    weight *= total_weight_inv;
  }

  return weights;
}

static bool apply_new_weights(vector<WorkBalanceInfo> &work_balance_infos,
                              const vector<double> &new_weights)
{
  const int num_infos = work_balance_infos.size();

  bool has_big_difference = false;
  for (int i = 0; i < num_infos; ++i) {
    /* Apparently, there is no `ccl::fabs()`. */
    if (std::fabs(work_balance_infos[i].weight - new_weights[i]) > 0.02) {
      has_big_difference = true;
    }
  }

  if (!has_big_difference) {
    return false;
  }

  for (int i = 0; i < num_infos; ++i) {
    WorkBalanceInfo &info = work_balance_infos[i];
    info.weight = new_weights[i];
    info.time_spent = 0;
  }

  return true;
}

bool work_balance_do_rebalance(vector<WorkBalanceInfo> &work_balance_infos)
{
  const vector<double> new_weights = calculate_normalized_weights(work_balance_infos);

  return apply_new_weights(work_balance_infos, new_weights);
}

CCL_NAMESPACE_END
