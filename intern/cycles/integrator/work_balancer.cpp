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

CCL_NAMESPACE_BEGIN

void work_balance_do_initial(vector<WorkBalanceInfo> &work_balance_infos)
{
  const int num_infos = work_balance_infos.size();

  if (num_infos == 1) {
    work_balance_infos[0].weight = 1.0;
    return;
  }

  const double weight = 1.0 / num_infos;
  for (WorkBalanceInfo &balance_info : work_balance_infos) {
    balance_info.weight = weight;
  }
}

bool work_balance_do_rebalance(vector<WorkBalanceInfo> &work_balance_infos)
{
  /* TODO(sergey): Needs implementation. */
  (void)work_balance_infos;
  return false;
}

CCL_NAMESPACE_END
