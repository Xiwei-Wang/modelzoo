# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def set_defaults(params, mode=None):
    params["model"]["loss_weight"] = params["model"].get("loss_weight", 1.0)
    params["model"]["layer_norm_epsilon"] = params["model"].get(
        "layer_norm_epsilon", 1.0e-5
    )
    params["model"]["vocab_file"] = params["eval_input"]["vocab_file"]
