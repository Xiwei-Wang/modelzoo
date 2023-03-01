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

import torch


class SGD(torch.optim.SGD):
    """
    SGD optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initializing optimizer state
    """

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                if group['momentum'] != 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(
                        p, device="cpu"
                    ).to(p.device)
