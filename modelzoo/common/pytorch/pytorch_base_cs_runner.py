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

"""Module containing the Base PyTorch CS Runner"""

import abc
import os

import torch

import cerebras.framework.torch.core.cb_model as cm
from modelzoo.common.pytorch import modes
from modelzoo.common.pytorch.pytorch_base_runner import PyTorchBaseRunner

COMPILE_MSG = (
    "Compiling the model and programming onto fabric. "
    "This may take a few minutes."
)


class PyTorchBaseCSRunner(PyTorchBaseRunner, metaclass=abc.ABCMeta):
    """Base Class containing common elements between CS runner and CS compiler"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device = cm.device()

        # batch size to be inferred on first iteration
        self._batch_size = None

        # Number of replicas to use for multireplica
        # 1 replica meaning no multireplica and -1 meaning
        # choose optimal number of replicas
        num_replicas = self._runconfig.get("num_replicas", 1)
        if num_replicas == 1 and self._runconfig.get("multireplica"):
            num_replicas = -1

        cm.set_target_num_replicas(num_replicas)

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def backward(self, loss):
        """Runs the backward pass."""
        self._model.grad_scaler(loss).backward()

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train_and_eval(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: torch.utils.data.DataLoader,
    ):
        raise RuntimeError(
            "Training with Eval on CS is not currently supported."
        )

    @property
    def _perf_dir(self) -> str:
        """Return the directory to use for saving perfomance metrics."""
        return os.path.join(self._model_dir, "performance")

    @property
    def world_global_step(self):
        return self._global_step * cm.num_receivers()

    def _increment_global_step(self):
        self._global_step += cm.get_run_step() - self._run_step

    def _save_stream(self, data_loader, mode: str):
        if mode == modes.TRAIN_AND_EVAL:
            train_data_loader, eval_data_loader = data_loader
            self._save_stream(train_data_loader, modes.TRAIN)
            self._save_stream(eval_data_loader, modes.EVAL)
            return

        from cerebras.framework.torch.utils.data.dataloader import DataLoader

        assert isinstance(
            data_loader, DataLoader
        ), f"DataLoader type: {type(data_loader)}"
        # Use non parallel loader to save stream
        # pylint: disable=protected-access
        super()._save_stream(data_loader._loader, mode)
