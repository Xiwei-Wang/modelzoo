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

import abc
import math
import warnings
from bisect import bisect_right
from typing import List

import torch

from modelzoo.common.pytorch import cb_model as cm


class LRScheduler(torch.optim.lr_scheduler.LambdaLR, abc.ABC):
    """
    Cerebras specific learning rate scheduler base class.

    The learning rate schedulers implemented in this file are specifically
    designed to be run on a Cerebras system. This means that there are certain
    caveats to these custom schedulers that differ from a typical LR scheduler
    found in core PyTorch.

    The learning rate schedulers here are intended to be stepped at every
    iteration. This means `lr_scheduler.step()` should be called after every
    `optimizer.step()`. Hence, the learning rate schedulers operate on a
    step-by-step basis. Having said that, there are some variables used such
    as `last_epoch` that might indicate otherwise. The only reason these
    variables are used is to match what is used in core PyTorch. It does *not*
    indicate that things are operating on an epoch-by-epoch basis.

    Also, note that the above means that our LR schedulers are incompatible with
    the LR schedulers found in core PyTorch. The state cannot simply be transferred
    between the two. So, one of the LR schedulers defined here must be used in
    order to have LR scheduling on the Cerebras system.
    """

    global_start_step = 0
    initial_epoch = 0

    def __init__(
        self,
        optimizer,
        decay_steps: int = None,
        disable_lr_steps_reset: bool = False,
    ):
        self.decay_steps = decay_steps
        self.disable_lr_steps_reset = disable_lr_steps_reset

        self.start_step = LRScheduler.global_start_step
        if decay_steps is not None:
            LRScheduler.global_start_step += decay_steps

        self.cb_scheduler = None

        # Cerebras specific learning rate scheduler configuration
        if cm.use_cs():
            from cerebras.framework.torch.optim import Optimizer

            if not isinstance(optimizer, Optimizer):
                raise TypeError(
                    f"Expected a Cerebras Optimizer. Got: {type(optimizer)}"
                )

            self.cb_scheduler = self._configure_cerebras_lrs(optimizer)

            super().__init__(optimizer._optimizer, lr_lambda=self.lr_function)

            self._last_lr = 0.1
        else:
            super().__init__(optimizer, lr_lambda=self.lr_function)

        LRScheduler.initial_epoch = self.last_epoch

    def _configure_cerebras_lrs(self, optimizer):
        raise NotImplementedError(
            f"Cerebras LR scheduler configuration is not implemented for: {self}"
        )

    @abc.abstractmethod
    def lr_function(self, global_step):
        raise NotImplementedError(f"lr_function is not implemented for: {self}")

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "to get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        # note, different from the parent class,
        # we ignore the base learning rate entirely
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]

    def state_dict(self):
        state = super().state_dict()
        return {
            key: val
            for key, val in state.items()
            if key not in ("cb_scheduler",)
        }

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        LRScheduler.initial_epoch = self.last_epoch

        # Make sure the learning rate schedules are set properly
        if not cm.use_cs():
            self._step_count = 0
            self.last_epoch -= 1
            super().step()

    def step(self, *args, **kwargs):
        """
        Steps the scheduler and computes the latest learning rate

        Only sets the last_epoch if running on CS
        """
        if cm.use_cs():
            if self.last_epoch == -1:
                self.last_epoch = 0
            else:
                self.last_epoch = cm.get_run_step() + LRScheduler.initial_epoch
        else:
            super().step(*args, **kwargs)


class Constant(LRScheduler):
    """
    Constant update

    Args:
        optimizer: The optimizer to schedule
        val: The actual learning_rate value
        decay_steps: The number of steps to decay for
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        val: int,
        decay_steps: int = None,
        disable_lr_steps_reset: bool = False,
    ):
        self.val = val
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from cerebras.framework.torch.optim.lr_scheduler import Constant

        return Constant(
            optimizer, self.val, self.decay_steps, self.disable_lr_steps_reset
        )

    def lr_function(self, global_step):
        return self.val


class Polynomial(LRScheduler):
    """
    Polynomial Decay

    Args:
        optimizer: The optimizer to schedule
        learning_rate: The initial learning rate.
        end_learning_rate: The final learning rate
        decay_steps: Number of steps to perform the decay
        power: Exponent to apply to "x" (as in y=mx+b),
            which is ratio of step completion (1 for linear)
            Default: 1.0 (only Linear supported at the moment)
        cycle: Whether to cycle
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        end_learning_rate: float,
        decay_steps: int,
        power: float = 1.0,
        cycle: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from cerebras.framework.torch.optim.lr_scheduler import Polynomial

        return Polynomial(
            optimizer,
            self.learning_rate,
            self.end_learning_rate,
            self.decay_steps,
            self.power,
            self.cycle,
            self.disable_lr_steps_reset,
        )

    def lr_function(self, global_step):
        lr_diff = self.learning_rate - self.end_learning_rate
        alpha = 1
        if self.cycle:
            alpha = math.ceil((global_step + 1) / self.decay_steps)

        return (
            lr_diff
            * (1 - global_step / (self.decay_steps * alpha)) ** self.power
            + self.end_learning_rate
        )


class Exponential(LRScheduler):
    """
    Exponential Decay

    Args:
        optimizer: The optimizer to schedule
        learning_rate: The initial learning rate.
        decay_steps: Number of steps to perform the decay
        decay_rate: The decay rate
        staircase: If True decay the learning rate at discrete intervals
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        decay_steps: int,
        decay_rate: int,
        staircase: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = float(learning_rate)
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from cerebras.framework.torch.optim.lr_scheduler import Exponential

        return Exponential(
            optimizer,
            self.learning_rate,
            self.decay_steps,
            self.decay_rate,
            self.staircase,
            self.disable_lr_steps_reset,
        )

    def lr_function(self, global_step):
        power = global_step / self.decay_steps
        if self.staircase:
            power = math.floor(power)
        return self.learning_rate * (self.decay_rate ** power)


class InverseExponentialTimeDecay(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        step_exponent: int,
        decay_steps: int,
        decay_rate: int,
        staircase: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        self.step_exponent = step_exponent
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from cerebras.framework.torch.optim.lr_scheduler import (
            InverseExponentialTimeDecay,
        )

        return InverseExponentialTimeDecay(
            optimizer,
            self.learning_rate,
            self.decay_steps,
            self.decay_rate,
            self.staircase,
            self.disable_lr_steps_reset,
        )

    def lr_function(self, global_step):
        alpha = (global_step ** self.step_exponent) / self.decay_steps
        if self.staircase:
            alpha = math.floor(alpha)
        return self.learning_rate / (1 + self.decay_rate * alpha)


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def __init__(self, optimizer, *args, **kwargs):
        if cm.use_cs():
            from cerebras.framework.torch.optim import Optimizer

            if not isinstance(optimizer, Optimizer):
                raise TypeError(
                    f"Expected a Cerebras Optimizer. Got: {type(optimizer)}"
                )

            optimizer = optimizer._optimizer

        super().__init__(optimizer, *args, **kwargs)
        LRScheduler.initial_epoch = self.last_epoch
        self._init_step()

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        LRScheduler.initial_epoch = self.last_epoch
        self._init_step()

    def _init_step(self):
        # Step the current schedule once more in order to
        # make sure the learning rate is set properly
        if not cm.use_cs():
            idx = bisect_right(self._milestones, self.last_epoch)
            scheduler = self._schedulers[idx]
            if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
                scheduler.last_epoch = -1
                scheduler._step_count = 0
            else:
                scheduler.last_epoch -= 1
                scheduler._step_count -= 1

            scheduler.step()

            self._last_lr = scheduler.get_last_lr()

    def step(self):
        if cm.use_cs():
            self.last_epoch = cm.get_run_step() + LRScheduler.initial_epoch
        else:
            self.last_epoch += 1

        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.last_epoch = 0

        scheduler.step()

        if not cm.use_cs():
            self._last_lr = scheduler.get_last_lr()


class PiecewiseConstant(SequentialLR):
    def __init__(
        self,
        optimizer,
        learning_rates: List[float],
        milestones: List[int],
        disable_lr_steps_reset: bool = False,
    ):
        schedulers = []
        boundaries = [0]
        boundaries.extend(milestones)
        for lr, b1, b2 in zip(learning_rates, boundaries[:-1], boundaries[1:]):
            schedulers.append(
                Constant(optimizer, lr, b2 - b1, disable_lr_steps_reset)
            )
        # Final learning rate
        schedulers.append(
            Constant(
                optimizer,
                learning_rates[-1],
                disable_lr_steps_reset=disable_lr_steps_reset,
            )
        )

        super().__init__(optimizer, schedulers, milestones)
