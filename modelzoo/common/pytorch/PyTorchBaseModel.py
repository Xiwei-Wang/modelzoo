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

"""
Abstract base class for PyTorch models.
"""
from abc import ABC, abstractmethod

import torch

from modelzoo.common.pytorch import amp
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import modes
from modelzoo.common.pytorch.gradient_clipper import GradientClipper
from modelzoo.common.pytorch.optim import lr_scheduler


class PyTorchBaseModel(ABC):
    def __init__(
        self, params: dict, model: torch.nn.Module, device: torch.device
    ):
        self.model = model
        if cm.use_cs():
            import cerebras.framework.torch as cbtorch

            self.model = cbtorch.module(self.model, device)
        elif device:
            self.model = self.model.to(device)

        self._post_device_transfer()

        self.mode = params["runconfig"]["mode"]
        self.mixed_precision = params["model"]["mixed_precision"]

        seed = params["runconfig"].get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)

        oparams = params["optimizer"]

        # Learning rate params
        self.lr_scheduler = None
        lr_params = {
            "learning_rate": oparams["learning_rate"],
            "disable_lr_steps_reset": oparams.get(
                "disable_lr_steps_reset", False
            ),
        }
        if not isinstance(lr_params["learning_rate"], (float, str, dict, list)):
            raise ValueError(
                f"Learning rate must be a float, a dict, or a list of dicts. "
                f"Got {type(lr_params['learning_rate'])}"
            )

        self.optimizer = None
        if self.mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
            self.optimizer = self._configure_optimizer(oparams)

            if cm.use_cs():
                import cerebras.framework.torch as cbtorch

                self.optimizer = cbtorch.optimizer(self.optimizer)

            self.lr_scheduler = self._configure_lr_scheduler(lr_params)

        if cm.use_cs():  # init grad scaler for mixed precision
            self.grad_scaler = amp.GradScaler(
                loss_scale=oparams.get("loss_scaling_factor"),
                initial_loss_scale=oparams.get("initial_loss_scale"),
                steps_per_increase=oparams.get("steps_per_increase"),
                min_loss_scale=oparams.get("min_loss_scale"),
                max_loss_scale=oparams.get("max_loss_scale"),
                max_gradient_norm=oparams.get("max_gradient_norm"),
                mixed_precision=self.mixed_precision,
            )

        if self.optimizer:
            # Gradient clipping params
            self.optimizer.gradient_clipper = GradientClipper(
                oparams.get("max_gradient_norm", 0.0),
                oparams.get("max_gradient_value", 0.0),
            )

    def train(self):
        """
        Sets the model into training mode, equivalent to .train() called on a torch.nn.Module.
        """
        self.model.train()
        self.mode = modes.TRAIN

    def eval(self):
        """
        Sets the model into eval mode, equivalent to .eval() called on a torch.nn.Module.
        """
        self.model.eval()
        self.mode = modes.EVAL

    @property
    def supported_cs_modes(self):
        """
        Returns a list of modes that are supported for CS runs.

        By default we support train and eval, however, this property
        is designed to be overriden on a model-by-model basis.
        """
        return (modes.TRAIN, modes.EVAL)

    @property
    def supported_non_cs_modes(self):
        """
        Returns a list of modes that are supported for non-CS (CPU/GPU) runs.

        By default we support train, eval and train_and_eval, however, this
        property is designed to be overriden on a model-by-model basis.
        """
        return (modes.TRAIN, modes.EVAL, modes.TRAIN_AND_EVAL)

    def supports_mode(self, mode) -> bool:
        if cm.use_cs():
            return mode in self.supported_cs_modes
        else:
            return mode in self.supported_non_cs_modes

    def _post_device_transfer(self):
        """
        Callback after model is copied to device, but before optimizers are
        configured.
        """

    def trainable_named_parameters(self):
        return self.model.named_parameters()

    def _configure_optimizer(self, oparams: dict):
        """
        Configure an optimizer based on the params and return it
        """
        if cm.use_cs():
            from modelzoo.common.pytorch.optim import (
                SGD,
                Adam,
                AdamW,
            )
        else:
            from torch.optim import SGD, Adam

            from modelzoo.transformers.pytorch.huggingface_common.optimization import (
                AdamW,
            )

        optimizer_type = oparams["optimizer_type"].lower()

        learning_rate = oparams["learning_rate"]
        if isinstance(learning_rate, (float, str)):
            learning_rate = float(learning_rate)
        else:  # Indicates learning rate scheduling which sets the LR in the scheduler
            learning_rate = 0.1

        if optimizer_type == "sgd":
            return SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=oparams["momentum"],
                weight_decay=oparams.get("weight_decay_rate", 0.0),
            )
        elif optimizer_type == "adamw":
            param_optimizer = list(self.trainable_named_parameters())
            no_decay = oparams.get(
                "exclude_from_weight_decay",
                ["bias", "LayerNorm.bias", "LayerNorm.weight"],
            )
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": oparams.get("weight_decay_rate", 0.0),
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            return AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(oparams.get("beta1", 0.9), oparams.get("beta2", 0.999)),
                correct_bias=oparams.get("correct_bias", False),
                eps=oparams.get("eps", 1e-6),
            )
        elif optimizer_type == "adam":
            param_optimizer = list(self.trainable_named_parameters())
            no_decay = oparams.get(
                "exclude_from_weight_decay",
                [
                    "bias",
                    "LayerNorm.bias",
                    "LayerNorm.weight",
                    "BatchNorm.bias",
                    "BatchNorm.weight",
                ],
            )
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": oparams.get("weight_decay_rate", 0.0),
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            return Adam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(oparams.get("beta1", 0.9), oparams.get("beta2", 0.999)),
                eps=oparams.get("eps", 1e-6),
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type {optimizer_type}. Supported types: ['SGD', 'AdamW', 'Adam']"
            )

    def get_optimizer(self):
        """
        Returns the optimizer associated with this model.
        """
        return self.optimizer

    def _configure_lr_scheduler(self, lr_params):
        """
        Initiates the LR Scheduler associated with this model.
        """
        learning_rate = lr_params["learning_rate"]
        disable_lr_steps_reset = lr_params["disable_lr_steps_reset"]

        def _set_initial_lr(optimizer, lr):
            for group in self.optimizer.param_groups:
                group['lr'] = float(lr)

        def _get_scheduler(optimizer, schedule_params):
            """
            Parses a dict of learning rate scheduler specifications and
            returns a learning rate tensor.

            :param dict schedule_params:
                    A dict with a "scheduler" key (e.g.,
                    schedule_params["scheduler"] = "Exponential") and all
                    params schedulers of that type need.

            :returns: The learning rate tensor.
            """
            scheduler = schedule_params["scheduler"].lower()

            # to handle discrepancy in step parameters
            if "steps" in scheduler:
                scheduler["decay_steps"] = scheduler["steps"]
            elif "decay_steps" in scheduler:
                scheduler["steps"] = scheduler["decay_steps"]

            def check_required_params(required_params):
                missing = list(set(required_params) - set(schedule_params))
                if missing:
                    raise ValueError(
                        f"Missing required parameters {missing} "
                        f"for the {scheduler} learning rate scheduler. "
                        f"Note, the {scheduler} learning rate scheduler "
                        f"requires the following parameters: {required_params}"
                    )

            if scheduler == "constant":
                check_required_params(["learning_rate"])
                return lr_scheduler.Constant(
                    optimizer,
                    val=schedule_params["learning_rate"],
                    decay_steps=schedule_params.get("steps", None),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "exponential":
                check_required_params(
                    ["initial_learning_rate", "decay_steps", "decay_rate"]
                )
                return lr_scheduler.Exponential(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    decay_steps=schedule_params["decay_steps"],
                    decay_rate=schedule_params["decay_rate"],
                    staircase=schedule_params.get("staircase", False),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "piecewiseconstant":
                check_required_params(["values", "boundaries"])
                return lr_scheduler.PiecewiseConstant(
                    optimizer,
                    learning_rates=schedule_params["values"],
                    milestones=schedule_params["boundaries"],
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler in ("polynomial", "linear"):
                check_required_params(
                    ["initial_learning_rate", "end_learning_rate", "steps"]
                )
                power = (
                    1.0
                    if scheduler == "linear"
                    else schedule_params.get("power", 1.0)
                )
                return lr_scheduler.Polynomial(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    end_learning_rate=schedule_params["end_learning_rate"],
                    decay_steps=schedule_params["steps"],
                    power=power,
                    cycle=schedule_params.get("cycle", False),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            elif scheduler == "inverseexponentialtimedecay":
                check_required_params(
                    [
                        "initial_learning_rate",
                        "step_exponent",
                        "decay_steps",
                        "decay_rate",
                    ]
                )
                return lr_scheduler.InverseExponentialTimeDecay(
                    optimizer,
                    learning_rate=float(
                        schedule_params["initial_learning_rate"]
                    ),
                    step_exponent=schedule_params["step_exponent"],
                    decay_steps=schedule_params["decay_steps"],
                    decay_rate=schedule_params["decay_rate"],
                    staircase=schedule_params.get("staircase", False),
                    disable_lr_steps_reset=disable_lr_steps_reset,
                )
            else:
                raise ValueError(f"Unsupported LR scheduler {scheduler}")

        # handle a constant learning rate
        # scientific notation (e.g. "1e-5") parsed as string in yaml
        if isinstance(learning_rate, (float, str)):
            _set_initial_lr(self.optimizer, learning_rate)

        # handle a single decay schedule
        elif isinstance(learning_rate, dict):
            return _get_scheduler(self.optimizer, learning_rate)

        elif isinstance(learning_rate, list):
            if len(learning_rate) == 1:
                return _get_scheduler(self.optimizer, learning_rate[0])
            else:

                for scheduler in learning_rate[:-1]:
                    assert "steps" in scheduler or "decay_steps" in scheduler, (
                        "Non final learning rate schedulers must have either "
                        "the 'steps' or 'decay_steps' parameter given."
                    )

                schedulers = [
                    _get_scheduler(self.optimizer, scheduler)
                    for scheduler in learning_rate
                ]
                milestones = [
                    scheduler.start_step for scheduler in schedulers[1:]
                ]

                return lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=schedulers,
                    milestones=milestones,
                )
        else:
            raise ValueError(
                f"Unsupported LR scheduler type {type(learning_rate)}"
                f"Supported LR schedulers are ['Constant', 'Exponential',"
                f" 'PiecewiseConstant', 'Polynomial',"
                f" 'InverseExponentialTimeDecay']"
            )

    def get_lr_scheduler(self):
        """
        Returns the LR Scheduler associated with this model.
        """
        return self.lr_scheduler

    def get_state(self):
        """
        Returns the state of the model and optimizer
        """
        state_dict = {
            "model": self.model.state_dict(),
        }

        if self.optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()

        if self.lr_scheduler:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        if self.mixed_precision and cm.is_wse_device():
            state_dict["amp"] = amp.state_dict()

        return state_dict

    def set_state(self, state, strict=True):
        """
        Sets the state of the model and optimizer
        """
        is_pretrained_checkpoint = self.params["runconfig"].get(
            "is_pretrained_checkpoint", False
        )
        mode = self.params["runconfig"]["mode"]
        if is_pretrained_checkpoint and mode != modes.EVAL:
            # allow loading weights ignoring the mising and unexpected keys
            # except when doing eval
            strict = False
        self.model.load_state_dict(state["model"], strict=strict)
        if (
            self.optimizer
            and "optimizer" in state
            and not is_pretrained_checkpoint
        ):
            # load optimizer state for resuming training
            self.optimizer.load_state_dict(state["optimizer"])
            if self.lr_scheduler and "lr_scheduler" in state:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if (
            self.mixed_precision
            and cm.is_wse_device()
            and not is_pretrained_checkpoint
        ):
            amp_state = state.get('amp')
            if amp_state:
                amp.load_state_dict(amp_state)

    @abstractmethod
    def __call__(self, data):
        """
        Given one iteration of a dataloader, returns the loss associated with
        one forward pass of that batch.
        """
        raise NotImplementedError(
            "__call__ must be implemented in a child class!"
        )
