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

import logging
from pathlib import Path

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import modes
from modelzoo.common.pytorch.utils import update_params_from_file


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    run_dir = Path(__file__).parent.resolve()
    update_params_from_file(
        params, run_dir.joinpath("configs", "default_params.yaml")
    )

    # Pass settings into data loader.
    for model_key in (
        "disable_nsp",
        "vocab_size",
        "enable_vts",
        "mixed_precision",
    ):
        for input_key in ("train_input", "eval_input"):
            params[input_key][model_key] = params["model"].get(model_key)

    params["model"]["max_position_embeddings"] = params["model"].get(
        "max_position_embeddings", params["train_input"]["max_sequence_length"],
    )


def set_custom_stack_params(params):
    if cm.use_cs():
        import cerebras.framework.torch as cbtorch

        state = cbtorch.state()
        state.full_config.placement.optimize_buses.deltat_relative_margin = 0.5
        if params["train_input"]["max_sequence_length"] <= 512:
            state.full_config.matching.kernel.no_dcache_spill_splits = True

        runconfig_params = params["runconfig"]
        if runconfig_params["mode"] == modes.EVAL:
            state.full_config.matching.add_pack_and_unpack.max_egress_per_pack = (
                1
            )
            state.full_config.placement.prep_recolor_kernels.wrap_pack_kernel = (
                True
            )


def check_unused_model_params(model_params):
    """
    While setting up the model, we pop used settings from model_params.
    This function sends a warning about any unused parameters.
    """
    model_params.pop("to_float16", None)
    model_params.pop("mixed_precision", None)
    if len(model_params) > 0:
        logging.warning(
            "The following model params are unused: "
            + ", ".join(model_params.keys())
        )
