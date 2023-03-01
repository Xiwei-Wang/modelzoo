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

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import json
import sys

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from transformers import GPTJConfig, GPTJForCausalLM


def get_runtime_args():
    """Create parser for command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Directory containing HuggingFace checkpoint. If it does not exist,"
            + " this path is used to store the downloaded checkpoint."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Model directory where converted checkpoints and states will be"
            + " written. If it does not exist, it is created during runtime."
        ),
    )
    parser.add_argument(
        "--share_embeddings",
        action='store_false',
        help=(
            "Whether to share embeddings with classifier layer for the model."
        ),
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help=("Whether to check model call from HuggingFace."),
    )
    return parser.parse_args(sys.argv[1:])


def create_pt_model(debug: bool = False):
    """Create a Gpt-J model using transformers PreTrainedConfig and output
    model parameters to a text file.

    Args:
        debug (bool): Enable debug for model creation

    Returns:
        A pytorch model with specified configuration
    """
    config = GPTJConfig(use_cache=False)
    model = GPTJForCausalLM(config)

    if debug:
        # print number of parameters for debuging
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        print(f"Initialized model with param count: {num_params}")

    return model


def get_weight_dict(tf_ckpt_path):
    """Reads TensorFlow checkpoint from specified path and returns
    the corresponding model's parameters as a dictionary of variable
    names to numpy arrays.

    Args:
        ckpt_path: (str) Path to TensorFlow checkpoint.
    Returns:
        A dictionary of variable names to numpy arrays with corresponding
        model parameters.
    """
    weight_dict = dict()
    reader = tf.compat.v1.train.NewCheckpointReader(tf_ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        weight_dict[key] = reader.get_tensor(key)
    return weight_dict


def load_mappings(tf_ckpt_path: str):
    """Load the mappings of variables between PyTorch and TensorFlow.

    Args:
        tf_ckpt_path (str): Path for storing checkpoints and associated mappings
    """
    with open(tf_ckpt_path + 'keys.json', 'r') as fin:
        key_mappings = json.load(fin)

    with open(tf_ckpt_path + 'shapes.json', 'r') as fin:
        shape_mappings = json.load(fin)

    return key_mappings, shape_mappings


def verify_pt_to_tf_conversion(
    pt_model: nn.Module,
    input_dir: str,
    output_dir: str,
    share_embeddings: bool,
):
    """Main function to verify conversion of PyTorch weights to TensorFlow.

    Args:
        pt_model (torch.nn.Module):
        pt_ckpt_path (str): Path to PyTorch checkpoint
        tf_ckpt_path (str): Path to TensorFlow checkpoint directory
        mapping_args (dict): An optional dictionary containing arguments for
            mapping. Defaults to `None`.
    """
    assert pt_model, "Expected a PyTorch model, got None"

    pt_ckpt_path = input_dir + "pytorch_model.bin"
    checkpoint = torch.load(pt_ckpt_path, map_location=torch.device('cpu'))
    pt_model.load_state_dict(checkpoint)
    pt_weight_dict = pt_model.state_dict()
    weight_keys = sorted(list(pt_weight_dict.keys()))

    tf_ckpt_path = output_dir + "tf_model.ckpt"
    keys_mapping, shape_mapping = load_mappings(output_dir)
    tf_weight_dict = get_weight_dict(tf_ckpt_path)

    is_successful = True
    for key in weight_keys:
        if share_embeddings and key == "lm_head.weight":
            continue
        if "attn.bias" or "attn.masked_bias" in key:
            continue

        tf_name, transpose = keys_mapping[key], shape_mapping[key]

        val = pt_weight_dict[key]
        update_val = val.T if transpose else val
        update_val = torch.Tensor(update_val)
        pt_weight = update_val.numpy()
        tf_weight = tf_weight_dict[tf_name]

        is_equal = np.array_equal(tf_weight, pt_weight)
        if not is_equal:
            is_successful = False
            _norm = np.linalg.norm(np.subtract(tf_weight, pt_weight))
            print(
                f"Tensors for {key} are off by {_norm}."
                + f" This is not a valid conversion of weights!!"
            )

    return is_successful


def validate_args(args):
    """Validate the user specified arguments.

    Args:
        args (namespace): Argparse arguments
    """
    if not os.path.isdir(args.input_dir):
        raise ValueError(
            "Input directory does not exist, cannot run verification."
        )

    if not os.path.isdir(args.output_dir):
        raise ValueError(
            "Output directory does not exist, cannot run verification."
        )

    pt_ckpt_path = args.input_dir + "pytorch_model.bin"
    if not os.path.isfile(pt_ckpt_path):
        raise ValueError(
            "Expected file for checkpoint, pass in correct path for PyTorch"
            + " checkpoint to load and verify."
        )

    for fname in os.listdir(args.output_dir):
        if "tf_model.ckpt" in fname:
            contains_filetype = True
            break

    if not contains_filetype:
        raise ValueError(
            "Expected file for checkpoint, pass in correct path for TensorFlow"
            + " checkpoint to load and verify."
        )


def main():
    args = get_runtime_args()
    validate_args(args)

    pt_model = create_pt_model(args.debug)
    print("Retrieved checkpoint and created HuggingFace model")

    # get TF checkpoint for shared embeddings setting
    is_successful = verify_pt_to_tf_conversion(
        pt_model, args.input_dir, args.output_dir, args.share_embeddings
    )
    if not is_successful:
        raise ValueError(
            "Conversion of HuggingFace checkpoint to Cerebras environment failed."
        )

    print(
        "Verified conversion of HuggingFace checkpoint to Cerebras environment."
    )


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    main()
