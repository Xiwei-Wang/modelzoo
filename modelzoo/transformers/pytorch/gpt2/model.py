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

import torch

from modelzoo.common.pytorch.metrics import AccuracyMetric, PerplexityMetric
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.gpt2.utils import set_custom_stack_params
from modelzoo.transformers.pytorch.huggingface_common.modeling_gpt2 import (
    GPT2Config,
    GPT2LMHeadModel,
)


class Gpt2Model(PyTorchBaseModel):
    """
    GPT-2 models
    """

    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        self.model = self.build_model(model_params)

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.perplexity_metric = PerplexityMetric(name="eval/lm_perplexity")
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

        super(Gpt2Model, self).__init__(
            params=params, model=self.model, device=device
        )

        # Add custom Cerebras stack flags
        set_custom_stack_params(params)

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, model_params):
        attention_type = model_params.pop("attention_type")
        if attention_type == "scaled_dot_product":
            scale_attn_weights = True
        elif attention_type == "dot_product":
            scale_attn_weights = False
        else:
            raise ValueError(
                "attention_type should be 'scaled_dot_product' or 'dot_product'."
            )

        kwargs = {
            # Embedding
            "vocab_size": model_params.pop("vocab_size"),
            "n_positions": model_params.pop("max_position_embeddings", 1024),
            "n_embd": model_params.pop("hidden_size"),
            "use_position_embedding": model_params.pop(
                "use_position_embedding"
            ),
            "tie_word_embeddings": model_params.pop(
                "share_embedding_weights", True,
            ),
            # Encoder
            "n_layer": model_params.pop("num_hidden_layers"),
            "resid_pdrop": model_params.pop("dropout_rate"),
            "layer_norm_epsilon": float(
                model_params.pop("layer_norm_epsilon", 1.0e-5),
            ),
            # Encoder - Attention
            "n_head": model_params.pop("num_heads"),
            "scale_attn_weights": scale_attn_weights,
            "use_projection_bias_in_attention": model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            "use_ffn_bias_in_attention": model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            "attn_pdrop": model_params.pop("attention_dropout_rate"),
            # Encoder - ffn
            "n_inner": model_params.pop("filter_size"),
            "activation_function": model_params.pop("nonlinearity", "gelu"),
            "use_ffn_bias": model_params.pop("use_ffn_bias", True),
            # Task-specific
            "use_bias_in_output": model_params.pop("use_bias_in_output", False),
        }

        if model_params.pop("position_embedding_type") != "learned":
            raise NotImplementedError(
                "Only learned position embeddings are supported."
            )

        self.loss_weight = model_params.pop("loss_weight", 1.0)

        model_params.pop("mixed_precision", None)
        if model_params:
            logging.warning(
                "The following model params are unused: "
                + ", ".join(model_params.keys())
            )

        model = GPT2LMHeadModel(
            GPT2Config(**kwargs), loss_weight=self.loss_weight,
        )
        self.loss_fn = model.loss_fn
        return model

    def __call__(self, data):
        kwargs = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "labels": data["labels"],
        }
        output = self.model(**kwargs)
        loss = output.loss
        lm_logits = output.logits

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            lm_labels = data["labels"].clone()
            lm_weights = data["attention_mask"].clone()
            lm_preds = lm_logits.argmax(-1).int()

            self.accuracy_metric(
                labels=lm_labels, predictions=lm_preds, weights=lm_weights,
            )

            unscaled_loss = loss * torch.tensor(
                lm_labels.shape[0] / self.loss_weight, dtype=torch.float32
            )
            self.perplexity_metric(
                labels=lm_labels, loss=unscaled_loss, weights=lm_weights,
            )

        return loss
