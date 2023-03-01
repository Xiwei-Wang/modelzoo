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

from modelzoo.common.pytorch.metrics import FBetaScoreMetric
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.data_processing.utils import get_label_id_map
from modelzoo.transformers.pytorch.bert.utils import check_unused_model_params
from modelzoo.transformers.pytorch.huggingface_common.modeling_bert import (
    BertConfig,
    BertForTokenClassification,
)


class BertForTokenClassificationModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        num_classes = model_params.pop("num_classes")
        self.model = BertForTokenClassification(
            BertConfig(
                vocab_size=model_params.pop("vocab_size"),
                hidden_size=model_params.pop("hidden_size"),
                num_hidden_layers=model_params.pop("num_hidden_layers"),
                num_attention_heads=model_params.pop("num_heads"),
                intermediate_size=model_params.pop("filter_size"),
                hidden_act=model_params.pop("encoder_nonlinearity"),
                num_labels=num_classes,
                classifier_dropout=model_params.pop(
                    "encoder_output_dropout_rate"
                ),
                max_position_embeddings=model_params.pop(
                    "max_position_embeddings"
                ),
                layer_norm_eps=float(model_params.pop("layer_norm_epsilon")),
            ),
            loss_weight=model_params.pop("loss_weight"),
            include_padding_in_loss=model_params.pop("include_padding_in_loss"),
        )
        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", False
        )
        if self.compute_eval_metrics:
            self.label_map_id = get_label_id_map(
                model_params.pop("label_vocab_file")
            )
            # Ignore token labels in eval which dont
            # refer to a token beginning or inside.
            # Labels such as
            # "O", [CLS], [SEP], [PAD], "O", "X"
            # are ignored during eval
            eval_ignore_labels = []
            if self.label_map_id is not None:
                for key, label_id in self.label_map_id.items():
                    if not (key.startswith("B") or key.startswith("I")):
                        eval_ignore_labels.append(label_id)
            self.f1_metric = FBetaScoreMetric(
                num_classes=num_classes,
                beta=1.0,
                average_type="macro",
                ignore_labels=eval_ignore_labels,
                name="eval/f1_score",
            )
        check_unused_model_params(model_params)
        self.loss_fn = self.model.loss_fn

        super(BertForTokenClassificationModel, self).__init__(
            params=params, model=self.model, device=device
        )

    def __call__(self, data):
        output = self.model(
            input_ids=data["input_ids"],
            token_type_ids=data["token_type_ids"],
            labels=data["labels"],
            attention_mask=data["attention_mask"],
            loss_mask=data["loss_mask"],
        )

        if not self.model.training and self.compute_eval_metrics:
            labels = data["labels"].clone()
            predictions = output.logits.argmax(-1).int()

            self.f1_metric(labels=labels, predictions=predictions)

        return output.loss
