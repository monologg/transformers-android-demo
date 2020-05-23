# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team and Jangwon Park
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

import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import tensorflow as tf

from transformers.modeling_electra import (
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_tf_electra import (
    TFElectraMainLayer,
    TFElectraPreTrainedModel
)


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        pooled_output = discriminator_hidden_states[0][:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,)

        softmax_logits = self.softmax(logits)
        outputs = (softmax_logits,) + outputs

        return outputs  # (softmax_logits, logits)


class TFElectraForSequenceClassification(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config, name="electra")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, training=training
        )
        pooled_output = discriminator_hidden_states[0][:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,)

        softmax_logits = self.softmax(logits)
        outputs = (softmax_logits,) + outputs

        return outputs  # (softmax_logits, logits)
