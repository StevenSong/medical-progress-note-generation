import torch
from torch import nn
from transformers import BertModel, BertConfig, BertPreTrainedModel


class TextEncoderConfig(BertConfig):
    model_type = "text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TextEncoder(BertPreTrainedModel):
    config_class = TextEncoderConfig
    base_model_prefix = "text"

    def __init__(self, config: TextEncoderConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        single_input = input_ids.dim() == 1
        if single_input:
            # add fake batch dimension
            input_ids = input_ids[None]
            attention_mask = attention_mask[None]
        enc_out = self.bert(
            input_ids,
            attention_mask,
        )[
            1
        ]  # just use the pooled output with shape (B, D)
        if single_input:
            enc_out = enc_out[0]  # get rid of batch dimension
        return enc_out
