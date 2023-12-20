import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from models.clinical_encoder import ClinicalEncoder, ClinicalEncoderConfig
from models.text_encoder import TextEncoder, TextEncoderConfig


class MultimodalConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        initializer_range: float = 0.02,
        cln_enc_cfg=ClinicalEncoderConfig(),
        txt_enc_cfg=TextEncoderConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        # to be honest, not sure why this works, would need to dive into config.from_pretrained...
        if isinstance(cln_enc_cfg, dict):
            cln_enc_cfg = ClinicalEncoderConfig(**cln_enc_cfg)
        if isinstance(txt_enc_cfg, dict):
            txt_enc_cfg = TextEncoderConfig(**txt_enc_cfg)

        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.cln_enc_cfg = cln_enc_cfg
        self.txt_enc_cfg = txt_enc_cfg


class MultimodalEncoder(PreTrainedModel):
    config_class = MultimodalConfig
    base_model_prefix = "multimodal"

    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.cln_enc = ClinicalEncoder(config.cln_enc_cfg)
        self.txt_enc = TextEncoder(config.txt_enc_cfg)

        self.merge = self._make_merge_layer()
        self.act = nn.Tanh()

        self.post_init()

    def forward(
        self,
        *,
        clinical_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        cln_enc_out = self.cln_enc(clinical_features=clinical_features)  # ([B], D1)
        txt_enc_out = self.txt_enc(
            input_ids=input_ids, attention_mask=attention_mask
        )  # ([B], D2)
        merge_in = torch.cat((cln_enc_out, txt_enc_out), dim=-1)  # ([B], D1+D2)
        merge_out = self.merge(merge_in)  # ([B], D)
        act_out = self.act(merge_out)  # ([B], D)
        return act_out

    def _make_merge_layer(self):
        concat_size = (
            self.config.cln_enc_cfg.hidden_size + self.config.txt_enc_cfg.hidden_size
        )
        return nn.Linear(concat_size, self.config.hidden_size)

    # Gets called by post_init in init to initialize this module's weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    # Use the below methods to load modality specific encoders from pretrained checkpoints
    def text_from_pretrained(self, mpath):
        del self.txt_enc
        self.txt_enc = TextEncoder.from_pretrained(mpath)
        self.config.txt_enc_cfg = self.txt_enc.config
        self.merge = self._make_merge_layer()
        self.post_init()

    def clinical_from_pretrained(self, mpath):
        del self.cln_enc
        self.cln_enc = ClinicalEncoder.from_pretrained(mpath)
        self.config.cln_enc_cfg = self.cln_enc.config
        self.merge = self._make_merge_layer()
        self.post_init()
