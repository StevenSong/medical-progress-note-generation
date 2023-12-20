import torch
from torch import nn
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder


class ClinicalEncoderConfig(BertConfig):
    model_type = "clinical"

    def __init__(self, *, feature_names: list[str] = [], **kwargs):
        super().__init__(**kwargs)
        self.feature_names = feature_names


class _FeatureMLP(nn.Module):
    def __init__(self, config: ClinicalEncoderConfig):
        super().__init__()
        mid_dim = config.hidden_size // 2
        self.fc1 = nn.Linear(1, mid_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mid_dim, config.hidden_size)
        self.nrm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.missing_embed = nn.Embedding(1, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x  # (B, T, 1) need last dim for fc1 matmul
        x = torch.nan_to_num(x)  # replace nan with 0, get summed away
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.nrm(x)  # (B, T, D)
        B, T, D = x.shape

        idx = torch.as_tensor([0], device=x.device)
        embed = self.missing_embed(idx)[None]  # (1, 1, D)
        embed = embed.expand(B, T, -1)  # (B, T, D)
        missing_mask = _x.isnan().to(x.device)  # (B, T, 1)
        missing_mask = missing_mask.float().expand(-1, -1, D)  # (B, T, D)
        x = x * (1 - missing_mask) + embed * missing_mask
        return x


class ClinicalEncoder(BertPreTrainedModel):
    config_class = ClinicalEncoderConfig
    base_model_prefix = "clinical"

    def __init__(self, config: ClinicalEncoderConfig):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.feature_names = config.feature_names
        self.feature_mlps = nn.ModuleList(
            [_FeatureMLP(config) for i in range(len(self.feature_names))]
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings)[None],
            persistent=False,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # see BertPooler
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

        # initialize weights
        self.post_init()

    def forward(self, clinical_features: torch.Tensor) -> torch.Tensor:
        # clinical_features is shape ([B], T, F) where:
        # - B is the batch size (optional)
        # - T is the time dimension
        # - F is the number of features
        single_input = clinical_features.dim() == 2
        if single_input:
            clinical_features = clinical_features[None]  # add fake batch dimension
        B, T, F = clinical_features.shape
        feat_encs = [
            enc(clinical_features[..., i : i + 1])
            for i, enc in enumerate(self.feature_mlps)
        ]
        feat_encs = torch.stack(feat_encs, dim=0)
        # feat_encs has shape (F, B, T, D) where D is encoder hidden_dim
        feat_sum = feat_encs.sum(dim=0)
        # feat_sum has shape (B, T, D)
        pos_ids = self.position_ids[:, :T]
        # pos_ids has shape (1, T)
        pos_embeds = self.position_embeddings(pos_ids)
        # pos_embeds has shape (1, T, D)
        enc_in = feat_sum + pos_embeds
        # enc_in has shape (B, T, D)

        enc_out = self.encoder(enc_in)[0]  # just use the hidden_states of xfrmer output
        # enc_out has shape (B, T, D)

        pool_in = enc_out[:, -1, :]  # use the encoding closest to note time
        # pool_in has shape (B, D)
        pool_out = self.act(self.fc(pool_in))
        # pool_out has shape (B, D)

        if single_input:
            pool_out = pool_out[0]  # get rid of batch dimension
        return pool_out
