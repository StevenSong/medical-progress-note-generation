import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class ClassifierConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(
        self,
        *,
        num_labels: int = 2,
        hidden_size: int = 768,
        initializer_range: float = 0.02,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.dropout = dropout


class Classifier(PreTrainedModel):
    config_class = ClassifierConfig
    base_model_prefix = "classifier"

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, embedding: torch.Tensor):
        dropped = self.dropout(embedding)
        logits = self.cls(dropped)
        return logits

    # Gets called by post_init in init to initialize this module's weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
