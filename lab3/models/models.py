import torch.nn as nn
from transformers import AutoModel



def get_model(model_name, require_grad=False):
    model = AutoModel.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = require_grad
    return model


class SimplerSentenceClassifier(nn.Module):
    def __init__(self, model_name, hidden_dim=768, train_backbone=False):
        super().__init__()
        self.backbone = get_model(model_name, require_grad=train_backbone)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.backbone(**x).last_hidden_state
        x = self.classification_head(x[:, 0, :])
        return x








