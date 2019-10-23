
import torch.nn as nn
from torchvision import models as torchmodels

from core.model.base import BaseModel


class TorchvisionModel(BaseModel):
    def __init__(self, model_name, num_classes="same",
                 model_args={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = getattr(torchmodels, model_name)(**model_args)
        if num_classes != "same":
            assert isinstance(num_classes, int)
            # pythorch is inconsistent with the final layer
            if hasattr(self.model, 'fc'):
                num_ftrs = self.model.fc.in_features
            if hasattr(self.model, 'classifier'):
                num_ftrs = self.model.classifier.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, inputs):
        return self.model.forward(inputs)
