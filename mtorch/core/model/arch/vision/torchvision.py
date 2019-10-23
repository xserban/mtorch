
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
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            elif hasattr(self.model, 'classifier'):
                try:
                    if len(self.model.classifier) > 1:
                        num_ftrs = self.model.classifier[-1].in_features
                        self.model.classifier[-1] = nn.Linear(
                            num_ftrs, num_classes)
                except Exception as e:
                    try:
                        num_ftrs = self.model.classifier.in_features
                        self.model.classifier = nn.Linear(
                            num_ftrs, num_classes)
                    except Exception as e:
                        print("[WARN] \t ", e)
                        raise e

    def forward(self, inputs):
        return self.model.forward(inputs)
