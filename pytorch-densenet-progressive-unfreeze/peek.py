import torch.nn as nn
import torchvision.models as models


class MyClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.model = models.densenet121(weights=weights)

        num_ftrs = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits


if __name__ == "__main__":
    model = MyClassifier()

    for name, layer in model.model.features.named_children():
        print(name)
    for param in model.model.named_parameters():
        print(param)
    for param in model.model.classifier.parameters():
        param.requires_grad = True
