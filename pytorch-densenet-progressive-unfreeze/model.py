import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2


class MyClassifier(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=False):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.model = models.densenet121(weights=weights)

        num_ftrs = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(num_ftrs, num_classes)
        )

        if freeze_backbone:
            # for name, layer in self.model.features.named_children():
            # ...     print(name)
            # ...
            # conv0
            # norm0
            # relu0
            # pool0
            # denseblock1
            # transition1
            # denseblock2
            # transition2
            # denseblock3
            # transition3
            # denseblock4
            # norm5
            # for layer in [
            #     model.features.denseblock3,
            #     model.features.transition3,
            #     model.features.denseblock4,
            #     model.features.norm5
            # ]:
            #     for param in layer.parameters():
            # param.requires_grad = True
            for param in self.model.named_parameters():
                if "denseblock4" in param[0]:
                    print(f"Unfreezing {param[0]}")
                    param[1].requires_grad = False
            # unfreeze classifier head
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        logits = self.model(x)
        return logits


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        models.DenseNet121_Weights.IMAGENET1K_V1.transforms(),
    ]
)
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# we use this to demonstrate how we load the model weights and class
# weights from the saved checkpoint file
# example, when you are laoding the checkpoint from a different modeule/python
# script
def test_loop(dataloader):
    model_data = torch.load("model.pth")
    # >>> type(model_data)
    # <class 'dict'>
    # >>> model_data.keys()
    # dict_keys(['model_state_dict', 'class_weights', 'loss_type'])

    model = MyClassifier()
    # we directly feed the state dict here
    model.load_state_dict(model_data["model_state_dict"])
    criterion = nn.CrossEntropyLoss(weight=model_data["class_weights"])

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    model = MyClassifier(freeze_backbone=True)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    weight_ratio = torch.tensor([1] * 10, dtype=torch.float32)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weight_ratio)
    criterion = nn.CrossEntropyLoss(weight=weight_ratio)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer)
        test_loop(test_dataloader)
    print("Done!")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_weights": weight_ratio,
            "loss_type": "cross_entropy_loss",
        },
        "model.pth",
    )
