import torch


class EarlyStopping:
    def __init__(
        self, class_weights, patience=5, min_delta=0.001, path="best_model.pth"
    ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """

        self.class_weights = class_weights
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss, model):
        # First epoch
        if self.best_loss is None:
            self.save_checkpoint(loss, model)
            self.best_loss = loss
        # change is within min_delta, start the early stopping counter
        elif abs(loss - self.best_loss) <= self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # Loss improved by more than delta
        elif loss < self.best_loss:
            self.save_checkpoint(loss, model)
            self.best_loss = loss
            self.counter = 0  # Reset counter
        # loss is greater than the best known so far
        else:
            return

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        print(f"Loss change ({self.best_loss} --> {val_loss}). Saving model...")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_weights": self.class_weights,
            },
            self.path,
        )
