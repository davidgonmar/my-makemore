import torch  # noqa
import torch.nn as nn
import torch.utils.data as data
from typing import Optional


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: data.DataLoader,
        val_loader: Optional[data.DataLoader] = None,
    ):
        """
        Initializes the Trainer object.

        Args:
        model: A PyTorch model to be trained.
        optimizer: A PyTorch optimizer to be used during training.
        loss_fn: A PyTorch loss function to be used during training.
        train_loader: A PyTorch DataLoader object containing the training data.
        val_loader: A PyTorch DataLoader object containing the validation data.
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, epochs: int, valinterval: int = -1, verbose: bool = True) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
        epochs: An integer representing the number of epochs to train the model.
        valinterval: An integer representing the number of epochs between each validation step.
            If set to 0 or a negative number, no validation will be performed.
        """
        for epoch in range(epochs):
            self.model.train()
            # train for the whole dataset
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if verbose:
                    print(f"Epoch {epoch}, loss: {loss.item()}", end="\r")

            # if we want to validate, get the average validation loss
            if epoch % valinterval == 0 and valinterval > 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}, Validation loss: {val_loss}")

    def validate(self) -> float:
        """
        Validates the model on the validation set.
        """
        if self.val_loader is None:
            raise RuntimeError("No validation set provided.")
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        return total_loss / total_samples

    def get_model(self) -> nn.Module:
        """
        Returns the trained model.

        Returns:
        A PyTorch model.
        """
        return self.model
