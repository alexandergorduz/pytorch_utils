# PyTorch utilities module

A collection of utility functions for training and validating PyTorch models. This repository is designed to simplify and streamline deep learning workflows by providing reusable components for common training tasks, including support for TensorBoard logging.

## train_step

The `train_step` function performs a single training epoch for the given PyTorch model. It iterates through the training data, calculates the loss, and updates the model's parameters using backpropagation.

### Arguments

- `model`: PyTorch model to be trained.
- `train_dataloader`: DataLoader for the training dataset.
- `criterion`: loss function.
- `optimizer`: optimizer for training.
- `device`: device to run the training on (default: "cpu").

### Returns

- average `training loss` for the epoch.

## val_step

The `val_step` function performs evaluation of the given PyTorch model on a validation dataset. It computes the average loss over the entire validation set without updating the model's parameters.

### Arguments

- `model`: PyTorch model to be evaluated.
- `val_dataloader`: DataLoader for the validation dataset.
- `criterion`: loss function.
- `device`: device to run the validation on (default: "cpu").

### Returns

- average `validation loss`.

## train

The `train` function manages the complete training and validation process for a PyTorch model over multiple epochs. It combines the `train_step` and `val_step` functions to handle both training and evaluation, and optionally logs the results to TensorBoard.

### Arguments

- `model`: PyTorch model to be trained.
- `train_dataloader`: DataLoader for the training dataset.
- `val_dataloader`: DataLoader for the validation dataset.
- `criterion`: loss function.
- `optimizer`: optimizer for training.
- `epochs`: number of epochs.
- `writer`: (optional) TensorBoard SummaryWriter for logging.
- `device`: device to run training and validation on (default: "cpu").