import torch
from torch.utils.tensorboard import SummaryWriter



def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = "cpu") -> float:
    """
    Trains a given model for one epoch.

    Args:
        model (torch.nn.Module): the model to be trained.
        train_dataloader (torch.utils.data.DataLoader): the DataLoader for training data.
        criterion (torch.nn.Module): the loss function.
        optimizer (torch.optim.Optimizer): the optimizer for model parameters.
        device (torch.device, optional): the device on which to perform training, default is CPU.
    
    Returns:
        float: the average training loss for the epoch.
    """

    model = model.to(device)

    train_loss = 0.0

    model.train()

    for X, y in train_dataloader:

        X, y = X.to(device), y.to(device)

        logit = model(X)

        loss = criterion(logit, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    
    train_loss /= len(train_dataloader)

    return train_loss



def val_step(model: torch.nn.Module,
             val_dataloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             device: torch.device = "cpu") -> float:
    """
    Evaluates a given model on the validation dataset.

    Args:
        model (torch.nn.Module): the model to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): the DataLoader for validation data.
        criterion (torch.nn.Module): the loss function.
        device (torch.device, optional): the device on which to perform evaluation, default is CPU.
    
    Returns:
        float: the average validation loss.
    """

    model = model.to(device)

    val_loss = 0.0

    model.eval()

    with torch.inference_mode():

        for X, y in val_dataloader:

            X, y = X.to(device), y.to(device)

            logit = model(X)

            loss = criterion(logit, y)

            val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
    
    return val_loss



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          writer: SummaryWriter = None,
          device: torch.device = "cpu") -> None:
    """
    Trains and validates a given model for a specified number of epochs.

    Args:
        model (torch.nn.Module): the model to be trained and validated.
        train_dataloader (torch.utils.data.DataLoader): the DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): the DataLoader for validation data.
        criterion (torch.nn.Module): the loss function.
        optimizer (torch.optim.Optimizer): the optimizer for model parameters.
        epochs (int): number of epochs to train the model.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging, default is None.
        device (torch.device, optional): the device on which to perform training and validation, default is CPU.
    """

    for epoch in range(epochs):

        train_loss = train_step(model=model,
                                train_dataloader=train_dataloader,
                                criterion=criterion,
                                optimizer=optimizer,
                                device=device)
        
        val_loss = val_step(model=model,
                            val_dataloader=val_dataloader,
                            criterion=criterion,
                            device=device)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")

        if writer is not None:

            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={
                                   "train_loss": train_loss,
                                   "val_loss": val_loss
                               }, global_step=epoch)
    
    if writer is not None:

        writer.close()