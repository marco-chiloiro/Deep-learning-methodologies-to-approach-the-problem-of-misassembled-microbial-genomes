import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def train_model(model, train_loader, val_loader, criterion, optimizer, device=0, num_epochs=10, patience=5, delta=1e-4, scheduler=None):
    """
    Train the PyTorch model with a dynamic progress bar that updates at each step,
    and evaluate on the validation set at the end of each epoch.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to train.
    - train_loader (torch.utils.data.DataLoader): The DataLoader providing the training data.
    - val_loader (torch.utils.data.DataLoader): The DataLoader providing the validation data.
    - criterion (torch.nn.Module): The loss function used to compute the training loss.
    - optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
    - num_epochs (int): Number of epochs to train the model.
    - patience (int): Number of epochs with no improvement on validation loss before stopping training.
    - delta (float): The minimum change in the validation loss to qualify as an improvement.
    - scheduler: learning rate scheduler

    Returns:
    - model (torch.nn.Module): The trained model after completing the specified number of epochs.
    - history (dict): A dictionary containing lists of 'train_loss' and 'val_loss' recorded at each epoch.
    """
    # Automatically choose device 
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device}')
        gpu_name = torch.cuda.get_device_name(0)  # Get the name of the GPU
        print(f"Using device: {device} - GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    model.to(device)  # Move model to the appropriate device (GPU or CPU)
    history = {'train_loss': [], 'val_loss' : []}
    # Initialize early stopping parameters
    best_val_loss = np.inf  
    epochs_without_improvement = 0
    best_model_wts = model.state_dict()  # Initialize the best model weights to the current model weights
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0  
        # Training loop with dynamic progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit=" batch", ncols=100)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            # Update the running loss and progress bar description
            running_loss += loss.item()
            avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)  
        # Print learning rate, training and validation loss after each epoch
        train_loss = running_loss / len(train_loader)
        val_loss, val_auc = evaluate_model(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], lr: {current_lr: .5f}, Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        # Check for improvement in validation loss for early stopping
        if best_val_loss - val_loss > delta:  
            best_val_loss = val_loss
            best_model_wts = model.state_dict()  
            epochs_without_improvement = 0  
            best_epoch = epoch+1
        else:
            epochs_without_improvement += 1
        # If no improvement for `patience` epochs, stop training early
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. Validation loss has not improved by more than {delta:.5f} for {patience} epochs.")
            break
    # Load the best model weights and return
    model.load_state_dict(best_model_wts)
    print(f"(Return best model on validation at epoch {best_epoch} with val_loss {best_val_loss:.4f}).")
    return model, history


def evaluate_model(model, val_loader, criterion):
    """
    Evaluate the model on the validation set, including both loss and AUC score.

    Parameters:
    - model (torch.nn.Module): The trained model.
    - val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
    - criterion (torch.nn.Module): The loss function used to compute the validation loss.

    Returns:
    - val_loss (float): The average loss on the validation set.
    - auc (float): The Area Under the Receiver Operating Characteristic Curve.
    """
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Collect the labels and predictions for AUC calculation
            labels = (labels >= 0.5).float()  
            all_labels.append(labels.numpy())
            # If the model is for binary classification or multiclass, use appropriate output
            all_preds.append(outputs.numpy())  
    val_loss /= len(val_loader)
    # Concatenate all labels and predictions
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    # Compute AUC
    auc = roc_auc_score(all_labels, all_preds) 
    return val_loss, auc


def save_model(model, history, model_path_dict, model_info):
    """
    Save the trained model, its training history, and performance metrics to disk.

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model.
    - history (dict): A dictionary containing the training history (loss, accuracy, AUC, etc.).
    - model_path_dict (dict): A dictionary containing information about the label and model name, which will be used to 
                              generate the folder path.
    - model_info (dict): A dictionary containing additional information about the model, such as the number of chimeras,
                         samples per chimera, batch size, etc.
    """
    # Create the directory to save outputs 
    base_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/results'
    model_path = base_path + f'/{model_path_dict["label"]}/{model_path_dict["model_name"]}'
    os.makedirs(model_path, exist_ok=True)
    training_folder_base = 'training_'
    folder_num = 1
    # Check for the next available folder number
    while os.path.exists(model_path + f'/{training_folder_base}{folder_num}'):
        folder_num += 1
    # Create the new training folder
    training_folder = model_path + f'/{training_folder_base}{folder_num}'
    os.makedirs(training_folder)
    # save history in a .npy file
    np.save(training_folder + '/history.npy', history)
    print(f"History saved in {training_folder}/history.npy")
    # save the model
    torch.save(model.state_dict(), training_folder + '/model.pt')
    print(f"Model saved in {training_folder}/model.pt")
    # save performance metrics in the info.txt file
    with open(training_folder + '/info.txt', 'w') as file:
        file.write(f"Total samples: {model_info['total_samples']}\n")
        file.write(f"Batch size: {model_info['batch_size']}\n")
        file.write(f"Window size: {model_info['window_size']}\n")
        file.write(f"Number of epochs set: {model_info['epochs']}\n")
        file.write(f"Number of epochs done: {len(history['train_loss'])}\n")
        file.write(f"Learning rate: {model_info['lr']}\n")
        file.write(f"Patience: {model_info['patience']}\n")
        file.write(f"Training loss: {history['train_loss'][-1]}\n")
        file.write(f"Validation loss: {history['val_loss'][-1]}\n")
        if 'auc' in history:
            file.write(f"Training AUC: {history['auc'][-1]}\n")
            file.write(f"Validation AUC: {history['val_auc'][-1]}\n")
    print(f"Model information saved in {training_folder}/info.txt")


def lr_scheduler(epochs, step_per_epochs, opt, warmup=0.1):
    """
    This function creates a learning rate scheduler that applies a two-phase learning rate adjustment: a warmup phase followed by a decay phase. The learning rate starts from 0 and linearly increases during the warmup period, then linearly decays to 0 after the warmup phase.
    
    Parameters:
    - epochs (int): The total number of epochs for training.
    - step_per_epochs (int): The number of steps (batches) per epoch.
    - opt (torch.optim.Optimizer): The optimizer to which the scheduler will be applied.
    - warmup (float, optional): The fraction of total steps to be used for warmup (default is 0.1 or 10%).
    
    Returns:
    - scheduler (torch.optim.lr_scheduler.LambdaLR): A learning rate scheduler that adjusts the learning rate according to the warmup and decay schedule.
    """
    num_training_steps = epochs * step_per_epochs  # Total number of steps
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% of steps as warmup
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))  # Linear warmup
        else:
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))  # Linear decay
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return scheduler
