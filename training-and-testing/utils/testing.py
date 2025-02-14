import json
import os
import numpy as np
import matplotlib.pyplot as plt
import models
import torch
import torch.nn as nn
import ast
from .functions import DataGenerator
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


### MODEL IMPORT ###
def model_import(model_path, summary=True):
    """
    Import weights and initialize the model given its path.

    Returns:
    - model
    """
    model_name = (model_path.split('/')[12]).split('.')[0]
    init_model = getattr(models, model_name)
    model = init_model()
    weights = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    if summary:
        model.summary()
    return model


### HISTORY PLOT ###
def plot_training_history(history_path, early_stop):
    """
    Plots the training history (Loss vs Epoch).

    Parameters:
    - history_path: The path of the history dictionary returned by model.fit().
    """
    history_dict = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history_dict['train_loss']) + 1)
    print('Number of epochs:', len(history_dict['train_loss']))
    train_loss = history_dict['train_loss']
    val_loss = history_dict.get('val_loss', [])
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.axvline(len(history_dict['train_loss'])-early_stop, linestyle = 'dashed', label='Early stop epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()


def save_training_history(history_path, early_stop):
    history_dict = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history_dict['train_loss']) + 1)
    train_loss = history_dict['train_loss']
    val_loss = history_dict.get('val_loss', [])
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.axvline(len(history_dict['train_loss'])-early_stop, linestyle = 'dashed', label='Early stop epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()    
    save_path = os.path.dirname(history_path)
    plt.savefig(save_path+'/history_plot.png', format='png')
    print(f"Plot saved as {save_path+'/history_plot.png'}")


### TEST(S) ###
def test(model, label_type, test_level, info_path):
    """
    Test the given model on a specified test dataset and compute the loss and AUC score.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        label_type (str): The type of label.
        test_level (int): The test level (1, 2, or 3) that determines the specific test dataset.
        info_path (str): Path to the information of training.

    Returns:
        tuple: A tuple containing two values:
            - test_loss (float): The average binary cross-entropy loss on the test dataset.
            - auc (float): The area under the ROC curve (AUC) score for the model's predictions.
    """
    # paths
    data_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/'
    coordinates_path = data_path +f'test_{test_level}_coordinates.txt'
    if test_level == 1:
        mega_dict_name = 'train_mega_dict.json'
    elif test_level == 2:
        mega_dict_name = 'test_2_mega_dict.json'
    elif test_level == 3:
        mega_dict_name = 'test_3_mega_dict.json'        
    mega_dict_path = data_path + f'{label_type}/{mega_dict_name}'
    # test
    test_loader = create_data_loaders_test(info_path, coordinates_path, mega_dict_path)
    criterion = nn.BCELoss()
    test_loss = 0.
    all_labels = []
    all_preds = []
    model.eval()
    progress_bar = tqdm(test_loader, desc=f"Test", unit=" batch", ncols=100)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            avg_loss = test_loss/(progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)  
            labels = (labels >= 0.5).float()  
            all_labels.append(labels.numpy())
            all_preds.append(outputs.numpy())   
    test_loss /= len(test_loader)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    auc = roc_auc_score(all_labels, all_preds) 
    # print and save results
    print(f'Loss on test {test_level}: {test_loss}')
    print(f'AUC on test {test_level}: {auc}')
    results_path = os.path.dirname(info_path)
    with open(results_path+f'/results_test_{test_level}.txt', 'w') as file:
        file.write(f'Loss: {loss} \n')
        file.write(f'AUC: {auc}')
    print(f"Results saved as {results_path+f'/results_test_{test_level}.txt'}")


def combined_test(models, label_type, test_level, info_path, file_name):
    """
    Given a list of models, combines their outputs using a soft voting procedure (average probability).
    """
    data_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/'
    coordinates_path = data_path +f'test_{test_level}_coordinates.txt'
    if test_level == 1:
        mega_dict_name = 'train_mega_dict.json'
    elif test_level == 2:
        mega_dict_name = 'test_2_mega_dict.json'
    elif test_level == 3:
        mega_dict_name = 'test_3_mega_dict.json'        
    mega_dict_path = data_path + f'{label_type}/{mega_dict_name}'
    # test
    test_loader = create_data_loaders_test(info_path, coordinates_path, mega_dict_path)
    criterion = nn.BCELoss()
    test_loss = 0.
    all_labels = []
    all_preds = []
    for model in models:
        model.eval()
    progress_bar = tqdm(test_loader, desc=f"Test", unit=" batch", ncols=100)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            accumulated_outputs = torch.zeros_like(labels, dtype=torch.float32)
            for model in models:
                outputs = model(inputs)
                accumulated_outputs += outputs
            outputs = accumulated_outputs/len(models)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            avg_loss = test_loss/(progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)  
            labels = (labels >= 0.5).float()  
            all_labels.append(labels.numpy())
            all_preds.append(outputs.numpy())   
    test_loss /= len(test_loader)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    auc = roc_auc_score(all_labels, all_preds) 
    # print and save results
    print(f'Loss on test {test_level}: {test_loss}')
    print(f'AUC on test {test_level}: {auc}')
    results_path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/results/{label_type}/{file_name}_test_{test_level}.txt'
    with open(results_path, 'w') as file:
        file.write(f'Loss: {loss} \n')
        file.write(f'AUC: {auc}')
    print(f"Results saved as {results_path}")
    
    
### UTILS ###
def build_results_paths(label_type, model_name, training_name):
    """ 
    Build paths to training results.
    
    Returns:
    - paths for: model, history and info
    """
    results_path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/results/{label_type}/{model_name}/{training_name}'
    model_path = results_path + '/model.pt'    
    history_path = results_path + '/history.npy' 
    info_path = results_path + '/info.txt'
    return model_path, history_path, info_path


def create_data_loaders_test(info_path, coordinates_path, mega_dict_path, batch_size=64):
    """
    Given coordinates_path and mega_dict_path, it builds DataLoaders. From info_path recover window size.
    """
    with open(info_path, 'r') as file:
        for i in range(3):
            row = file.readline().strip()
            if i == 2:
                window_size = int(row.split()[2])
                break
    with open(mega_dict_path, 'r') as file:
        mega_dict = json.load(file)
    for contig in mega_dict:
        mega_dict[contig]['coverages'] = np.array(mega_dict[contig]['coverages'])
        mega_dict[contig]['labels'] = np.array(mega_dict[contig]['labels'])
    with open(coordinates_path, 'r') as file:
        for i in range(2):
            row = file.readline().strip()
            if i == 1: 
                coordinates = row.split(';')
    coordinates = [ast.literal_eval(item) for item in coordinates]
    data_gen = DataGenerator(mega_dict, coordinates, window_size=window_size)
    return DataLoader(data_gen, batch_size=batch_size, shuffle=False)