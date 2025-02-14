import os
import ast
import json
import random
import numpy as np
import bz2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#### DATA IMPORT ####
def create_sgb_chimera_dict(N, n, num_genomes=2):
    """
    Creates a dictionary that randomly selects 'N' SGBs and, for each SGB, selects 'n' random chimeras.
    
    Parameters:
    N (int): Number of SGBs to randomly select.
    n (int): Number of chimeras to randomly select for each SGB.
    num_genomes (int) : Number of genomes used per chimera.
    
    Returns:
    dict: Dictionary of selected SGBs with their randomly selected chimeras.
    """
    base_path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/data/chimeras_{num_genomes}genomes'
    sgbs = os.listdir(base_path)
    sgb_chimera_dict = {}
    for sgb in sgbs:
        sgb_chimera_dict[sgb] = os.listdir(base_path+f'/{sgb}')
    # Randomly pick 7 SGBs from the dictionary
    selected_sgb = random.sample(list(sgb_chimera_dict.keys()), N)
    # Initialize a new dictionary to store the selected SGBs and chimeras
    selected_chimera_dict = {}
    # For each selected SGB, pick 2 random chimeras
    for sgb in selected_sgb:
        chimeras = sgb_chimera_dict[sgb]
        # Randomly pick 2 chimeras for this SGB
        selected_chimeras = random.sample(chimeras, n)
        # Add the selected chimeras to the dictionary
        selected_chimera_dict[sgb] = selected_chimeras
    # Display the selected SGBs and their chimeras
    return selected_chimera_dict


def create_chimera_paths(sgb_chim_dict, coverage=100):
    """
    This function generates file paths for chimera data based on the provided dictionary.

    Parameters:
    sgb_chim_dict (dict): A dictionary where keys represent SGBs (Species Genome Bins) and values are lists of chimeras.
    coverage (int): It indicates the folder name corresponding to the reads simulation's coverage parameter. 

    Returns:
    paths (list): A list of file paths corresponding to each chimera in the input dictionary.
    """
    # find out number of genomes from sgb_chim_dict
    num_genomes = len(sgb_chim_dict[list(sgb_chim_dict.keys())[0]][0].split('_'))
    base_path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/data/chimeras_{num_genomes}genomes'  
    paths = []
    for sgb in sgb_chim_dict:
        for chim in sgb_chim_dict[sgb]:
            paths.append(f'{base_path}/{sgb}/{chim}/coverage_{coverage}/{chim}.seqcov.bz2')
    return paths


def import_seqcov_bzip_file(file_path, prefix=None):
    """
    Imports a seqcovent formatted bzip2 compressed.

    Parameters:
    file_path (str): Path to the input bzip2 file.
    prefix (str): Prefix to add to the contig names.

    Returns:
    dictionary: A dictionary of dictionaries containing the parsed data for each contig (json like).
    """
    # extrapolate information from the path (num_genomes)
    splitted = file_path.split('/')
    num_genomes = len(splitted[-3].split('_'))
    contigs = {}  # Dictionary to store data for each contig
    current_contig = None  # To hold data for the current contig
    with bz2.open(file_path, 'rt') as file:  # Open the bzip2 file in text mode
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            # Check for general header line
            if line.startswith('#'):
                continue  # Skip general header lines
            # Check for contig header
            if line.startswith('>'):
                # If there's a current contig, save it before starting a new one
                if current_contig is not None:
                    contigs[contig_name] = current_contig
                # Start a new contig
                header_parts = line[1:].split()  # Split header
                contig_info = {'contig_name':header_parts[0], header_parts[1].split('=')[0]:header_parts[1].split('=')[1]}
                # remove 1500_ from the contig name (or 2gen_)
                contig_name = contig_info['contig_name'][5:]
                if prefix is not None:
                    contig_name = prefix + contig_name
                current_contig = {
                    'length': int(contig_info['len']),
                    'sequence': '',
                    'coverages': []
                }
            # Check for sequence line
            elif current_contig is not None and current_contig['sequence'] == '':
                current_contig['sequence'] = line
            # Check for coverage lines 
            elif current_contig is not None and len(current_contig['coverages']) < num_genomes-1:
                current_contig['coverages'].append(list(map(int, line.split(','))))
            elif current_contig is not None and len(current_contig['coverages']) == num_genomes-1:
                current_contig['coverages'].append(list(map(int, line.split(','))))
                current_contig['coverages'] = np.array(current_contig['coverages'])
        if current_contig is not None:
            contigs[contig_name] = current_contig
    return contigs


#### ENCODING ####
def sequence_encoding(sequence):
    """
    Given a sequence, returns a one-hot encoding of it as a PyTorch tensor.

    Parameters:
    sequence (str): Input nucleotide sequence.

    Returns:
    torch.Tensor: A 4xN tensor representing the one-hot encoding of the input sequence.
    """
    encoding = {
        'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
        'T': torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        'M': torch.tensor([1, 1, 0, 0], dtype=torch.float32) / 2,  # A or C
        'R': torch.tensor([1, 0, 1, 0], dtype=torch.float32) / 2,  # A or G
        'W': torch.tensor([1, 0, 0, 1], dtype=torch.float32) / 2,  # A or T
        'S': torch.tensor([0, 1, 1, 0], dtype=torch.float32) / 2,  # G or C
        'Y': torch.tensor([0, 1, 0, 1], dtype=torch.float32) / 2,  # C or T    
        'K': torch.tensor([0, 0, 1, 1], dtype=torch.float32) / 2,  # G or T
        'V': torch.tensor([1, 1, 1, 0], dtype=torch.float32) / 3,  # A, C, or G
        'H': torch.tensor([1, 1, 0, 1], dtype=torch.float32) / 3,  # A, C, or T
        'D': torch.tensor([1, 0, 1, 1], dtype=torch.float32) / 3,  # A, G, or T   
        'B': torch.tensor([0, 1, 1, 1], dtype=torch.float32) / 3,  # C, G, or T 
        'N': torch.tensor([1, 1, 1, 1], dtype=torch.float32) / 4   # Any nucleotide
    }
    encoded_sequence = torch.zeros(4, len(sequence), dtype=torch.float32)
    for i, char in enumerate(sequence):
        if char in encoding:
            encoded_sequence[:, i] = encoding[char]
        else:
            raise ValueError(f"Unknown character: {char}")
    return encoded_sequence


#### LABELS ####
def filter_coverage(coverage, low_t):
    """
    Filters the coverage sequence and returns an array with:
    - 0 if the coverage is below low_t
    - 1 otherwise.

    Parameters:
    coverage (np.array): An array or list representing the coverage sequence.
    low_t (int)
    high_t (int)

    Returns
    np.array: An array of 0s and 1s with the same length as 'coverage'.
    """
    result = np.zeros_like(coverage)
    result[coverage >= low_t] = 1
    return result


def remove_oscillations(labels, window_size):
    """
    Removes oscillations in the labels by replacing each value with the majority of its neighbors
    within a window of size `window_size`.

    Parameters:
    labels (np.array or list): An array or list of labels (0 or 1).
    window_size (int): The size of the smoothing window (must be an odd number).

    Returns:
    np.array: An array of smoothed labels with the same length as 'labels'.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")
    # Extend the vector with boundary values to avoid errors at the edges
    extended_labels = np.pad(labels, (window_size // 2, window_size // 2), mode='constant', constant_values=0)
    smoothed_labels = np.copy(labels)
    for i in range(len(labels)):
        # Extract the window centered at 'i'
        local_window = extended_labels[i:i + window_size]
        # Calculate the majority in the window (0 or 1)
        majority = np.round(np.mean(local_window)).astype(int)
        # Assign the majority to the current point
        smoothed_labels[i] = majority    
    return smoothed_labels


def remove_short_sequences(labels, min_length):
    """
    Removes sequences of 1s that are shorter than `min_length` by setting them to 0.

    Parameters:
    labels (np.array or list): An array or list of labels (0 or 1).
    min_length (int): The minimum length required for a sequence of 1s to be retained.

    Returns:
    np.array: An array of labels where short sequences of 1s have been replaced with 0s.
    """
    in_sequence = False
    current_length = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            if not in_sequence:
                in_sequence = True
                current_length = 1
            else:
                current_length += 1
        else:
            if in_sequence:
                if current_length < min_length:
                    labels[i - current_length:i] = 0
                in_sequence = False
                current_length = 0
    if in_sequence and current_length < min_length:
        labels[len(labels) - current_length:] = 0
    return labels


# 1)
def binary_positive_mean(coverages, thr):
    """
    Returns 1 if the mean of positive coverage values is greater than the threshold, otherwise 0.

    Parameters:
    covereages (np.array): np.array containg coverages for all genomes.
    thr (float): The threshold above which the class is set to 1.

    Returns:
    torch.tensor: binary classes (either 0 or 1)
    """
    results = []
    for i in range(coverages.shape[1]):
        position_coverage = coverages[:, i]
        positives = np.array(position_coverage[position_coverage > 0])
        if len(positives) == 0:
            pos_mean = 0.0
        else:
            pos_mean = np.mean(positives)
        results.append(1 if pos_mean > thr else 0)
    return torch.tensor(results, dtype=torch.float32)


# 2)
def merge_labels(labels):
    """
    Merges overlapping or adjacent sequences of '1' values in the input label array.

    This function processes a binary matrix (labels), where each row represents a genome's
    label sequence over a contig. It identifies continuous sequences of '1' values (representing
    selected regions) in each genome, merges overlapping or adjacent sequences, and then returns
    a new label array representing the merged sequences.

    Parameters:
    labels (numpy.ndarray): A 2D binary array where rows represent genomes, and columns represent positions in a contig. Each element is either 0 (not part of a sequence) or 1 (part of a sequence).

    Returns:
    numpy.ndarray: A 1D binary array where merged sequences of '1' values are represented in a single sequence, with overlapping or adjacent sequences combined into one.
    
    Example:
    labels = np.array([[0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1]])
    merged_labels = merge_labels(labels)
    # merged_labels will represent the merged sequences from all genomes.
    """
    num_genomes = labels.shape[0]
    len_contig = labels.shape[1]
    all_sequences = []
    for k in range(num_genomes):
        in_sequence = False
        for i in range(len_contig):
            if labels[k, i] == 1 and not in_sequence:
                start = i
                in_sequence = True
            if labels[k, i] == 0 and in_sequence:
                end = i
                in_sequence = False
                all_sequences.append((start, end))
    # merge sequences
    merged_sequences = []
    # order sequences by length
    all_sequences.sort(key=lambda x: x[1] - x[0], reverse=True)
    # starting from the longest, remove the overlapping sequences
    while all_sequences:
        start, end = all_sequences.pop(0)  # Take the first sequence from the list
        # Remove the sequences that are inside the current one
        all_sequences = [(s, e) for s, e in all_sequences if e < start or s > end]
        # Add the current sequence to the merged sequences
        merged_sequences.append((start, end))
    # now convert the merged sequences to a label array
    merged_labels = np.zeros(len_contig)
    for start, end in merged_sequences:
        merged_labels[start:end] = 1
    return merged_labels


#### TRAINING, VALIDATION, TEST ####
def create_coordinates(chimera, N, training_fraction):
    """
    Creates a list of coordinates to sample from the chimera dictionary. Specifically: a training set composed by Ntraining_fraction coordinates; a validation and a test set composed by N(1-training_fraction)/2 each. The samples in the last 2 sets are sampled so that each sample is far away at least 500 postions from each training sample.

    Parameters:
    chimera (dictionary): A dictionary containing the chimera file.
    N (int): The number of coordinates to sample.
    training_fraction (float): The percentage of coordinates to use for training.

    Returns:
    list: 3 lists of coordinates to sample. (training, validation and test)
    """
    # total number of positions
    total_length = seqcov_length(chimera)
    if N > total_length:
        raise ValueError('N must be less or equal to the total number of positions')
    if training_fraction > 1:
        raise ValueError('training_fraction must be less or equal to 1')
    if training_fraction <= 0:
        raise ValueError('training_fraction must be greater than 0')
    # select N*training_fraction random positions to sample for training
    training_positions = np.random.choice(total_length, int(N*training_fraction), replace=False)
    # for each postition, create a tuple (contig, position)
    training_coordinates = []
    for pos in training_positions:
        length = 0
        for contig in chimera:
            length += chimera[contig]['length']
            if pos < length:
                pos = pos - (length - chimera[contig]['length'])
                training_coordinates.append((contig, pos))
                break
    val_test_coordinates = []
    if training_fraction != 1:
        # build the list of positions from which we can sample the validation and test positions
        excluded_positions = set()
        for pos in training_positions:
            excluded_positions.update(range(max(0, pos - 500), min(total_length, pos + 500 + 1)))
        all_positions = set(range(total_length))  
        remaining_positions = list(all_positions - excluded_positions)   
        # select N*(1-training_fraction) for validation and test
        val_test_positions = np.random.choice(remaining_positions, N-int(N*training_fraction), replace=False)
        # for each postition, create a tuple (contig, position)
        for pos in val_test_positions:
            length = 0
            for contig in chimera:
                length += chimera[contig]['length']
                if pos < length:
                    pos = pos - (length - chimera[contig]['length'])
                    val_test_coordinates.append((contig, pos))
                    break
        # separate validation from test
        half = (N-int(N*training_fraction))//2
        val_coordinates = val_test_coordinates[:half]
        test_coordinates = val_test_coordinates[half:]
    return training_coordinates, val_coordinates, test_coordinates


def create_mega_dict(sgb_chim_dict, labels_type, **kwargs):
    """
    Creates a dictionary containing all the chimeras in the input dictionary.

    Parameters:
    sgb_chim_dict (dict): A dictionary where keys are SGB (Species Genome Bin) identifiers (e.g., 'SGB1') 
                           and values are lists of chimeras (e.g., [chim1, chim2, ...]) associated with each SGB.
    labels_type (str): Kind of label ('BinaryPositiveMean' or 'EuristicLabels')
    **kwargs: create_coordinates parameters

    Returns:
    dict: A dictionary containing all chimeras, where each chimera is represented by its 
          respective contig (key), and associated data (value) as a dictionary with keys:
          - 'length' (int): The length of the chimera sequence.
          - 'sequence' (str): The nucleotide sequence of the chimera.
          - 'coverages' (np.array): The coverage values for each position in the chimera.
    create_coordinates outputs
    """
    # create paths
    paths = create_chimera_paths(sgb_chim_dict)
    num_genomes = len(sgb_chim_dict[list(sgb_chim_dict.keys())[0]][0].split('_'))
    # mega_dict
    chimeras = {}
    # sets
    train_c = []
    val_c = []
    test_c = []
    for i, path in enumerate(paths):
        # import file
        chim = import_seqcov_bzip_file(path, prefix=f'{i}')
        print(f'File {path} imported.')
        for contig_name in chim:
            contig = chim[contig_name]
            if labels_type == 'EuristicLabels':
                all_labels = []
                for j in range(num_genomes):
                    coverage = contig['coverages'][j,:]
                    labels_per_genome = make_labels(coverage, labels_type)
                    all_labels.append(labels_per_genome)    
                all_labels = np.array(all_labels)
                chim[contig_name]['labels'] = merge_labels(all_labels)
            elif labels_type == 'BinaryPositiveMean':
                coverages = contig['coverages']
                labels = make_labels(coverages, labels_type)   
                chim[contig_name]['labels'] = labels
        print('Labels created.')  
        train_c_chim, val_c_chim, test_c_chim = create_coordinates(chim, **kwargs)
        print('Coordinates created.')
        train_c.extend(train_c_chim)
        val_c.extend(val_c_chim)
        test_c.extend(test_c_chim)
        chimeras.update(chim)
    # empty the dictionary
    chim = None
    return chimeras, train_c, val_c, test_c


def check_labels_distribution(mega_dict, coordinates):
    labels = 0
    for contig, pos in coordinates:
        labels += mega_dict[contig]['labels'][pos]
    return labels/len(coordinates)


#### DATA GENERATOR ####
class DataGenerator(Dataset):
    """
    Data generator for training a model, generating batches of data based on a set of coordinates and a window size.

    Parameters:
    mega_dict (dict): Dictionary containing the data for each contig. The dictionary should contain:
                      - 'sequence': nucleotide sequence for each contig.
                      - 'features': optional additional features for each contig (2D array).
                      - 'coverages': coverage values for each contig.
    coordinates (list of tuples): List of tuples (contig, position) where the contig refers to the contig name 
                                  and the position refers to the index within that contig to sample data from.
    batch_size (int): Number of samples to return per batch.
    window_size (int): Size of the window to extract around each position. Must be an odd number.
    shuffle (bool): Whether to shuffle the order of coordinates at the end of each epoch. Defaults to True.

    Raises:
    ValueError: If `window_size` is not an odd number.
    """
    def __init__(self, mega_dict, coordinates, window_size): 
        self.mega_dict = mega_dict
        self.coordinates = coordinates
        self.window_size = window_size
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number.")
            
    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        # Fetch coordinates using the shuffled index
        contig, pos = self.coordinates[index]
        X, y = self.__data_generation(contig, pos)
        return X, y

    def __data_generation(self, contig, pos):
        # Padding if necessary
        if pos < self.window_size // 2:
            pad = self.window_size//2 - pos
            window = self.mega_dict[contig]['sequence'][:pos + self.window_size // 2 + 1]
            window = 'N'*pad + window
            window_labels = self.mega_dict[contig]['labels'][:pos + self.window_size // 2 + 1]
            window_labels = np.pad(window_labels, (pad, 0), 'constant', constant_values=0) 
        elif pos > self.mega_dict[contig]['length'] - self.window_size // 2 - 1:
            pad = self.window_size // 2 - (self.mega_dict[contig]['length'] - pos - 1)   
            window = self.mega_dict[contig]['sequence'][pos - self.window_size // 2:]
            window = window + 'N'*pad
            window_labels = self.mega_dict[contig]['labels'][pos - self.window_size // 2:]
            window_labels = np.pad(window_labels, (0, pad), 'constant', constant_values=0)
        else:
            window = self.mega_dict[contig]['sequence'][pos - self.window_size//2:pos + self.window_size//2 + 1]
            window_labels = self.mega_dict[contig]['labels'][pos - self.window_size//2:pos + self.window_size//2 + 1]
        # encoding 
        X = sequence_encoding(window)
        # labels (with smoothing)
        y = torch.tensor([np.mean(window_labels)], dtype=torch.float32)
        return X, y


#### SUPPORT ####
def seqcov_length(seqcov):
    """
    Returns the length of the sequence in the input seqcov dictionary.

    Parameters:
    seqcov (dictionary): A dictionary containing the data for a contig.

    Returns:
    int: The length of the sequence.
    """
    # total number of positions
    tot_pos = 0
    for contig in seqcov:
        tot_pos += seqcov[contig]['length']
    return tot_pos


def make_labels(coverages, labels_type):
    """
    Given the coverages of a contig, returns the sequence of labels.

    Parameters:
    coverages (np.array): np.array containg coverages for all genomes.
    labels_type (str): Kind of label ('BinaryPositiveMean' or 'EuristicLabels')

    Returns:
    np.array: Labels sequence
    """
    T = 80
    if labels_type == 'BinaryPositiveMean':
        labels = binary_positive_mean(coverages, thr=T)
        labels = remove_oscillations(labels, window_size=21)
        labels = remove_short_sequences(labels, min_length=300)
    elif labels_type == 'EuristicLabels':
        labels = filter_coverage(coverages, low_t=T)
        for i in range(2):
            labels[i,:] = remove_oscillations(labels[i,:], window_size=21)
            labels[i,:] = remove_short_sequences(labels[i,:], min_length=300)
    else:
        raise ValueError('Invalid label type', labels_type)
    return labels


def create_data_loaders(label, window_size, batch_size):
    """
    Import mega_dict and coordinates and create generators and then loaders.
    """
    path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/'
    with open(path+f'{label}/train_mega_dict.json', 'r') as file:
        mega_dict = json.load(file)
    for contig in mega_dict:
        mega_dict[contig]['coverages'] = np.array(mega_dict[contig]['coverages'])
        mega_dict[contig]['labels'] = np.array(mega_dict[contig]['labels'])
    with open(path+'train_val_coordinates.txt', 'r') as file:
        for i in range(3):
            row = file.readline().strip()
            if i == 1: 
                train_c = row.split(';')
            elif i == 2:
                val_c = row.split(';')
    train_c = [ast.literal_eval(item) for item in train_c]
    val_c = [ast.literal_eval(item) for item in val_c]
    train_gen = DataGenerator(mega_dict, train_c, window_size=window_size)
    val_gen = DataGenerator(mega_dict, val_c, window_size=window_size)
    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


#### TEST SETS CREATION ####
def create_mega_dict_test(sgb_chim_dict, labels_type, **kwargs):
    """
    Creates a dictionary containing all the chimeras in the input dictionary.

    Parameters:
    sgb_chim_dict (dict): A dictionary where keys are SGB (Species Genome Bin) identifiers (e.g., 'SGB1') 
                           and values are lists of chimeras (e.g., [chim1, chim2, ...]) associated with each SGB.
    labels_type (str): Kind of label ('BinaryPositiveMean' or 'EuristicLabels')
    **kwargs: create_coordinates_one_set parameters

    Returns:
    dict: A dictionary containing all chimeras, where each chimera is represented by its 
          respective contig (key), and associated data (value) as a dictionary with keys:
          - 'length' (int): The length of the chimera sequence.
          - 'sequence' (str): The nucleotide sequence of the chimera.
          - 'coverages' (np.array): The coverage values for each position in the chimera.
    list: create_coordinates_one_set output
    """
    paths = create_chimera_paths(sgb_chim_dict)
    num_genomes = len(sgb_chim_dict[list(sgb_chim_dict.keys())[0]][0].split('_'))
    chimeras = {}
    coordinates = []
    for i, path in enumerate(paths):
        chim = import_seqcov_bzip_file(path, prefix=f'{i}')
        print(f'File {path} imported.')
        for contig_name in chim:
            contig = chim[contig_name]
            if labels_type == 'EuristicLabels':
                all_labels = []
                for j in range(num_genomes):
                    coverage = contig['coverages'][j,:]
                    labels_per_genome = make_labels(coverage, labels_type)
                    all_labels.append(labels_per_genome)    
                all_labels = np.array(all_labels)
                chim[contig_name]['labels'] = merge_labels(all_labels)
            elif labels_type == 'BinaryPositiveMean':
                coverages = contig['coverages']
                labels = make_labels(coverages, labels_type)   
                chim[contig_name]['labels'] = labels
        print('Labels created.')  
        coordinates_chim = create_coordinates_one_set(chim, **kwargs)
        print('Coordinates created.')
        coordinates.extend(coordinates_chim)
        chimeras.update(chim)
    chim = None
    return chimeras, coordinates


def create_coordinates_one_set(chimera, N):
    """
    Creates only one set to sample from the chimera dictionary.

    Parameters:
    chimera (dictionary): A dictionary containing the chimera file.
    N (int): The number of coordinates to sample.

    Returns:
    list: A list of coordinates to sample. 
    """
    total_length = seqcov_length(chimera)
    if N > total_length:
        raise ValueError('N must be less or equal to the total number of positions')
    positions = np.random.choice(total_length, int(N), replace=False)
    coordinates = []
    for pos in positions:
        length = 0
        for contig in chimera:
            length += chimera[contig]['length']
            if pos < length:
                pos = pos - (length - chimera[contig]['length'])
                coordinates.append((contig, pos))
                break
    return coordinates


### TEST 2 ###
def test_2_create_sgb_chim_dict():
    """
    Create the sgb_chim_dict for test2 (chimeras with different genomes but same SGBs of training).
    """
    # import sgb_chim_dict
    dict_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/train_sgb_chim_dict.json'
    with open(dict_path, 'r') as file:
        sgb_chim_dict = json.load(file)
    # list of genomes used in training for each SGB
    genomes_used = {}
    for sgb in sgb_chim_dict:
        chimeras = sgb_chim_dict[sgb]
        genomes = []
        for chim in chimeras:
            genomes.extend(chim.split('_'))
        genomes_used[sgb] = set(genomes)
    # chimeras not used
    base_path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/data/chimeras_2genomes'
    chims_not_used = {}
    for sgb in sgb_chim_dict:
        all_chims = set(os.listdir(base_path+f'/{sgb}'))
        chims_used = set(sgb_chim_dict[sgb])
        chims_not_used[sgb] = list(all_chims - chims_used)   
    # esclude chimeras with used genomes
    test2_chims_dict = {}
    for sgb in chims_not_used:
        clean_chims = []
        for chim in chims_not_used[sgb]:
            gens = chim.split('_')
            if gens[0] in genomes_used[sgb] or gens[1] in genomes_used[sgb]:
                continue
            else:
                clean_chims.append(chim)
        test2_chims_dict[sgb] = clean_chims 
    # for each sgb extract from 0 to 2 chimeras
    test2 = {}
    for sgb in test2_chims_dict:
        if len(test2_chims_dict[sgb]) >= 2:
            test2[sgb] = test2_chims_dict[sgb][:2]
        elif len(test2_chims_dict[sgb]) == 1:
            test2[sgb] = [test2_chims_dict[sgb][0]]
        else:
            continue
    count = 0
    for sgb in test2:
        count += len(test2[sgb])
    print(f'Test 2 contains {count} chimeras.')
    # save test 2 sgb_chim dictionary
    data_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/'
    file_path = data_path + 'test_2_sgb_chim_dict.json'
    with open(file_path, "w") as file:
        json.dump(test2, file)
    print('test_2_sgb_chim_dict.json saved.')
    return test2


### TEST 3 ###
def test_3_create_sgb_chim_dict():
    """
    Create the sgb_chim_dict for test3 (chimeras within different SGBs w.r.t. the training ones).
    """
    dict_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/train_sgb_chim_dict.json'
    with open(dict_path, 'r') as file:
        sgb_chim_dict = json.load(file)  
    base_path = f'/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/data/chimeras_2genomes'
    all_sgb = set(os.listdir(base_path))
    used_sgb = set(sgb_chim_dict.keys())
    sgb_not_used = list(all_sgb-used_sgb)
    sgbs = random.sample(sgb_not_used, 26)
    test3 = {}
    for sgb in sgbs:
        test3[sgb] = random.sample(os.listdir(base_path+f'/{sgb}'), 5)
        count = 0
    for sgb in test3:
        count += len(test3[sgb])
    print(f'Test 3 contains {count} chimeras.')
    # save test 3 sgb_chim dictionary
    data_path = '/shares/CIBIO-Storage/CM/scratch/users/marco.chiloiro/thesis/models/final_version/training/data/'
    file_path = data_path + 'test_3_sgb_chim_dict.json'
    with open(file_path, "w") as file:
        json.dump(test3, file)
    print('test_3_sgb_chim_dict.json saved.')
    return test3