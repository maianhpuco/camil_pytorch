import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

class CustomDataset(Dataset):
    def __init__(
        self, 
        train_or_test_or_val,
        split_filepath, 
        label_file,
        feature_folder, 
        feature_file_end ='h5',  
        shuffle=True, 
        dry_run=False, 
    ):
        """
        Args:
            label_file (str): Path to the CSV file containing labels for each sample.
            shuffle (bool): Whether to shuffle the data or not. Default is False.
        """
        assert train_or_test_or_val in ['train','test', 'val'], f"Invalid argument: {train_or_test_or_val}. Must be 'train', 'val', or 'tstv'."
        self.train_or_test_or_val = train_or_test_or_val 
        self.feature_folder = feature_folder
        self.split_filepath = split_filepath 
        self.label_file = label_file
        self.shuffle = shuffle
        self.feature_file_end = feature_file_end
        
        # Load labels from CSV file
        # self.labels_df = pd.read_csv(label_file)
        
        # Optionally shuffle  if required
        # self.indices = np.arange(len(self.file_paths))
        df = pd.read_csv(self.split_filepath)
        df.dropna(subset=[self.train_or_test_or_val, f'{self.train_or_test_or_val}_label'], inplace=True)
  
        self.name_label_dict = df.set_index(self.train_or_test_or_val)[
            f'{self.train_or_test_or_val}_label'].to_dict()
        self.names = [k for k, v in self.name_label_dict.items()]
        if dry_run is True: 
            import random 
            self.names = random.sample(self.names, 10 )
        # print(self.names)
        self.indices = np.arange(len(self.names))
        
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def get_feature_path(self, basename):
        return os.path.join(self.feature_folder, f'{basename}.{self.feature_file_end}')
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.indices)

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: (feature, adjacency matrix, label)
        """
        # Get the index of the file
        file_idx = self.indices[index]
        file_basename = self.names[file_idx]
        file_path = self.get_feature_path(file_basename)
        # Load the HDF5 file
        with h5py.File(file_path,  "r") as f:
            features = f['features'][:]
            neighbor_indices = f['indices'][:]
            values = f['similarities'][:]
            values = np.nan_to_num(values)
            # base_name = os.path.splitext(os.path.basename(file_path))[0]
            label = self.name_label_dict[file_basename]
        # label_tensor = torch.tensor(label, dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32).view(1, 1)
        # print(label_tensor)
        # Process adjacency matrix
        idx = neighbor_indices[:, :8]
        rows = np.asarray([[enum] * len(item) for enum, item in enumerate(idx)]).ravel()
        columns = idx.ravel()

        neighbor_matrix = values[:, 1:]
        normalized_matrix = normalize(neighbor_matrix, norm="l2")

        similarities = np.exp(-normalized_matrix)
        values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)
        values = values[:, :8]
        values = values.ravel().tolist()
        sparse_coords = list(zip(rows, columns))

        # Create sparse adjacency matrix
        sparse_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(sparse_coords).t(),
            values=torch.tensor(values),
            size=(features.shape[0], features.shape[0])
        )
        

        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        # print("- features_tensor", features_tensor.shape)
        # print("- sparse_matrix", sparse_matrix.shape)         
        # print("- label", torch.tensor(label, dtype=torch.float32))
        # features_tensor = features_tensor.unsqueeze(0)  # Adding the batch dimension
        # sparse_matrix = sparse_matrix.unsqueeze(0)  # Adding the batch dimension to sparse matrix
 
        return features_tensor, sparse_matrix, label_tensor

