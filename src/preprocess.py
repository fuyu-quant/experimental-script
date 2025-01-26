import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    """
    Dataset class for handling sequence data
    """
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def prepare_data(data_path, batch_size=32):
    """
    Prepare data loaders for training, validation, and testing
    """
    # Load and preprocess data
    # Note: Implement actual data loading logic based on your data format
    
    # Example data creation (replace with actual data loading)
    def create_dummy_data(num_samples, seq_len, input_dim):
        half_samples = num_samples // 2
        sequences_0 = np.random.randn(half_samples, seq_len, input_dim)
        labels_0 = np.zeros(half_samples, dtype=int)
        sequences_1 = np.random.randn(num_samples - half_samples, seq_len, input_dim)
        labels_1 = np.ones(num_samples - half_samples, dtype=int)
        
        sequences = np.concatenate([sequences_0, sequences_1], axis=0)
        labels = np.concatenate([labels_0, labels_1], axis=0)
        return sequences, labels
    
    # Create datasets
    train_sequences, train_labels = create_dummy_data(1000, 10, 768)
    val_sequences, val_labels = create_dummy_data(200, 10, 768)
    test_sequences, test_labels = create_dummy_data(200, 10, 768)
    
    # Create dataset objects
    train_dataset = SequenceDataset(train_sequences, train_labels)
    val_dataset = SequenceDataset(val_sequences, val_labels)
    test_dataset = SequenceDataset(test_sequences, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
