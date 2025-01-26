import torch

def load_and_preprocess_data(batch_size=32, seq_len=10, input_dim=768):
    """
    Simulates loading and preprocessing of data for the LearnableGatedPooling model.
    
    Args:
        batch_size (int): Number of samples in a batch
        seq_len (int): Length of the input sequence
        input_dim (int): Dimension of each vector in the sequence
        
    Returns:
        tuple: (embeddings, labels)
            - embeddings: torch.Tensor of shape (batch_size, seq_len, input_dim)
            - labels: torch.Tensor of shape (batch_size,) with binary values
    """
    # Placeholder that simulates loading data
    # In a real scenario, load data from `data/` directory and preprocess as needed
    embeddings = torch.randn(batch_size, seq_len, input_dim)
    labels = torch.randint(0, 2, (batch_size,))
    return embeddings, labels
