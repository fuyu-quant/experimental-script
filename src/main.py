import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from models import LearnableGatedPooling
from preprocess import prepare_data
from train import train_model
from evaluate import evaluate_model

def main():
    # Configuration
    input_dim = 768  # Example: BERT embedding dimension
    batch_size = 32
    seq_len = 10
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = LearnableGatedPooling(input_dim=input_dim, seq_len=seq_len)
    
    # Example data (replace with your actual data loading)
    dummy_sequences = [torch.randn(seq_len, input_dim) for _ in range(100)]
    
    # Preprocess data
    processed_data, max_seq_len = prepare_data(dummy_sequences, batch_size)
    
    # Create dummy targets (replace with your actual targets)
    dummy_targets = torch.randn(100, input_dim)
    
    # Create data loaders
    dataset = torch.utils.data.TensorDataset(processed_data, dummy_targets)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train model
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    print("\nTraining completed!")
    print(f"Final test loss: {evaluation_results['test_loss']:.4f}")

if __name__ == "__main__":
    main()
