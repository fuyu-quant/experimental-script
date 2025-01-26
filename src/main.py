import torch
import torch.nn as nn
import torch.optim as optim
from models import LearnableGatedPooling
from train import train_model
from evaluate import evaluate_model
from preprocess import prepare_data
import yaml
import os

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        data_path=config['data_path'],
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = LearnableGatedPooling(
        input_dim=config['input_dim'],
        seq_len=config['seq_len']
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device
    )
    
    # Evaluate model
    evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

if __name__ == '__main__':
    main()
