import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from models import LearnableGatedPooling
from train import train_model
from evaluate import evaluate_model
from preprocess import prepare_data

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        args.data_path,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = LearnableGatedPooling(
        input_dim=args.input_dim,
        seq_len=args.seq_len
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate LearnableGatedPooling model')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path to data directory')
    parser.add_argument('--input_dim', type=int, default=768,
                        help='input dimension size')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    
    args = parser.parse_args()
    main(args)
