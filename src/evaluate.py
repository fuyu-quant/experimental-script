import torch
from models import LearnableGatedPooling

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluation function for the LearnableGatedPooling model
    
    Args:
        model: Trained LearnableGatedPooling model instance
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on ('cuda' or 'cpu')
    """
    model = model.to(device)
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss
