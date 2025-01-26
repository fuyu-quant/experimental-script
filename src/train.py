import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocess import load_and_preprocess_data
from src.model_components import LearnableGatedPooling

def train_model():
    """
    Trains the LearnableGatedPooling model with a simple classifier.
    Returns the trained model components.
    """

    # Hyperparameters
    input_dim = 768
    learning_rate = 1e-3
    num_epochs = 2

    # Prepare data
    embeddings, labels = load_and_preprocess_data()

    # Define model
    pooling_layer = LearnableGatedPooling(input_dim)
    classifier = nn.Linear(input_dim, 2)

    # Optimizer
    optimizer = optim.Adam(list(pooling_layer.parameters()) + list(classifier.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        pooled_output = pooling_layer(embeddings)
        logits = classifier(pooled_output)
        loss = loss_fn(logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    return pooling_layer, classifier
