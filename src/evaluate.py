import torch
from src.preprocess import load_and_preprocess_data
from src.model_components import LearnableGatedPooling

def evaluate_model(pooling_layer, classifier):
    """
    Evaluates the trained model using accuracy metric.
    
    Args:
        pooling_layer: Trained LearnableGatedPooling instance
        classifier: Trained classifier layer
        
    Returns:
        float: Accuracy of the model on the evaluation data
    """
    embeddings, labels = load_and_preprocess_data()
    with torch.no_grad():
        pooled_output = pooling_layer(embeddings)
        logits = classifier(pooled_output)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
    return accuracy.item()
