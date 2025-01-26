from src.train import train_model
from src.evaluate import evaluate_model

def run_experiment():
    """
    Runs the complete experiment workflow:
    1. Trains the model using LearnableGatedPooling
    2. Evaluates the model and reports accuracy
    """
    # Train
    print("Starting training...")
    trained_pooling, trained_classifier = train_model()

    # Evaluate
    print("\nStarting evaluation...")
    accuracy = evaluate_model(trained_pooling, trained_classifier)
    print(f"Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_experiment()
