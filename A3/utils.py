def calculate_accuracy(outputs, targets):
    predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
    correct_predictions = (predictions == targets).float()
    accuracy = correct_predictions.sum().item() / targets.shape[1]
    return accuracy