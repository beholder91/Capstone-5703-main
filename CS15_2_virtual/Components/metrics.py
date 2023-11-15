import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
# import logging_config

# Define a function to calculate F1 score
def calculate_f1(outputs, labels, average):
    # Convert outputs to predicted classes
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, predicted, average=average)


def print_classification_report(outputs, labels):
    # Convert outputs to predicted classes
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    # Print the confusion matrix
    conf_matrix = confusion_matrix(labels, predicted)
    # print("Confusion Matrix:")
    # print(conf_matrix)

    # Print the classification report
    report = classification_report(labels, predicted, target_names=["positive", "negative", "neutral"])
    # logging_config.logger.info(report)
    # print(report)
    return report, conf_matrix
