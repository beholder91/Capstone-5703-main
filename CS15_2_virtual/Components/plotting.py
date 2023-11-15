import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_learning_curve(history, folder_name):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plotting training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history['train_loss'], 'g-^', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'b-^', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_name, "training_validation_loss.png"))
    # plt.show()

    # Plotting training and validation F1 score
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history['train_f1'], 'r-^', label='Training F1')
    plt.plot(epochs, history['val_f1'], 'y-^', label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(folder_name, "training_validation_f1.png"))
    # plt.show()


def plot_conf_matrix(conf_matrix, folder_name):
    class_names = ["positive", "negative", "neutral"]
    plt.figure(figsize=(len(class_names), len(class_names)))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(folder_name, "confusion_matrix.png"))
    # plt.show()
    