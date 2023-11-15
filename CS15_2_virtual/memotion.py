import transformers
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from skimage import io
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import classification_report
import logging
import time
import json

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


from CS15_2_virtual.Components.text_processing import process_text
from CS15_2_virtual.Components.metrics import calculate_f1, print_classification_report
from CS15_2_virtual.Components.plotting import plot_learning_curve, plot_conf_matrix
from CS15_2_virtual.Components.memotion_preprocess import memotion_preprocess, dataset_loader
from CS15_2_virtual.Components.multimodaldataset import MultimodalDataset
from CS15_2_virtual.Components.models import Concatate_MultimodalModel, Baseline_MultimodalModel
from CS15_2_virtual.Components.losses import FocalLoss
from CS15_2_virtual.Components.earlyStop import EarlyStopping
from CS15_2_virtual.Components.multi_fusion import MultimodalFusion
from CS15_2_virtual.configs.config import memo_config, mvsa_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed_all(2023)  # Set Seed

def setup_logger(filename):
    # Clear existing loggers and handlers
    logging.getLogger().handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

# Function to load existing histories from a JSON file
def load_histories(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and return an empty dictionary
        with open(file_path, 'w') as file:
            json.dump({}, file, indent=2)
        return {}

# Function to save a new history to a JSON file
def save_history(file_path, history_name, history_data):
    all_histories = load_histories(file_path)
    all_histories[history_name] = history_data

    with open(file_path, 'w') as file:
        json.dump(all_histories, file, indent=2)

# Function to load existing histories from a JSON file
def load_histories(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and return an empty dictionary
        with open(file_path, 'w') as file:
            json.dump({}, file, indent=2)
        return {}

# Define the train function with batch information output
def train(model, train_loader, val_loader, loss_function, optimizer, folder_name, plot_name, num_epochs=10, patience=10):
    early_stopping = EarlyStopping(model = model, filepath = f'{folder_name}/best_model.pth', patience=patience)

    # Dictionary to store the training and validation loss and accuracy for each epoch
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': []
    }

    for epoch in range(num_epochs):
        model.train()  # Setting the model to training mode
        total_loss = 0
        train_outputs_list = []
        train_labels_list = []

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Getting the input data and labels
            text_data = [batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device)]
            image_data = batch['image'].to(device)
            labels = batch['targets'].to(device)

            # Zeroing the gradients of the parameters
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(text_data, image_data)

            # Calculating the loss
            loss = loss_function(outputs, labels)

            # Backward propagation and optimization
            loss.backward()
            optimizer.step()

            # Accumulating the loss
            total_loss += loss.item()

            # Storing outputs and labels for F1 calculation
            train_outputs_list.append(outputs)
            train_labels_list.append(labels)

            # Printing the batch information
            if batch_idx % 12 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                print(f'\t \t ----- Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculating and printing the average loss and accuracy for each epoch
        avg_loss = total_loss / len(train_loader)
        train_f1 = calculate_f1(torch.cat(train_outputs_list), torch.cat(train_labels_list), 'macro')
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'\t \t ----- Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}')
        report, matrix = print_classification_report(torch.cat(train_outputs_list), torch.cat(train_labels_list))
        logging.info(f"Report: \n{report}")
        print(report)
        logging.info(f"Confusion matrix: \n{matrix}")

        # Validation loop
        model.eval()
        val_total_loss = 0
        val_outputs_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                text_data = [batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device)]
                image_data = batch['image'].to(device)
                labels = batch['targets'].to(device)

                outputs = model(text_data, image_data)
                loss = loss_function(outputs, labels)
                val_total_loss += loss.item()

                # Storing outputs and labels for F1 calculation
                val_outputs_list.append(outputs)
                val_labels_list.append(labels)

        val_avg_loss = val_total_loss / len(val_loader)
        val_f1 = calculate_f1(torch.cat(val_outputs_list), torch.cat(val_labels_list), 'macro')
        logging.info(f'Validation - Average Loss: {val_avg_loss:.4f}, Validation F1: {val_f1:.4f}')
        print(f'\t \t ----- Validation - Average Loss: {val_avg_loss:.4f}, Validation F1: {val_f1:.4f}')
        report, matrix = print_classification_report(torch.cat(val_outputs_list), torch.cat(val_labels_list))
        logging.info(f"Report: \n{report}")
        logging.info(f"Confusion matrix: \n{matrix}")

        # Storing the loss and accuracy values to plot the learning curve
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_avg_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Check early stopping
        early_stopping.check(val_f1)
        if early_stopping.stop:
            ### load the last checkpoint with the best model
            early_stopping.save_best_weights()
            print("\t \t ----- Early Stopping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.")
            logging.info("Early Stopping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.")
            logging.info(f"stop at epoch {epoch}")
            break

    # save_history(f'/content/all_histories.json', f'{folder_name}_history', history)
    save_history(f'output/memotion/all_histories.json', f'{plot_name}', history)
    # Return the history for plotting
    return history

def test(model, test_loader, loss_function, folder_name):
    # Setting the model to evaluation mode
    model.eval()
    total_loss = 0
    outputs_list = []
    true_labels_list = []

    with torch.no_grad():  # Disabling gradient calculation
        for batch_idx, batch in enumerate(test_loader):
            # Getting the input data and labels
            text_data = [batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device)]
            image_data = batch['image'].to(device)
            labels = batch['targets'].to(device)

            # Forward propagation
            outputs = model(text_data, image_data)

            # Calculating the loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            # Storing outputs and true labels
            outputs_list.append(outputs)
            true_labels_list.append(labels)

    # Calculating and printing the average loss
    avg_loss = total_loss / len(test_loader)
    logging.info(f'Test - Average Loss: {avg_loss:.4f}')

    # Flatten lists of outputs and labels
    outputs_flat = torch.cat(outputs_list)
    true_labels_flat = torch.cat(true_labels_list)

    # Calculating and printing the F1 score
    test_f1 = calculate_f1(outputs_flat, true_labels_flat, 'macro')
    logging.info(f'Test - F1 Score: {test_f1:.4f}')
    print(f'\t \t ----- Test - F1 Score: {test_f1:.4f}')

    # Generating and printing the classification report
    report, conf_matrix = print_classification_report(outputs_flat, true_labels_flat)
    logging.info(f"Report: \n{report}")
    logging.info(f"Confusion matrix: \n{conf_matrix}")

    print(f"report: {report}")
    print(f"conf_matric: {conf_matrix}")
    plot_conf_matrix(conf_matrix, folder_name)

def trainmodel(memotion_label_path, memotion_image_path, configs, modelname):
    max_len = configs['max_len']
    train_size = configs['train_size']
    BATCH_SIZE = configs['BATCH_SIZE']
    dropout_rate = configs['dropout_rate']
    learning_rate = configs['learning_rate']
    gamma = configs['gamma']
    momentum = configs['momentum']
    num_epochs = configs['num_epochs']
    patience = configs['patience']
    # modelname = configs['model']

    # Record the start time
    start_time = time.time()

    print("\t ++++ Create folders and log files")
    folder_name = f'output/memotion/{modelname}/drop-{dropout_rate} lr-{learning_rate} gamma-{gamma}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # log_filename = f"/content/{dropout_rate}_{learning_rate}_{gamma}.log"
    log_filename = f"output/memotion/{modelname}/{dropout_rate}_{learning_rate}_{gamma}.log"
    setup_logger(log_filename)
    logging.info(f"\n{modelname}\ndropout: {dropout_rate}\nlearning rate: {learning_rate}\ngamma: {gamma}")

    plot_name = f"memotion {modelname} dr-{dropout_rate} lr-{learning_rate} gamma-{gamma}"

    print("\t ++++ Preprocessing dataset ... ")
    df = memotion_preprocess(memotion_label_path)

    train_loader, val_loader, test_loader = dataset_loader(
        path=memotion_image_path, df=df, batch_size=BATCH_SIZE, train_size=train_size, max_len=max_len)

    print("\t ++++ Construct model ... ")
    # Creating a model instance
    print(f"\t ++++ Model {modelname}")
    if modelname == 'baseline':
        model = Baseline_MultimodalModel(dropout=dropout_rate, freeze_bert=False, freeze_dinov2=False)
    elif modelname == 'concate': 
        model = Concatate_MultimodalModel(dropout=dropout_rate, freeze_bert=False, freeze_dinov2=False)
    elif modelname == 'fusion':
        model = MultimodalFusion(dropout=dropout_rate, freeze_bert=False, freeze_dinov2=False)
    else: 
        print("No model select")
        return None
    
    # model = MultimodalFusion()
    model = model.to(device)  # Moving the model to GPU
    
    # Defining the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss(alpha=1, gamma=gamma, reduction='mean')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)

    print("\t ++++ Start to train ... ")
    # Starting the training process & save the history to plot
    history = train(model, train_loader, val_loader, loss_function, optimizer, num_epochs=num_epochs, patience=patience, folder_name=folder_name, plot_name=plot_name)

    # Plot
    print("\t ++++ Plot ... ")
    plot_learning_curve(history, folder_name)

    # test
    print("\t ++++ Testing ... ")
    test(model, test_loader, loss_function, folder_name)

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")
    print(f"\t ++++ Elapsed time: {elapsed_time} seconds")

def run_memotion():
    # memotion_label_path = '/content/memotion_dataset_7k/labels.csv'
    # image_dir = "/content/memotion_dataset_7k/images"
    memotion_label_path = 'CS15_2_virtual/DATASET/memotion_dataset_7k/labels.csv'
    image_dir = "CS15_2_virtual/DATASET/memotion_dataset_7k/images"


    # models = ['concate', 'fusion']
    models = ['baseline', 'concate', 'fusion']

    memotion_configs = {}

    for dropout_rate in memo_config['dropout_rates']:
        for learning_rate in memo_config['learning_rates']:
            for gamma in memo_config['gamma_values']:
                # for model in models:
                config_name = f'dropout_{dropout_rate}_lr_{learning_rate}_gamma_{gamma}'
                memotion_configs[config_name] = {
                    'max_len': memo_config['max_len'],  # or any other value depending on your preference
                    'train_size': memo_config['train_size'],
                    'BATCH_SIZE': memo_config['batch_size'], 
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'gamma': gamma,
                    'momentum': memo_config['momentum'], 
                    'num_epochs': memo_config['num_epochs'],
                    'patience': memo_config['patience'], 
                    # 'model': model, 
                }


    for config_name, config in memotion_configs.items():
        for model in models:
            print(f"\n============ Train memotion in model {model}, config here: {config_name} ============ ")
            try:
                trainmodel(memotion_label_path, image_dir, config, model)
            except Exception as e:
                print(f"Error in model {model}: {e}")
                continue  # Continue to the next model


    # Load histories from the JSON file
    all_histories = load_histories('output/memotion/all_histories.json')

    # Here, you can set up a loop to iterate over each keyword ('train_loss', 'val_loss', 'train_f1', 'val_f1')
    keywords = ['train_loss', 'val_loss', 'train_f1', 'val_f1']

    for keyword in keywords:
        plt.figure(figsize=(10, 6))
        
        # Iterate over each history
        for history_name, history in all_histories.items():
            values = history[keyword]
            epochs = np.arange(1, len(values) + 1)
            plt.plot(epochs, values, marker='^', label=history_name)

        plt.title(f'{keyword} Trend Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(keyword)
        plt.legend()
        plt.savefig(os.path.join('output/memotion', f"{keyword}.png"))
        # plt.show()
