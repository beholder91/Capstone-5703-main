import transformers
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from CS15_2_virtual.Components.text_processing import process_text
from CS15_2_virtual.Components.multimodaldataset import MultimodalDataset

# Task A: Negative and Very Negative => -1 Positive and Very Positive => 1 Neutral => 0
def convert_to_numeric(x):
    if x == 'negative' or x == 'very_negative':
        return "negative"
    elif x == 'positive' or x == 'very_positive':
        return "positive"
    else:
        return x

def memotion_preprocess(path):
    df = pd.read_csv(path)

    # keep only the columns we need
    df = df[['image_name', 'overall_sentiment', 'text_corrected']]

    # fill the missing values with empty strings
    df = df.fillna('')

    # apply the function to the text
    df['text_corrected'] = df['text_corrected'].apply(process_text)

    
    df['overall_sentiment'] = df['overall_sentiment'].apply(convert_to_numeric)

    return df 


def dataset_loader(path, df, batch_size=256, train_size=0.8, max_len=30):
    # First, split the dataset into train (80%) and temporary val_test (20%)
    train_dataset, val_test_dataset = np.split(df.sample(frac=1, random_state=200),
                                            [int(train_size*len(df))])

    # Then, split the temporary val_test set into validation (10%) and test set (10%)
    val_dataset, test_dataset = np.split(val_test_dataset, [int(0.5*len(val_test_dataset))])

    # Reset the index for all the datasets
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    # print("FULL Dataset: {}".format(df.shape))
    # print("TRAIN Dataset: {}".format(train_dataset.shape))
    # print("VAL Dataset: {}".format(val_dataset.shape))
    # print("TEST Dataset: {}".format(test_dataset.shape))

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create instances of the MultimodalDataset class for training, validation, and testing
    training_set = MultimodalDataset(dataframe=train_dataset, tokenizer=tokenizer, max_len=max_len, image_dir=path, mode='train')
    validation_set = MultimodalDataset(dataframe=val_dataset, tokenizer=tokenizer, max_len=max_len, image_dir=path, mode='val')
    testing_set = MultimodalDataset(dataframe=test_dataset, tokenizer=tokenizer, max_len=max_len, image_dir=path, mode='test')


    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader