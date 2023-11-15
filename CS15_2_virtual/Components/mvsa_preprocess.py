import pandas as pd


# retrieve a list of labels and generate a dataframe
def read_labels_file(path):
    dataframe = pd.read_csv(path, sep="\s+|,", engine="python")
    return dataframe

def merge_multi_label(dataframe):
    anno_1 = list(dataframe.loc[:, ['text', 'image']].itertuples(index=False, name=None))
    anno_2 = list(dataframe.loc[:, ['text.1', 'image.1']].itertuples(index=False, name=None))
    anno_3 = list(dataframe.loc[:, ['text.2', 'image.2']].itertuples(index=False, name=None))
    IDs = list(dataframe.iloc[:, 0])

    valid_pairs = []

    for i in range(len(anno_1)):
        pairs = [anno_1[i], anno_2[i], anno_3[i]]
        ID = IDs[i]

        text_labels = [pair[0] for pair in pairs]
        image_labels = [pair[1] for pair in pairs]

        max_occur_text_label = max(text_labels, key=text_labels.count)
        max_occur_image_label = max(image_labels, key=image_labels.count)

        if text_labels.count(max_occur_text_label) > 1 and image_labels.count(max_occur_image_label) > 1:
            valid_pair = (ID, max_occur_text_label, max_occur_image_label)
        else:
            valid_pair = (ID, 'invalid', 'invalid')
        valid_pairs.append(valid_pair)
    valid_dataframe = pd.DataFrame(valid_pairs, columns=['ID', 'text', 'image'])
    return valid_dataframe

def multimodal_label(text_label, image_label):
    if text_label == image_label:
        label = text_label
    elif (text_label == 'positive' and image_label == 'negative') or (text_label == 'negative' and image_label == 'positive'):
        label = 'invalid'
    elif (text_label == 'neutral' and image_label != 'neutral') or (text_label != 'neutral' or image_label == 'neutral'):
        label = image_label if text_label == 'neutral' else text_label
    return label

# retrieve the text content
def get_text_content(path):
  return open(path, 'r', encoding='utf-8', errors='replace').read().rstrip('\n')

def get_text_ocr(path, ID):
  text_file_path = f"{path}/{ID}.txt"
  try:
      text_ocr = get_text_content(text_file_path)
  except FileNotFoundError:
      text_ocr = None
  return text_ocr

# Add columns
def generate_multi_label(label_path, data_path):
    labels_df = read_labels_file(label_path)

    labels_df['image_name'] = labels_df['ID'].apply(lambda x: f"{x}.jpg")
    labels_df['text_corrected'] = labels_df.apply(lambda row: get_text_ocr(data_path, row['ID']), axis=1)
    # labels_df['text_corrected'] = labels_df['ID'].apply(get_text_ocr)

    merged_df = merge_multi_label(labels_df)
    labels_df['text_sentiment'] = merged_df['text']
    labels_df['image_sentiment'] = merged_df['image']


    overall_sentiments = [multimodal_label(row['text_sentiment'], row['image_sentiment']) for index, row in labels_df.iterrows()]
    labels_df['overall_sentiment'] = overall_sentiments

    return labels_df


def process_multi_label(label_path, data_path):
    labels = generate_multi_label(label_path, data_path)

    filtered_labels = labels[labels['overall_sentiment'] != 'invalid']

    return filtered_labels

def generate_single_label(label_path, data_path):
    labels_df = read_labels_file(label_path)

    labels_df['image_name'] = labels_df['ID'].apply(lambda x: f"{x}.jpg")
    labels_df['text_corrected'] = labels_df.apply(lambda row: get_text_ocr(data_path, row['ID']), axis=1)
    # labels_df['text_corrected'] = labels_df['ID'].apply(get_text_ocr)

    labels_df['text_sentiment'] = labels_df['text']
    labels_df['image_sentiment'] = labels_df['image']

    overall_sentiments = [multimodal_label(row['text_sentiment'], row['image_sentiment']) for index, row in labels_df.iterrows()]
    labels_df['overall_sentiment'] = overall_sentiments

    return labels_df

def process_single_label(label_path, data_path):
    labels = generate_single_label(label_path, data_path)

    filtered_labels = labels[labels['overall_sentiment'] != 'invalid']

    return filtered_labels
