import torch.nn as nn
import torch
import os
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from CS15_2_virtual.Components.image_processing import Ensure3Channels, ConvertToRGBA


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, image_dir, mode='train'):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_len = max_len
        self.image_dir = image_dir
        self.sentiment = ["positive", "negative", "neutral"]  # List of classes
        self.transform = self.data_transforms[mode]
        self.erroneous_images = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Handling the text data (BERT)
        comment_text = str(self.dataframe.iloc[idx, 2])
        comment_text = " ".join(comment_text.split())
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Handling the image data
        image_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        try:
            if not os.path.exists(image_name):
                raise FileNotFoundError(f"File does not exist: {image_name}")

            image = io.imread(image_name)

            if len(image.shape) == 4 and image.shape[0] == 1:
                image = image.squeeze(0)

            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # print(e)
            self.erroneous_images.append(image_name)
            # Create a placeholder image in case of an error
            image = torch.zeros((3, 224, 224))  # Assuming 3 channels and 224x224 size. Adjust if necessary

        label = self.dataframe.iloc[idx, 1]
        label_index = self.sentiment.index(label)
        label_tensor = torch.tensor(label_index, dtype=torch.long)
        # label = self.dataframe.iloc[idx, 1]
        # label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': label_tensor,
            'image': image
        }

    # Image Pre-Processing
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            ConvertToRGBA(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Ensure3Channels(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            ConvertToRGBA(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            Ensure3Channels(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            ConvertToRGBA(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            Ensure3Channels(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }