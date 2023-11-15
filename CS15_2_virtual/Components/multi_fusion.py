import torch
import torch.nn.functional as F
import transformers
import torch.nn as nn
from CS15_2_virtual.Components.models import BERTClass, DINOv2

class SoftmaxAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SoftmaxAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attention_weights = torch.nn.functional.softmax(q @ k.transpose(-1, -2), dim=-1)
        attention_output = attention_weights @ v

        return attention_output

class MultimodalFusion(nn.Module):
    def __init__(self, dropout=0.2, freeze_bert=True, freeze_dinov2=True):
        super(MultimodalFusion, self).__init__()
        self.bert_model = BERTClass()
        self.dinov2_model = DINOv2()
        self.att_image = SoftmaxAttention(256, 32)  # Softmax attention layer for image features
        self.att_text = SoftmaxAttention(256, 32)   # Softmax attention layer for text features
        self.self_attention = nn.MultiheadAttention(embed_dim=32, num_heads=2)  # Self-attention layer

        # If freeze_bert is set to True, freeze all BERT model parameters
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            self.bert_model.eval()

        #######################################################################

        # If freeze_dinov2 is set to True, freeze all DINOv2 model parameters
        if freeze_dinov2:
            for param in self.dinov2_model.parameters():
                param.requires_grad = False
            self.dinov2_model.eval()
        ##############################################################################

        # self.fc = torch.nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(32, 3)
        # )

        ####################################### add BN in fc layer ####################################################
        self.fc = torch.nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),  # BatchNorm after Linear layer
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),  # BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 3)
        )

    def forward(self, text_data, image_data):
        # Step 1: Obtain features from both DINOv2 and BERT
        image_features = self.dinov2_model(image_data)
        text_features = self.bert_model(*text_data)

        # Ensuring the dimensionality is the same (if not, we need to adapt it)
        if image_features.size(-1) != text_features.size(-1):
            image_features = F.adaptive_avg_pool1d(image_features.unsqueeze(0), text_features.size(-1)).squeeze(0)

        # Constructing attention layers, using attention layers to process the features parallelly
        intermediate_image = self.att_image(image_features, text_features)
        intermediate_text = self.att_text(text_features, image_features)

        # Concatenating features and including cosine similarity in the self-attention layer
        concatenated_features = torch.cat((intermediate_text.unsqueeze(0), intermediate_image.unsqueeze(0)), dim=0)
        output, _ = self.self_attention(concatenated_features, concatenated_features, concatenated_features)

        # Reshape the output of the long attention to the full connection layer
        output = output.transpose(0, 1).contiguous().view(-1, 64)
        output = self.fc(output)

        return output