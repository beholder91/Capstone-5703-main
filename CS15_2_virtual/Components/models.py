import torch
import torch.nn.functional as F
import transformers
import torch.nn as nn

# def self_attention(query, key, value):
#     attention_scores = torch.matmul(query, key.transpose(-2, -1))
#     attention_weights = F.softmax(attention_scores, dim=-1)
#     return torch.matmul(attention_weights, value)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')

        self.l2 = torch.nn.Sequential(
            torch.nn.Linear(768, 256)
        )

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.l2(output_1)
        return output

#DinoV2 Model Class Definition
class DINOv2(nn.Module):
    def __init__(self):
        super(DINOv2, self).__init__()
        # Load the pretrained ViT model
        self.vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # Adjust linear layer to fit the classification
        self.classifier = nn.Linear(384, 256)

    def forward(self, image):
        output = self.vits14(image)
        output = self.classifier(output)
        return output

class Concatate_MultimodalModel(nn.Module):
    def __init__(self, dropout=0.2, freeze_bert=True, freeze_dinov2=True):
        super(Concatate_MultimodalModel, self).__init__()

        # Initializing the BERT and DINOv2 model instances
        self.bert_model = BERTClass()
        self.dinov2_model = DINOv2()

        ##################################### Add MultiheadAttention layer #############################################
        self.multihead_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)  

        # If freeze_bert is set to True, freeze all BERT model parameters
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            self.bert_model.eval() #######################################################################

        # If freeze_dinov2 is set to True, freeze all DINOv2 model parameters
        if freeze_dinov2:
            for param in self.dinov2_model.parameters():
                param.requires_grad = False
            self.dinov2_model.eval() ##############################################################################
        # # KQV
        # self.query_layer = nn.Linear(512, 512)
        # self.key_layer = nn.Linear(512, 512)
        # self.value_layer = nn.Linear(512, 512)

        # Fully connected layer to combine the outputs of the BERT and DINOv2 models
        # self.fc = torch.nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(32, 3)
        # )
        ####################################### add BN in fc layer ####################################################
        self.fc = torch.nn.Sequential(
            nn.Linear(512, 256),
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
        # Forward pass for the BERT model
        bert_output = self.bert_model(*text_data)

        # Forward pass for the DINOv2 model
        dinov2_output = self.dinov2_model(image_data)

        # Combining the outputs of the BERT and DINOv2 models
        combined_output = torch.cat((bert_output, dinov2_output), dim=1)

        # # Generate KQV
        # query = self.query_layer(combined_output)
        # key = self.key_layer(combined_output)
        # value = self.value_layer(combined_output)

        # attention_output = self_attention(query, key, value)
        ######################################### add MultiheadAttention layer #########################################
        attention_output, _ = self.multihead_attn(combined_output, combined_output, combined_output)

        # Passing the attention output through the fully connected layer
        output = self.fc(attention_output)

        return output

class Baseline_MultimodalModel(nn.Module):
    # def __init__(self, dropout=0.2):
    def __init__(self, dropout=0.2, freeze_bert=True, freeze_dinov2=True):
        super(Baseline_MultimodalModel, self).__init__()

        # Initializing the BERT and DINOv2 model instances
        self.bert_model = BERTClass()
        self.dinov2_model = DINOv2()

        # If freeze_bert is set to True, freeze all BERT model parameters
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            self.bert_model.eval() #######################################################################

        # If freeze_dinov2 is set to True, freeze all DINOv2 model parameters
        if freeze_dinov2:
            for param in self.dinov2_model.parameters():
                param.requires_grad = False
            self.dinov2_model.eval() ##############################################################################

        # Fully connected layer to combine the outputs of the BERT and DINOv2 models
        self.fc = torch.nn.Sequential(
            nn.Linear(512, 256),
            ### batch normalization
            nn.BatchNorm1d(256),  # BatchNorm after Linear layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),  # BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, text_data, image_data):
        # Forward pass for the BERT model
        bert_output = self.bert_model(*text_data)

        # Forward pass for the DINOv2 model
        dinov2_output = self.dinov2_model(image_data)

        # Combining the outputs of the BERT and DINOv2 models
        combined_output = torch.cat((bert_output, dinov2_output), dim=1)

        # Passing the combined output through the fully connected layer
        output = self.fc(combined_output)

        return output
    
