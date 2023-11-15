
# # Define the EarlyStopping class
# class EarlyStopping:
#     def __init__(self, patience=10):
#         self.patience = patience
#         self.counter = 0
#         self.best_f1 = 0.0
#         self.stop = False

#     def check(self, f1):
#         if f1 > self.best_f1:
#             self.best_f1 = f1
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.stop = True

import torch

class EarlyStopping:
    def __init__(self, model, filepath="best_model.pth", patience=10):
        self.model = model
        self.filepath = filepath
        self.patience = patience
        self.counter = 0
        self.best_f1 = 0.0
        self.stop = False
        self.best_model_weights = None

    def check(self, f1):
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.counter = 0
            self.best_model_weights = self.model.state_dict().copy()  # Save the current model weights
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def save_best_weights(self):
        if self.best_model_weights:
            torch.save(self.best_model_weights, self.filepath)
