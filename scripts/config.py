import torch

class Config:
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "resnet50"
    BATCH_SIZE = 32
    TEXT_LR = 1e-5
    IMAGE_LR = 1e-5
    HEAD_LR = 1e-3
    EPOCHS = 50
    DROPOUT = 0.15
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    SAVE_PATH = "best_model.pth" 
    TARGET_MAE = 50
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS_NO_IMPROVEMENTS = 5