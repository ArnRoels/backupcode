#!pip install torch# config.py
import torch

class Config:
    # Paths
    DATA_DIR = "/media/arnout/BS5/qc_nn_bsubt_kd"
    GOOD_DIR = f"{DATA_DIR}/Good"
    BAD_DIR = f"{DATA_DIR}/bad"
    MODEL_PATH = f"{DATA_DIR}/quality_model_best_m4_v2.pth"
    TEST_GOOD = f"{DATA_DIR}/Test/Good_test"
    TEST_BAD = f"{DATA_DIR}/Test/Bad_test"
    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    VAL_SPLIT = 0.2
    EPOCHS = 1000
    PATIENCE = 5
    LEARNING_RATE = 0.0001
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
