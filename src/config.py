import os

# Paths
DATA_ROOT = 'data/'
MODELS_ROOT = 'models/'
TEMP_CACHE_DIR = 'temp_cache/'

# UCF101 Dataset
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
CLASSES = ['HandstandPushups', 'HandstandWalking', 'PullUps', 'Punch', 'PushUps']

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048 # InceptionV3 output shape
