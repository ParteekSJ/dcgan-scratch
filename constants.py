import os
import torch
from torchvision import transforms

os.environ["IPDB_CONTEXT_SIZE"] = "7"
BASE_DIR = "."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = "./logs"
IMAGE_DIR = "./images"
CHECKPOINT_DIR = "./checkpoints"
CHANNELS_IMG = 1
TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.1307], std=[0.3081]),
        # dont need to modify transforms for RGB images.
        transforms.Normalize(
            mean=[0.5 for _ in range(CHANNELS_IMG)],
            std=[0.5 for _ in range(CHANNELS_IMG)],
        ),
    ]
)
NUM_EPOCHS = 200
BATCH_SIZE = 256
HIDDEN_DIM = 32
Z_DIM = 64
CRITERION = torch.nn.BCEWithLogitsLoss()
D_LR, G_LR = 2e-4, 2e-4
DISPLAY_STEP = 10
