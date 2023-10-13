import torch
from torchvision import transforms
from score_models import ScoreModel, EnergyModel, NCSNpp, MLP, DDPM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args): 
    
    # Importing the models hparams and weights
    #hyper_params_json = open("/home/mjybarth/scratch/CHECKPOINTS/model_hparams.json")
    #model_hparams = json.load(hyper_params_json)
    #sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]
    #score_model = ScoreModel(checkpoints_directory="/home/mjybarth/scratch/CHECKPOINTS")
    B = args.batchsize
    epochs = args.epochs
    IMAGE_SIZE = (64, 64, 3)
    BATCH_SIZE = B
    DEVICE = "cuda"
    
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomRotation(5),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.96, 1.0), ratio=(0.95, 1.05)),
            transforms.ToTensor(),
            # normalize images to [-1, 1] range
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Grayscale()
        ]),
    }
    class CatsDogsDataset(Dataset):
        """Custom Dataset for loading cat images"""

        def __init__(self, img_dir, transform=None, device=DEVICE):

            self.img_dir = img_dir

            self.img_names = [i for i in 
                              os.listdir(img_dir) 
                              if i.endswith('.jpg')]

            self.transform = transforms.Compose([
            #transforms.RandomRotation(5),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.96, 1.0), ratio=(0.95, 1.05)),
            transforms.ToTensor(),
            # normalize images to [-1, 1] range
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Grayscale()
        ])
            self.device = device


        def __getitem__(self, index):
            img = Image.open(os.path.join(self.img_dir,
                                          self.img_names[index]))

            if self.transform is not None:
                img = self.transform(img)

            return img.to(self.device)

        def __len__(self):
            return len(self.img_names)
        
    train_dataset = CatsDogsDataset(img_dir='/home/mjybarth/scratch/PetImages/Cat/', transform=data_transforms['train'])
    C = 1
    dimensions = (64, 64)
    checkpoints_directory = "/home/mjybarth/scratch/CHECKPOINTS"

    # Create a ScoreModel instance with Yang Song's NCSN++ architecture and the VESDE
    #net = NCSNpp(channels=C, dimensions=len(dimensions), nf=128, ch_mult=[2, 2, 2, 2])
    #model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=100, device="cuda")
    model = ScoreModel(checkpoints_directory = "/home/mjybarth/scratch/cats_2_checkpoints")
    # Train the score model, and save its weight in checkpoints_directory
    model.fit(train_dataset, epochs=epochs, batch_size=B, learning_rate=1e-4, checkpoints_directory="/home/mjybarth/scratch/cats_3_checkpoints")
        

if __name__ == "__main__": 
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--batchsize",          required = True,    default = 64,   type = int,     help = "Training batch size")
    parser.add_argument("--epochs",          required = True,    default = 100,   type = int,     help = "Training batch size")
    args = parser.parse_args()
    main(args) 
