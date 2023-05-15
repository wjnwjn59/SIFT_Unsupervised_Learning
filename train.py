import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from networks.SiameseNet import (
    SiameseVGG16,
    contrastive_loss
)
from data.datasets import PatchesDataset
from utils.loss import ContrastiveLoss

EPOCHS = 200
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
LR = 1e-4
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 8
RANDOM_SEED = 0

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(
    model,
    val_dataloader,
    device,
    criterion,
    margin=2.0
):
    model.eval()
    val_loss = 0
    running_items = 0
    correct = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_dataloader):
            images1, images2 = inputs
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            outputs1, outputs2 = model(images1, images2)
            euclid_dist = nn.functional.pairwise_distance(outputs1, outputs2)

            # loss = criterion(outputs1, outputs2, euclid_dist, labels, margin)
            loss  = criterion(outputs1, outputs2, labels)
            
            running_items += images1.size(0)
            val_loss += loss.item() * images1.size(0)

            predictions = (euclid_dist < margin).float()
            correct += (predictions == labels).sum().item() * images1.size(0)

    val_loss = val_loss / running_items
    val_acc = correct / running_items

    return val_loss, val_acc

def train(
    model, 
    dataset,
    optimizer, 
    #scheduler,
    criterion, 
    num_epochs, 
    device,
    margin=2.0
):
    train_dataloader, val_dataloader = dataset
    for epoch in range(num_epochs):
        model.train()
        
        train_loss = 0.0
        running_items = 0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            images1, images2 = inputs
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            outputs1, outputs2 = model(images1, images2)
            euclid_dist = nn.functional.pairwise_distance(outputs1, outputs2)

            # loss = criterion(outputs1, outputs2, euclid_dist, labels, margin)
            loss  = criterion(outputs1, outputs2, labels)
            loss.backward()

            optimizer.step()
            
            running_items += images1.size(0)
            train_loss += loss.item() * images1.size(0)
            
        train_loss = train_loss / running_items
        val_loss, val_acc = test(
            model,
            val_dataloader,
            device, 
            criterion,
            margin=2.0
        )
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        type=str,
        default='./data'
    )
    parser.add_argument(
    	'--epochs', 
    	type=int, 
        default=50,
    	help='Number of iterations'
    )
    parser.add_argument(
    	'--learning_rate', 
    	type=float, 
        default=1e-5,
    )
    parser.add_argument(
    	'--batch_size', 
    	type=int, 
        default=128,
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    if args.batch_size:
        TRAIN_BATCH_SIZE = args.batch_size
    if args.learning_rate:
        LR = args.learning_rate
    if args.epochs:
        EPOCHS = args.epochs

    image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    task_dataset = PatchesDataset(
        root_dataset=args.root_dir,
        transforms=image_transforms
    )

    train_size = int(len(task_dataset) * TRAIN_SIZE)
    val_size = int(len(task_dataset) * VAL_SIZE)
    test_size = len(task_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        task_dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=TEST_BATCH_SIZE
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=TEST_BATCH_SIZE
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SiameseVGG16().to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=LR
    )
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, 
    #     step_size=10, 
    #     gamma=0.1
    # )
    criterion = ContrastiveLoss()

    train(
        model=model,
        dataset=(train_dataloader, val_dataloader),
        optimizer=optimizer,
        #scheduler=scheduler,
        criterion=criterion,
        num_epochs=EPOCHS,
        device=device
    )
    
if __name__ == '__main__':
    main()