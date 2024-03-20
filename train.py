import torch
import json
from utils.seed_everything import seed_everything
from utils.train_logger import TrainLogger
from tqdm import tqdm
from dataset import DatasetCLAP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from models.clap import HANCECLAP
import os
import time
from loss_fn import norm_temp_scaled_cross_entropy_loss
import pandas as pd
import datetime
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import random

def set_cuda_device():
    device_name = "CPU"
    if os.name == "posix":
        device = torch.device('mps') if torch.cuda.is_available() else torch.device('cpu')
    if os.name == "nt":
        if (torch.cuda.is_available() != True):
            device = torch.device('cpu')
        else:
            id = torch.cuda.current_device()
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(id)

    return device, device_name

@torch.no_grad()
def evaluate(settings, model, train_loader:DataLoader=None, val_loader:DataLoader=None, n_batches:int=5, device='cpu'):
    """
    Make sure data loaders are shuffling the data to get random batches.

    Adapted from Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
    """

    if (train_loader == None and val_loader == None): 
        print("evaluate(): You need to supply at least one of these: train_loader, val_loader")
        return
    
    model.eval()
    loaders = {'train': train_loader, 'validation': val_loader}
    average_loss = {}
    print("Evaluation in progress...")
    for split in loaders:
        batch_loss = 0
        for i, batch in tqdm(enumerate(loaders[split])):
            audio_features, text_features = model(batch[0], batch[1], batch[2]) # audio, label, target class
            loss = norm_temp_scaled_cross_entropy_loss(audio_features, text_features, temp=settings['nt_xent_temperature'], device=device)
            batch_loss += loss.item()
            if (i == n_batches - 1): 
                break

        average_loss[split] = batch_loss / n_batches
    
    model.train()
    return average_loss

def train_step(settings, model, optim:Optimizer, scheduler, train_loader:DataLoader, device='cpu'):
    model.train()
    total_loss = 0
    for i, (audio, label, target) in enumerate(tqdm(train_loader)):
        optim.zero_grad()
        #with torch.cuda.amp.autocast(enabled=self.args.fp16_precision): # Can't get this to work.
        audio_features, text_features = model(audio, label, target)
        print(audio_features.shape)
        print(text_features.shape)
        loss = norm_temp_scaled_cross_entropy_loss(audio_features, text_features, temp=settings['nt_xent_temperature'], device=device)
        # Idea: Calculate NT-Xent separately for augmented audios and augmented texts, then for audio-text pairs.
        total_loss += loss.item()
        print(f"Current Loss: {loss.item()}")
        loss.backward()
        optim.step()
        scheduler.step()

    return total_loss

def train_fn(settings, model, optim:Optimizer, scheduler, train_loader:DataLoader, val_loader:DataLoader, logger:TrainLogger, device='cpu', current_epoch=0):
    n_epochs = settings['num_of_epochs'] - current_epoch

    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log['parameters'] = n_parameters
    logger.log['trainable_parameters'] = n_trainable_parameters

    train_start_date = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    start_time = time.time()
    for i, epoch in enumerate(range(n_epochs)):
        loss = train_step(settings, model, optim, scheduler, train_loader)
        elapsed_training_time = time.time() - start_time

        eval_start_time = time.time()
        eval_loss = evaluate(settings, model, train_loader, val_loader, n_batches=settings['n_eval_batches'], device=device)
        eval_time = time.time() - eval_start_time

        print(f"Average Loss on Training Set for Epoch {i}: {eval_loss['train']} ")
        print(f"Average Loss on Validation Set for Epoch {i}: {eval_loss['validation']} ")

        # Save and log after each epoch.
        logger.log_epoch({
            "elapsed_epoch_time": elapsed_training_time,
            "elapsed_training_time": elapsed_training_time-eval_time,
            "elapsed_evaluation_time": eval_time,
            "loss": loss / len(train_loader),
            "eval" : eval_loss['validation']
        },
        epoch=epoch+1+current_epoch)

        logger.save_log()

        torch.save(model.state_dict(), f"model_weights/hance_clap/{settings['model_name']}-{train_start_date}-{epoch+1+current_epoch}.hance")

def main():
    with open ('model_definitions/clap.json', 'r') as f:
        settings = json.load (f)

    seed_everything(settings["seed"])

    device, device_name = set_cuda_device()
    print(f"Starting training session on: {device_name}")

    logger = TrainLogger('logs', settings=settings)

    # Load dataset
    train_df = pd.read_csv(settings['dataset_drive'] + settings['train_set_path'])
    dev_df = pd.read_csv(settings['dataset_drive'] + settings['dev_set_path'])
    test_df = pd.read_csv(settings['dataset_drive'] + settings['test_set_path'])

    print(train_df.iloc[:settings['n_sanity_train_samples']])

    if(settings['n_sanity_train_samples'] > 0):
        dataset_train = DatasetCLAP(train_df.iloc[:settings['n_sanity_train_samples']], settings=settings, device=device, class_reduction=settings['classes_to_include'], include=True)
    else:
        dataset_train = DatasetCLAP(train_df, settings=settings, device=device, class_reduction=settings['classes_to_include'], include=True)

    dataset_dev = DatasetCLAP(dev_df, settings=settings, device=device, class_reduction=settings['classes_to_include'], include=True)
    dataset_test = DatasetCLAP(test_df, settings=settings, device=device)

    train_loader = DataLoader(dataset_train, batch_size=settings['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_dev, batch_size=settings['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=settings['batch_size'], shuffle=True)

    # Train
    
    model = HANCECLAP(settings=settings, device=device, print_shapes=False).to(device)
    #model.load_state_dict(torch.load("model_weights/hance_clap/hance_clap-n_classes_2-water_doors-sr8Khz-2024_03_16-13_25_51-6.hance"))

    optim = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=0.1)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    n_training_steps = settings['num_of_epochs'] * len(train_loader)
    n_warmup_steps = np.floor(n_training_steps * 0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_training_steps, eta_min=1e-7)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optim, num_warmup_steps=n_warmup_steps, num_training_steps=n_training_steps)
    
    model.to(device)
    train_fn(settings=settings, model=model, optim=optim, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, logger=logger)
    #loss = evaluate(settings, model, train_loader, val_loader, device=device)
    #print(f"Train Loss = {loss['train']} Val Loss = {loss['validation']}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device='cuda', abbreviated=False)

    main()
