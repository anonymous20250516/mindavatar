import os 
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import FMRI2FACEModel
from dataset import FMRIDataset
import pandas as pd
import math
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    category = None

    train_dataset = FMRIDataset(sub=args.sub, split='train', use_vc=args.use_vc, category=category)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = FMRIDataset(sub=args.sub, split='test', category=category, use_vc=args.use_vc)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    input_dim = train_dataset.voxel_num
    output_dim = train_dataset.pca_dim
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    model = FMRI2FACEModel(input_dim=input_dim, output_dim=output_dim).to(device)

    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=1000, cycle_mult=2, 
        max_lr=args.lr, min_lr=args.lr * 0.01, warmup_steps=100, gamma=1.0,
    )

    args.ckpt_dir = os.path.join(args.ckpt_dir, f"sub{args.sub:02d}", datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = os.path.join(args.ckpt_dir, 'tblog')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_test_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}/{args.epochs}", total=len(train_loader))
        all_outputs = []
        all_targets = []
        all_categories = []
        for batch_idx, batch in progress_bar:
            fmri = batch['fmri'].to(device)  # shape: (B, input_dim)
            pca = batch['pca'].to(device)    # shape: (B, output_dim)

            optimizer.zero_grad()
            outputs = model(fmri)

            
            valid_mask = (pca.abs().sum(dim=1) != 0)  # shape: (B,)
            valid_pca = pca[valid_mask]
            valid_outputs = outputs[valid_mask, :]
            if valid_outputs.size(0) > 0:
                loss = criterion(valid_outputs, valid_pca)
            else:
                loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)

            loss.backward()
            optimizer.step()
            
            current_iter = epoch - 1 + (batch_idx + 1) / len(train_loader)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_rate/Train', current_lr, global_step)
            
            running_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(pca.detach().cpu())
            all_categories.extend(batch['category'])
        
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Train/Epoch_Loss', epoch_loss, epoch)
        print(f"Epoch {epoch} Train Loss: {epoch_loss:.6f}")

        model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []
        all_categories = []
        all_video_names = []
        with torch.no_grad():
            for batch in test_loader:
                fmri = batch['fmri'].to(device)
                pca = batch['pca'].to(device)
                outputs = model(fmri)

                loss = criterion(outputs, pca)

                test_loss += loss.item()

                all_outputs.append(outputs.detach().cpu())
                all_targets.append(pca.detach().cpu())
                all_categories.extend(batch['category'])
                all_video_names.extend(batch['video_name'])
        test_loss = test_loss / len(test_loader)
        writer.add_scalar('Test/Epoch_Loss', test_loss, epoch)
        print(f"Epoch {epoch} Test Loss: {test_loss:.6f}")

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            ckpt_path = os.path.join(args.ckpt_dir, f"sub_{args.sub}.pth")
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved model to {ckpt_path}")

            # save pred results
            output_path = os.path.join(args.ckpt_dir, f"sub_{args.sub}_test_pred.npy")
            np.save(output_path, all_outputs.numpy())
            output_path = os.path.join(args.ckpt_dir, f"sub_{args.sub}_test_gt.npy")
            np.save(output_path, all_targets.numpy())

        
    writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train fMRI Regression Model with CosineAnnealingWarmRestarts")
    parser.add_argument("--sub", type=int, default=1, choices=[1,2,3,4], help="Subject number")
    parser.add_argument("--train_num", type=int, default=0, help="Number of training data")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay (L2 regularization)")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory for tensorboard logs")
    parser.add_argument("--use_vc", action='store_true', help="Whether to use visual cortex ROI selection")
    parser.add_argument("--use_time", 
                       type=int, 
                       nargs='+',  # Accepts one or more integers
                       choices=range(0, 13),  # Valid values: 1-13
                       default=list(range(0, 13)),  # Default: all times (1-13)
                       help="List of time points to use (1-13)")
    args = parser.parse_args()

    train(args)
