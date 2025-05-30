"""
Script for fine-tuning Swin Transformer on PASTIS dataset
"""
import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import timm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.dataset import PASTIS_Dataset
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src import utils
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import random

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    def value(self):
        return self.avg

class RandomHorizontalFlip:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask

class RandomVerticalFlip:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(img, angle), TF.rotate(mask, angle)

class MultispectralColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img, mask):
        # Apply brightness and contrast adjustments to each channel independently
        if random.random() < 0.5:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            img = img * brightness_factor
        if random.random() < 0.5:
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = img.mean(dim=[-2, -1], keepdim=True)
            img = (img - mean) * contrast_factor + mean
        return img, mask

class RandomErasing:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            h, w = img.shape[-2:]
            area = h * w
            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3.0)
            h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
            if h_erase < h and w_erase < w:
                i = random.randint(0, h - h_erase)
                j = random.randint(0, w - w_erase)
                img[..., i:i+h_erase, j:j+w_erase] = 0
        return img, mask

class SwinForSemanticSegmentation(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        # Load pre-trained Swin Transformer
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=10,    # Accept 10-channel input
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        # Get the feature dimensions from the last stage
        self.feature_dim = self.swin.feature_info.channels()[-1]
        
        # Add segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Upsampling layers to match input resolution
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
        )

    def forward(self, x):
        # Input shape: (B, C, H, W)
        features = self.swin(x)
        # Use the last feature map
        x = features[-1]
        if x.shape[1] != self.feature_dim:
            # If shape is [B, H, W, C], permute to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.segmentation_head(x)
        x = self.upsample(x)
        return x

def collate_central_timestamp(batch):
    # batch is a list of tuples: ( (x, dates), y )
    xs, ys = [], []
    for (x, dates), y in batch:
        # x shape: (T, C, H, W) or (C, H, W)
        if x.ndim == 4:
            t = x.shape[0] // 2
            x_central = x[t]  # (C, H, W)
        else:
            x_central = x
        # Resize to 224x224
        x_central = F.interpolate(x_central.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        xs.append(x_central)
        ys.append(y)
    xs = torch.stack(xs, 0)
    ys = torch.stack(ys, 0)
    return (xs, None), ys

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    iou_meter = IoU(num_classes=args.num_classes, ignore_index=args.ignore_index)
    
    # Data augmentation transforms
    transforms_list = [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=15),
        MultispectralColorJitter(brightness=0.2, contrast=0.2),  # Using the new multispectral version
        RandomErasing(p=0.2)
    ]
    
    for i, batch in enumerate(loader):
        (x, dates), y = batch
        if x.dim() == 5:
            t = x.shape[1] // 2  # Use central timestamp
            x = x[:, t]  # shape: (B, C, H, W)
        y = y.long()
        
        # Apply data augmentation
        if model.training:
            for transform in transforms_list:
                x, y = transform(x, y)
        
        x, y = x.to(device), y.to(device)
        
        # Clip values to valid range after augmentation
        x = torch.clamp(x, min=0.0, max=1.0)
        
        optimizer.zero_grad()
        out = model(x)
        # Resize output to match target if needed
        if out.shape[-2:] != y.shape[-2:]:
            out = torch.nn.functional.interpolate(out, size=y.shape[-2:], mode='bilinear', align_corners=False)
        
        # Add label smoothing to loss
        loss = criterion(out, y)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = out.argmax(dim=1)
            iou_meter.add(pred, y)
            loss_meter.add(loss.item())
        
        if (i + 1) % args.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(f'Step [{i+1}/{len(loader)}], Loss: {loss_meter.value():.4f}, '
                  f'Acc: {acc:.2f}, mIoU: {miou:.2f}')
    
    miou, acc = iou_meter.get_miou_acc()
    return {
        'loss': loss_meter.value(),
        'accuracy': acc,
        'miou': miou
    }

def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = IoU(num_classes=args.num_classes, ignore_index=args.ignore_index)
    
    with torch.no_grad():
        for batch in loader:
            (x, dates), y = batch
            if x.dim() == 5:
                t = x.shape[1] // 2  # Use central timestamp
                x = x[:, t]
            y = y.long()
            
            x, y = x.to(device), y.to(device)
            
            out = model(x)
            # Resize output to match target if needed
            if out.shape[-2:] != y.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=y.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(out, y)
            
            pred = out.argmax(dim=1)
            iou_meter.add(pred, y)
            loss_meter.add(loss.item())
    
    miou, acc = iou_meter.get_miou_acc()
    return {
        'loss': loss_meter.value(),
        'accuracy': acc,
        'miou': miou
    }

def main(args):
    # Set up device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.res_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.res_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Dataset
    dt_args = dict(
        folder=args.dataset_folder,
        norm=True,
        reference_date=args.ref_date,
        mono_date=args.mono_date,
        target="semantic",
        sats=["S2"]
    )
    
    # Use first fold for training, second for validation
    train_dataset = PASTIS_Dataset(**dt_args, folds=[1, 2, 3], cache=args.cache)
    val_dataset = PASTIS_Dataset(**dt_args, folds=[4], cache=args.cache)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_central_timestamp
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_central_timestamp
    )
    
    # Model with increased regularization
    model = SwinForSemanticSegmentation(
        model_name=args.pretrained_model,
        num_classes=args.num_classes,
        pretrained=True,
        drop_rate=0.1,  # Increased from 0.0
        attn_drop_rate=0.1,  # Increased from 0.0
        drop_path_rate=0.2  # Increased from 0.1
    ).to(device)
    
    # Loss with label smoothing
    weights = torch.ones(args.num_classes, device=device)
    weights[args.ignore_index] = 0
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    # Optimizer with reduced learning rates
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'swin' in name:
            encoder_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr * 0.5},  # Reduced from 1e-5
        {'params': other_params, 'lr': args.lr * 0.5}  # Reduced from 1e-4
    ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, args.epochs // 3),
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training loop
    best_miou = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
              f'Acc: {train_metrics["accuracy"]:.2f}, '
              f'mIoU: {train_metrics["miou"]:.2f}')
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f'Val - Loss: {val_metrics["loss"]:.4f}, '
              f'Acc: {val_metrics["accuracy"]:.2f}, '
              f'mIoU: {val_metrics["miou"]:.2f}')
        
        # Update learning rate
        scheduler.step()

        print("Hi: -> ", args.res_dir)
        
        # Save best model
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(args.res_dir, 'best_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
        }, os.path.join(args.res_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--pretrained_model', default='swin_tiny_patch4_window7_224',
                      help='Name of pre-trained Swin model to use')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--ignore_index', type=int, default=-1)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5,  # Reduced from 1e-4
                      help='Learning rate for non-encoder parameters')
    parser.add_argument('--encoder_lr', type=float, default=5e-6,  # Reduced from 1e-5
                      help='Learning rate for encoder parameters')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--display_step', type=int, default=50)
    
    # Dataset parameters
    parser.add_argument('--dataset_folder', type=str, required=True,
                      help='Path to PASTIS dataset')
    parser.add_argument('--ref_date', type=str, default='2018-09-01')
    parser.add_argument('--mono_date', type=str, default=None)
    parser.add_argument('--cache', action='store_true',
                      help='Cache dataset in memory')
    
    # Other parameters
    parser.add_argument('--res_dir', type=str, default='./results_swin',
                      help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args) 