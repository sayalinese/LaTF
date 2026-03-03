import os
import sys
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dotenv import load_dotenv

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

# Imports
from service.dataset import ImageDataset
from service.model_v11_fusion import LaREDeepFakeV11

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, writer, args):
    model.train()
    loader.dataset.set_val_mode(False)
    
    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0
    correct = 0
    total = 0
    
    # AutoCast settings
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # [V12] Seg Loss
    seg_criterion = nn.BCEWithLogitsLoss()
    
    for i, batch in enumerate(loader):
        # [V13] Unified Unpacking Logic
        # V13: (img_clip, label, loss_map, img_highres, mask, trufor) -> len 6
        # V12: (img_clip, label, loss_map, img_highres, mask) -> len 5
        # V11: (img_clip, label, loss_map, img_highres) -> len 4
        
        batch_len = len(batch)
        masks = None
        trufor = None
        use_seg = False
        
        img_clip = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True).view(-1)
        loss_maps = batch[2].to(device, non_blocking=True)
        
        if batch_len >= 4:
             img_highres = batch[3].to(device, non_blocking=True)
             
        if batch_len >= 5:
             # V12 Mask
             masks = batch[4].to(device, non_blocking=True)
             use_seg = True
             
        if batch_len >= 6:
             # V13 TruFor
             trufor = batch[5].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
            # Forward Pass: Dual Stream + LaRE + Seg Head + [V13 TruFor]
            if use_seg:
                # Pass TruFor map if available (V13)
                logits, seg_logits = model(img_clip, img_highres, loss_maps, trufor_map=trufor, return_seg=True)
                
                # [V12] Classification Loss
                cls_loss = criterion(logits, labels)
                
                # [V12] Segmentation Loss
                # Resize mask to match seg_logits size (32x32) for efficiency
                # masks shape: [B, 1, 512, 512] -> [B, 1, 32, 32]
                masks_small = torch.nn.functional.interpolate(masks, size=(32, 32), mode='nearest')
                seg_loss = seg_criterion(seg_logits, masks_small)
                
                # Joint Loss
                loss = cls_loss + 0.5 * seg_loss
                
                total_seg_loss += seg_loss.item()
                total_cls_loss += cls_loss.item()
            else:
                logits = model(img_clip, img_highres, loss_maps, trufor_map=trufor)
                loss = criterion(logits, labels)
                total_cls_loss += loss.item()

        # Backward
        if torch.isnan(loss):
            logger.error(f"NaN Loss at step {i}")
            continue
        
        if scaler and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        # Stats
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if i % 20 == 0:
            msg = f"Epoch [{epoch}] Step [{i}/{len(loader)}] Loss: {loss.item():.4f}"
            if use_seg:
                msg += f" (Cls: {cls_loss.item():.4f}, Seg: {seg_loss.item():.4f})"
            msg += f" Acc: {correct/total:.4f}"
            
            logger.info(msg)
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(loader) + i)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total
    logger.info(f"Epoch {epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Acc', avg_acc, epoch)

def evaluate(model, loader, criterion, device, epoch, writer):
    model.eval()
    loader.dataset.set_val_mode(True)
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # [V13] Unified Unpacking Logic
            batch_len = len(batch)
            img_clip = batch[0].to(device)
            labels = batch[1].to(device).view(-1)
            loss_maps = batch[2].to(device)
            
            img_highres = None
            masks = None
            trufor = None
            
            if batch_len >= 4:
                img_highres = batch[3].to(device)
            if batch_len >= 5:
                # Mask not really used in eval loop unless we evaluate iou, but skipping for now
                pass
            if batch_len >= 6:
                # V13 TruFor
                trufor = batch[5].to(device)
            
            # Forward
            logits = model(img_clip, img_highres, loss_maps, trufor_map=trufor)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    # 防止验证集为空的情况
    if len(loader) == 0:
        logger.warning(f"Validation loader is empty! Skipping validation for epoch {epoch}")
        return 0.0
    
    avg_loss = total_loss / len(loader)
    avg_acc = correct / total if total > 0 else 0.0
    
    logger.info(f"Val Epoch {epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Acc', avg_acc, epoch)
    return avg_acc

def main():
    parser = argparse.ArgumentParser(description='Train LaRE V11 Fusion Model')
    parser.add_argument('--epochs', type=int, default=int(os.getenv('EPOCHES', '30')))
    parser.add_argument('--batch_size', type=int, default=int(os.getenv('BATCH_SIZE', '16')))
    parser.add_argument('--lr', type=float, default=float(os.getenv('LR', '1e-4')))
    parser.add_argument('--data_root', type=str, default=os.getenv('DATA_ROOT', 'data'), help='Path to dataset root')
    parser.add_argument('--train_file', type=str, default=os.getenv('TRAIN_FILE', 'annotation/train_sdxl.txt'), help='Train txt list')
    parser.add_argument('--val_file', type=str, default=os.getenv('VAL_FILE', 'annotation/val_sdxl.txt'), help='Val txt list')
    parser.add_argument('--map_file', type=str, default=os.getenv('MAP_FILE', 'annotation/map_sdxl_train.txt'), help='Precomputed LaRE features path')
    parser.add_argument('--out_dir', type=str, default=os.getenv('OUT_DIR', 'outputs/v11_fusion'))
    parser.add_argument('--seed', type=int, default=int(os.getenv('SEED', '42')))
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--texture_model', type=str, default=os.getenv('TEXTURE_MODEL', 'convnext_tiny'), help='Backbone for texture branch')
    parser.add_argument('--clip_type', type=str, default=os.getenv('CLIP_TYPE', 'RN50x64'), help='CLIP visual backbone')
    parser.add_argument('--highres_size', type=int, default=int(os.getenv('HIGHRES_SIZE', '512')))
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Dir Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / 'logs'))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset (V11 Mode)
    logger.info("Initializing Datasets (V11 Dual-Stream + V13 TruFor Fusion)...")
    train_dataset = ImageDataset(
        data_root=args.data_root,
        train_file=args.train_file,
        map_file=args.map_file,
        data_size=448,          # For CLIP
        highres_size=args.highres_size, # For Texture
        enable_v11=True,        # ENABLE V11
        enable_trufor=True,     # ENABLE V13 TruFor
        is_train=True
    )
    
    val_dataset = ImageDataset(
        data_root=args.data_root,
        train_file=args.val_file, # Reuse loader logic but point to val file
        map_file=args.map_file,
        data_size=448,
        highres_size=args.highres_size,
        enable_v11=True,
        enable_trufor=True,     # ENABLE V13 TruFor
        is_train=False,
        drop_no_map=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Model
    logger.info(f"Creating Model V11 with texture backbone: {args.texture_model}")
    model = LaREDeepFakeV11(
        num_classes=2, 
        clip_type=args.clip_type, 
        texture_model=args.texture_model
    ).to(device)
    
    # Optimizer (Typically AdamW for ViTs/ConvNeXts)
    # CLIP is frozen by default inside model __init__, so only Texture + Heads are trained.
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler() if args.use_amp else None
    
    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training Loop
    best_acc = 0.0
    
    # Early Stopping setup
    early_stop_patience = int(os.getenv('EARLY_STOP_PATIENCE', 6))
    no_improve_epochs = 0
    
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, writer, args)
        val_acc = evaluate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step()
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0 # Reset counter
            logger.info(f"New Best Acc: {best_acc:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, out_dir / 'best.pth')
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epochs. Best Acc: {best_acc:.4f}")
            
        # Save Latest
        torch.save(model.state_dict(), out_dir / 'latest.pth')
        
        # Early Stopping Check
        if no_improve_epochs >= early_stop_patience:
            logger.info(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
            break
        
    writer.close()
    logger.info("Training Complete.")

    # Post-training: run per-category evaluation and save CSV
    try:
        best_ckpt = out_dir / 'best.pth'
        if best_ckpt.exists():
            logger.info(f"Running post-training per-category evaluation using {best_ckpt}")
            try:
                # Fix import path for test module
                import sys
                test_module_path = str(PROJECT_ROOT / 'test')
                if test_module_path not in sys.path:
                    sys.path.insert(0, test_module_path)
                from evaluate_by_category import evaluate_category_accuracy
                results = evaluate_category_accuracy(model_path=str(best_ckpt))
                # Write CSV
                import csv
                csv_path = out_dir / 'per_category_results.csv'
                with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                    writer_csv = csv.writer(cf)
                    header = ['Category','Total','Accuracy','Real_Acc','Fake_Acc','Real_Count','Fake_Count']
                    writer_csv.writerow(header)
                    for cat, res in results.items():
                        writer_csv.writerow([
                            cat,
                            res.get('Total', 0),
                            f"{res.get('Accuracy', 0):.4f}",
                            f"{res.get('Real_Acc', 0):.4f}",
                            f"{res.get('Fake_Acc', 0):.4f}",
                            res.get('Real_Count', 0),
                            res.get('Fake_Count', 0)
                        ])
                logger.info(f"Saved per-category CSV: {csv_path}")
            except Exception as e:
                logger.exception(f"Post-training evaluation failed: {e}")
        else:
            logger.warning("Best checkpoint not found; skipping post-training evaluation.")
    except Exception as e:
        logger.exception(f"Unexpected error during post-training evaluation: {e}")

if __name__ == '__main__':
    main()
