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
    return avg_loss, avg_acc

def evaluate(model, loader, criterion, device, epoch, writer):
    model.eval()
    loader.dataset.set_val_mode(True)
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    all_preds = []
    
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
            
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of class 1 (AI)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # 防止验证集为空的情况
    if len(loader) == 0:
        logger.warning(f"Validation loader is empty! Skipping validation for epoch {epoch}")
        return 0.0, 0.0, [], [], []
    
    avg_loss = total_loss / len(loader)
    avg_acc = correct / total if total > 0 else 0.0
    
    logger.info(f"Val Epoch {epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Acc', avg_acc, epoch)
    return avg_loss, avg_acc, all_labels, all_probs, all_preds

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
    
    # Track statistics
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, writer, args)
        val_loss, val_acc, val_labels, val_probs, val_preds = evaluate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step()
        
        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
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
    
    # --- Generate Plots & History ---
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
        
        # 1. Save History CSV
        df_history = pd.DataFrame(history)
        df_history.to_csv(out_dir / 'training_history.csv', index=False)
        logger.info(f"Saved training history to {out_dir / 'training_history.csv'}")
        
        # 2. Plot Training Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        ax1.plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
        ax1.plot(df_history['epoch'], df_history['val_loss'], label='Val Loss')
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curve
        ax2.plot(df_history['epoch'], df_history['train_acc'], label='Train Acc')
        ax2.plot(df_history['epoch'], df_history['val_acc'], label='Val Acc')
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(out_dir / 'training_curves.png')
        plt.close()
        logger.info(f"Saved curves to {out_dir / 'training_curves.png'}")
        
        # Load best model for final evaluation plots
        best_ckpt = out_dir / 'best.pth'
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
            
            logger.info("Running evaluation with best model for final plots...")
            _, _, final_labels, final_probs, final_preds = evaluate(model, val_loader, criterion, device, 'Final', None)
            
            if len(final_labels) > 0:
                # 3. Plot Confusion Matrix
                cm = confusion_matrix(final_labels, final_preds)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Real (0)', 'AI (1)'],
                            yticklabels=['Real (0)', 'AI (1)'])
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(out_dir / 'confusion_matrix.png')
                plt.close()
                logger.info(f"Saved Confusion Matrix to {out_dir / 'confusion_matrix.png'}")
                
                # 4. Plot ROC Curve
                fpr, tpr, _ = roc_curve(final_labels, final_probs)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(out_dir / 'roc_curve.png')
                plt.close()
                logger.info(f"Saved ROC Curve to {out_dir / 'roc_curve.png'}")
                
                # 5. Bad Cases Visualisation
                try:
                    bad_cases_dir = out_dir / 'bad_cases'
                    bad_cases_dir.mkdir(exist_ok=True)
                    bad_cases_count = 0
                    
                    import cv2
                    import numpy as np
                    
                    for i in range(len(final_labels)):
                        if final_labels[i] == 0 and final_preds[i] == 1:
                            # 假阳性 (False Positive): 把真图当成假图
                            case_type = "FP_RealAsAI"
                            prob = final_probs[i]
                            # Since dataloader uses random batching/shuffling for simple iteration,
                            # mapping index back to filename requires info.
                            # Oh actually val_loader is shuffle=False, so index is deterministic!
                            img_path = val_dataset.entries[i]['image']
                            
                            if bad_cases_count < 10:  # Select top 10 mistakes or first 10
                                # read origin image
                                orig_img = cv2.imread(img_path)
                                if orig_img is not None:
                                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                                    plt.figure()
                                    plt.imshow(orig_img)
                                    plt.title(f"True: Real | Pred: AI (Score: {prob:.3f})\nPath: {Path(img_path).name}")
                                    plt.axis('off')
                                    plt.savefig(bad_cases_dir / f'bad_{bad_cases_count}_{case_type}_{Path(img_path).name}')
                                    plt.close()
                                    bad_cases_count += 1
                                    
                except Exception as e:
                    logger.warning(f"Failed to generate bad cases: {e}")
                
    except Exception as e:
        logger.warning(f"Error during plotting: {e}")

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
