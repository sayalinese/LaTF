import os
import sys
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
        # [V17 Perf] 输入尺寸固定 → benchmark=True 让 cuDNN 自动选最快卷积算法
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# [V17 Perf] 全局初始化 seg_criterion，避免每 epoch 重复 import + 创建
from service.forensic_localization import FLHLoss
_seg_criterion = FLHLoss(focal_weight=0.5, dice_weight=0.5)


def compute_multiscale_seg_loss(seg_outputs, masks, seg_criterion, args):
    if isinstance(seg_outputs, (tuple, list)) and len(seg_outputs) == 3:
        coarse_logits, mid_logits, fine_logits = seg_outputs
    else:
        fine_logits = seg_outputs
        mid_logits = F.interpolate(fine_logits, size=(64, 64), mode='bilinear', align_corners=False)
        coarse_logits = F.interpolate(fine_logits, size=(32, 32), mode='bilinear', align_corners=False)

    masks_32 = F.interpolate(masks, size=coarse_logits.shape[-2:], mode='nearest')
    masks_64 = F.interpolate(masks, size=mid_logits.shape[-2:], mode='nearest')
    masks_128 = F.interpolate(masks, size=fine_logits.shape[-2:], mode='nearest')

    loss_32 = seg_criterion(coarse_logits, masks_32)
    loss_64 = seg_criterion(mid_logits, masks_64)
    loss_128 = seg_criterion(fine_logits, masks_128)
    total_loss = (
        args.seg_weight_32 * loss_32
        + args.seg_weight_64 * loss_64
        + args.seg_weight_128 * loss_128
    )

    return total_loss, {
        'coarse': loss_32,
        'mid': loss_64,
        'fine': loss_128,
    }

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, writer, args):
    model.train()
    loader.dataset.set_val_mode(False)
    
    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0
    total_seg_loss_32 = 0
    total_seg_loss_64 = 0
    total_seg_loss_128 = 0
    correct = 0
    total = 0
    
    # AutoCast settings
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    seg_criterion = _seg_criterion
    
    for i, batch in enumerate(loader):
        # [V13] Unified Unpacking Logic
        # V13: (img_clip, label, loss_map, img_highres, mask, luma) -> len 6
        # V12: (img_clip, label, loss_map, img_highres, mask) -> len 5
        # V11: (img_clip, label, loss_map, img_highres) -> len 4
        
        batch_len = len(batch)
        masks = None
        luma = None
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
             # V13 Luma map (replaces TruFor)
             luma = batch[5].to(device, non_blocking=True)

        # [V14] 豆包真图非对称惩罚：提取路径，对 label=0 且路径含 doubao 的样本施加更高权重
        img_paths = batch[-1]  # dataset 末位始终追加 path 字符串
        sample_weights = torch.ones(labels.size(0), device=device)
        doubao_real_w = getattr(args, 'doubao_real_weight', 2.0)
        for _j, (_p, _lbl) in enumerate(zip(img_paths, labels.cpu().tolist())):
            if _lbl == 0 and 'doubao' in _p.lower().replace('\\', '/'):
                sample_weights[_j] = doubao_real_w

        optimizer.zero_grad()

        with autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
            # [V17] Forward Pass: Dual Stream + LaRE + Integrated Localization
            if use_seg:
                # 计算定位损失的白名单逻辑:
                # 1. 标签为 0 的真图 (强制空 mask, 惩罚 FP)
                # 2. 有实际局部掩码的假图 (Change, Doubao)
                # 屏蔽: 无掩码的全图假图 (如 SDXL/Flux), 避免全红热图干扰定位
                has_mask = (labels == 0) | (masks.sum(dim=(1, 2, 3)) > 0)  # [B] bool
                need_pyramid = has_mask.any().item()

                if need_pyramid:
                    logits, seg_outputs = model(
                        img_clip, img_highres, loss_maps,
                        luma_map=luma, return_seg_pyramid=True,
                    )
                else:
                    logits = model(img_clip, img_highres, loss_maps, luma_map=luma)
                    seg_outputs = None

                # [V14] 逐样本加权交叉熵（豆包真图惩罚倍率 doubao_real_weight）
                cls_loss_raw = F.cross_entropy(logits, labels, label_smoothing=0.1, reduction='none')
                cls_loss = (cls_loss_raw * sample_weights).mean()

                if need_pyramid and seg_outputs is not None:
                    # 只用有 mask 的样本计算 seg loss
                    seg_outputs_masked = tuple(s[has_mask] for s in seg_outputs)
                    masks_masked = masks[has_mask]
                    seg_loss, seg_stats = compute_multiscale_seg_loss(seg_outputs_masked, masks_masked, seg_criterion, args)
                    seg_loss = seg_loss * args.seg_loss_scale
                    loss = cls_loss + seg_loss
                    total_seg_loss += seg_loss.item()
                    total_seg_loss_32 += seg_stats['coarse'].item()
                    total_seg_loss_64 += seg_stats['mid'].item()
                    total_seg_loss_128 += seg_stats['fine'].item()
                else:
                    loss = cls_loss

                total_cls_loss += cls_loss.item()
            else:
                logits = model(img_clip, img_highres, loss_maps,
                               luma_map=luma)
                loss_raw = F.cross_entropy(logits, labels, label_smoothing=0.1, reduction='none')
                loss = (loss_raw * sample_weights).mean()
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
                msg += (
                    f" (Cls: {cls_loss.item():.4f}, Seg: {seg_loss.item():.4f}, "
                    f"32: {seg_stats['coarse'].item():.4f}, 64: {seg_stats['mid'].item():.4f}, "
                    f"128: {seg_stats['fine'].item():.4f})"
                )
            msg += f" Acc: {correct/total:.4f}"
            
            logger.info(msg)
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(loader) + i)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total
    logger.info(f"Epoch {epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Acc', avg_acc, epoch)
    if total_seg_loss > 0:
        writer.add_scalar('Train/SegLoss', total_seg_loss / len(loader), epoch)
        writer.add_scalar('Train/SegLoss32', total_seg_loss_32 / len(loader), epoch)
        writer.add_scalar('Train/SegLoss64', total_seg_loss_64 / len(loader), epoch)
        writer.add_scalar('Train/SegLoss128', total_seg_loss_128 / len(loader), epoch)
    return avg_loss, avg_acc

def validate(model, loader, criterion, device, epoch, writer):
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
            img_clip = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True).view(-1)
            loss_maps = batch[2].to(device, non_blocking=True)
            
            img_highres = None
            masks = None
            luma = None
            
            if batch_len >= 4:
                img_highres = batch[3].to(device, non_blocking=True)
            if batch_len >= 5:
                # Mask not really used in eval loop unless we evaluate iou, but skipping for now
                pass
            if batch_len >= 6:
                # V13 Luma map (replaces TruFor)
                luma = batch[5].to(device, non_blocking=True)

            # [V17] Forward — no external FLH (AMP for val too)
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with autocast('cuda', dtype=amp_dtype, enabled=True):
                logits = model(img_clip, img_highres, loss_maps, luma_map=luma)
                loss = criterion(logits, labels)
            
            probs = torch.softmax(logits.float(), dim=1)[:, 1] # Probability of class 1 (AI)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().float().numpy())
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


def build_optimizer(model, lr, weight_decay=0.01):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, total_epochs, eta_min=1e-6, warmup_epochs=0):
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
        eta_min=eta_min,
    )

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
    parser.add_argument('--doubao_real_weight', type=float,
                        default=float(os.getenv('DOUBAO_REAL_WEIGHT', '2.0')),
                        help='豆包真图(label=0)被误判时的损失惩罚倍率')
    parser.add_argument('--num_workers', type=int, default=int(os.getenv('WORKERS', '8')),
                        help='DataLoader CPU worker 数量，建议设为 CPU 核数的一半')
    parser.add_argument('--seg_loss_scale', type=float,
                        default=float(os.getenv('SEG_LOSS_SCALE', '1.0')),
                        help='Global scaling factor for total seg loss (cls:seg balance)')
    parser.add_argument('--warmup_epochs', type=int,
                        default=int(os.getenv('WARMUP_EPOCHS', '3')),
                        help='Linear warmup epochs before cosine decay')
    parser.add_argument('--seg_weight_32', type=float,
                        default=float(os.getenv('SEG_WEIGHT_32', '0.15')),
                        help='32x32 coarse localization loss weight')
    parser.add_argument('--seg_weight_64', type=float,
                        default=float(os.getenv('SEG_WEIGHT_64', '0.25')),
                        help='64x64 mid localization loss weight')
    parser.add_argument('--seg_weight_128', type=float,
                        default=float(os.getenv('SEG_WEIGHT_128', '0.50')),
                        help='128x128 fine localization loss weight')
    parser.add_argument('--freeze_loc_epochs', type=int,
                        default=int(os.getenv('FREEZE_LOC_EPOCHS', '0')),
                        help='Freeze localization branch for first N epochs to preserve cls dynamics')
    parser.add_argument('--stage3_finetune', action='store_true',
                        help='Stage 3: Freeze texture/CLIP/FLH, only train loc_refine and fusion_mlp')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a checkpoint (.pth) to resume/finetune from')

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
    logger.info("Initializing Datasets (V11 Dual-Stream + V15 FLH Fusion)...")
    train_dataset = ImageDataset(
        data_root=args.data_root,
        train_file=args.train_file,
        map_file=args.map_file,
        data_size=448,          # For CLIP
        highres_size=args.highres_size, # For Texture
        enable_v11=True,        # ENABLE V11
        enable_luma=True,       # ENABLE V15 Luma (replaces TruFor)
        is_train=True
    )
    
    val_dataset = ImageDataset(
        data_root=args.data_root,
        train_file=args.val_file, # Reuse loader logic but point to val file
        map_file=args.map_file,
        data_size=448,
        highres_size=args.highres_size,
        enable_v11=True,
        enable_luma=True,       # ENABLE V15 Luma (replaces TruFor)
        is_train=False,
        drop_no_map=False
    )
    
    # [V18] WeightedRandomSampler: 提高有 mask 样本的采样概率
    # change/doubao 样本权重更高，保证每 batch ~60% 有 mask 的定位数据
    mask_boost = float(os.getenv('MASK_SAMPLE_WEIGHT', '2.0'))
    sample_weights_list = []
    for path, _label in train_dataset.train_list:
        path_lower = path.lower().replace('\\', '/')
        if '/change/' in path_lower or '/doubao/' in path_lower:
            sample_weights_list.append(mask_boost)
        else:
            sample_weights_list.append(1.0)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_list,
        num_samples=len(sample_weights_list),
        replacement=True,
    )
    n_mask = sum(1 for w in sample_weights_list if w > 1.0)
    logger.info(
        f"[V18] WeightedRandomSampler: {len(sample_weights_list)} samples, "
        f"{n_mask} with mask (weight={mask_boost}), "
        f"{len(sample_weights_list) - n_mask} without mask (weight=1.0)"
    )

    # [V17 Perf] prefetch_factor=4 让 worker 预加载更多 batch，减少 GPU 等待
    _nw_train = args.num_workers
    _nw_val = max(0, args.num_workers // 2)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=_nw_train, pin_memory=True, persistent_workers=(_nw_train > 0),
        prefetch_factor=4 if _nw_train > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=_nw_val, pin_memory=True, persistent_workers=(_nw_val > 0),
        prefetch_factor=4 if _nw_val > 0 else None
    )
    
    # Model
    logger.info(f"Creating Model V11 with texture backbone: {args.texture_model}")
    model = LaREDeepFakeV11(
        num_classes=2, 
        clip_type=args.clip_type, 
        texture_model=args.texture_model
    ).to(device)

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Checkpoint loaded. {msg}")

    # [V17] FLH is now integrated into the model — no external loading needed

    # [V14] Localization branch freeze: identify all loc-only parameters
    _LOC_PREFIXES = (
        'luma_gate_loc.', 'ch_interact_loc.',
        'loc_stem.', 'loc_cbam1.', 'loc_cbam2.', 'loc_cbam3.', 'loc_head.',
        'loc_refine64.', 'loc_refine128.', 'mid_head.', 'fine_head.',
    )

    def _set_stage3_requires_grad(model):
        """
        [V14] Stage 3 Multi-stage Finetuning: 
        Freeze EVERYTHING (texture_branch, clip, luma_gate_loc, loc_stem, loc_cbam, loc_head, lare_conv)
        ONLY train loc_refine64, loc_refine128, mid_head, fine_head, and possibly fusion_mlp.
        """
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the refinement layers, final heads, AND coarse attention blocks
        unfreeze_names = [
            'loc_refine64', 'loc_refine128', 'mid_head', 'fine_head',
            'fusion_mlp', 'loc_head',
        ]
        for name, module in model.named_modules():
            if name in unfreeze_names:
                for param in module.parameters():
                    param.requires_grad = True

    def _set_loc_requires_grad(model, requires_grad):
        """Set requires_grad for all localization-only parameters."""
        count = 0
        for name, param in model.named_parameters():
            if name.startswith(_LOC_PREFIXES):
                param.requires_grad = requires_grad
                count += 1
        tag = "Unfroze" if requires_grad else "Froze"
        logger.info(f"{tag} {count} localization parameters")

    loc_frozen = False
    
    if args.stage3_finetune:
        _set_stage3_requires_grad(model)
        logger.info("[V14] STAGE 3 FINETUNE ENABLED: Froze base feature extractors and FLH head. Only training loc_refine blocks and fusion head.")
    elif args.freeze_loc_epochs > 0:
        _set_loc_requires_grad(model, False)
        loc_frozen = True
        logger.info(f"[V14] Localization branch frozen for first {args.freeze_loc_epochs} epochs")
    
    # [V17 Perf] torch.compile 仅在 Linux + Triton 可用时启用（Windows 不支持 Triton）
    import platform
    if platform.system() != 'Windows' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            logger.warning(f"torch.compile failed, fallback to eager: {e}")
    else:
        logger.info("torch.compile skipped (Windows/Triton not available)")
    
    # Optimizer (Typically AdamW for ViTs/ConvNeXts)
    # CLIP is frozen by default inside model __init__, so only Texture + Heads are trained.
    optimizer = build_optimizer(model, args.lr, weight_decay=0.01)
    _warmup = args.warmup_epochs if args.stage3_finetune else 0
    scheduler = build_scheduler(optimizer, args.epochs, eta_min=1e-6, warmup_epochs=_warmup)
    scaler = GradScaler() if args.use_amp else None
    
    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training Loop
    best_acc = 0.0
    
    # Early Stopping setup
    early_stop_patience = int(os.getenv('EARLY_STOP_PATIENCE', 6))
    no_improve_epochs = 0
    early_stop_start_epoch = max(1, args.freeze_loc_epochs + 1)
    if args.freeze_loc_epochs > 0:
        logger.info(
            f"[V14] Early stopping will start after epoch {args.freeze_loc_epochs} "
            f"(monitoring begins at epoch {early_stop_start_epoch})"
        )
    
    # Track statistics
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'is_best': [],
    }
    
    for epoch in range(1, args.epochs + 1):
        # [V14] Unfreeze localization branch after freeze period
        if loc_frozen and epoch > args.freeze_loc_epochs:
            _set_loc_requires_grad(model, True)
            loc_frozen = False
            current_lr = scheduler.get_last_lr()[0]
            remaining_epochs = args.epochs - epoch + 1
            optimizer = build_optimizer(model, current_lr, weight_decay=0.01)
            scheduler = build_scheduler(optimizer, remaining_epochs, eta_min=1e-6)
            no_improve_epochs = 0
            logger.info(
                f"[V14] Localization unfrozen at epoch {epoch}; rebuilt optimizer/scheduler "
                f"with lr={current_lr:.6f}, remaining_epochs={remaining_epochs}"
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, writer, args)
        val_loss, val_acc, val_labels, val_probs, val_preds = validate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step()
        
        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['is_best'].append(False)  # 先占位，下面再覆盖
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0 # Reset counter
            history['is_best'][-1] = True  # 标记当前 epoch 为 best
            logger.info(f"New Best Acc: {best_acc:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, out_dir / 'best.pth')
        else:
            if epoch >= early_stop_start_epoch:
                no_improve_epochs += 1
                logger.info(f"No improvement for {no_improve_epochs} epochs. Best Acc: {best_acc:.4f}")
            else:
                logger.info(
                    f"No improvement at epoch {epoch}, but early stopping is disabled during "
                    f"localization freeze phase (starts at epoch {early_stop_start_epoch})."
                )
            
        # Save Latest
        torch.save(model.state_dict(), out_dir / 'latest.pth')
        
        # Early Stopping Check
        if epoch >= early_stop_start_epoch and no_improve_epochs >= early_stop_patience:
            logger.info(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
            break
        
    writer.close()
    logger.info("Training Complete.")
    
    # --- Generate Plots & History ---
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

        # 1. Save History CSV
        df_history = pd.DataFrame(history)
        df_history.to_csv(out_dir / 'training_history.csv', index=False)
        logger.info(f"Saved training history to {out_dir / 'training_history.csv'}")

        # 2. 训练曲线（带早停点标注 + 学习率副轴）
        best_epochs = [e for e, b in zip(df_history['epoch'], df_history['is_best']) if b]
        early_stop_epoch = df_history['epoch'].iloc[-1] if len(df_history) < args.epochs else None

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(df_history['epoch'], df_history['train_loss'], label='Train Loss', color='steelblue')
        axes[0].plot(df_history['epoch'], df_history['val_loss'], label='Val Loss', color='tomato')
        for be in best_epochs:
            axes[0].axvline(x=be, color='green', linestyle='--', alpha=0.5, label=f'Best @ ep{be}' if be == best_epochs[-1] else None)
        if early_stop_epoch and early_stop_epoch < args.epochs:
            axes[0].axvline(x=early_stop_epoch, color='gray', linestyle=':', alpha=0.8, label=f'Early Stop @ ep{early_stop_epoch}')
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(fontsize=8)

        # Accuracy
        axes[1].plot(df_history['epoch'], df_history['train_acc'], label='Train Acc', color='steelblue')
        axes[1].plot(df_history['epoch'], df_history['val_acc'], label='Val Acc', color='tomato')
        for be in best_epochs:
            axes[1].axvline(x=be, color='green', linestyle='--', alpha=0.5)
        if early_stop_epoch and early_stop_epoch < args.epochs:
            axes[1].axvline(x=early_stop_epoch, color='gray', linestyle=':', alpha=0.8)
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(fontsize=8)

        # Learning Rate
        ax_lr = axes[2]
        ax_lr.plot(df_history['epoch'], df_history['lr'], color='purple', label='Learning Rate')
        ax_lr.set_title('Learning Rate Schedule')
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('LR')
        ax_lr.set_yscale('log')
        ax_lr.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / 'training_curves.png', dpi=150)
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
                plt.savefig(out_dir / 'roc_curve.png', dpi=150)
                plt.close()
                logger.info(f"Saved ROC Curve to {out_dir / 'roc_curve.png'}")

                # 5. PR 曲线（Precision-Recall）
                precision_arr, recall_arr, _ = precision_recall_curve(final_labels, final_probs)
                ap_score = average_precision_score(final_labels, final_probs)
                plt.figure(figsize=(6, 5))
                plt.plot(recall_arr, precision_arr, color='darkorchid', lw=2,
                         label=f'PR curve (AP = {ap_score:.4f})')
                plt.axhline(y=sum(final_labels)/len(final_labels), color='gray',
                            linestyle='--', label='Baseline (random)')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='lower left')
                plt.tight_layout()
                plt.savefig(out_dir / 'pr_curve.png', dpi=150)
                plt.close()
                logger.info(f"Saved PR Curve to {out_dir / 'pr_curve.png'}")

                # 6. 置信度分布直方图（Score Distribution）
                real_probs  = [p for p, l in zip(final_probs, final_labels) if l == 0]
                fake_probs  = [p for p, l in zip(final_probs, final_labels) if l == 1]
                plt.figure(figsize=(8, 5))
                plt.hist(real_probs, bins=40, alpha=0.6, color='steelblue',
                         label=f'Real (n={len(real_probs)})', density=True)
                plt.hist(fake_probs, bins=40, alpha=0.6, color='tomato',
                         label=f'AI Fake (n={len(fake_probs)})', density=True)
                plt.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
                plt.xlabel('P(AI Fake)')
                plt.ylabel('Density')
                plt.title('Score Distribution: Real vs AI Fake')
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / 'score_distribution.png', dpi=150)
                plt.close()
                logger.info(f"Saved Score Distribution to {out_dir / 'score_distribution.png'}")

                # 7. 按类别置信度笱线图（Per-Category Boxplot）
                # 读取 evaluate_by_category 的临时标注文件来区分类别
                try:
                    import collections
                    possible_ann = [
                        'annotation/val_sdxl.txt',
                        'annotation/test_sdxl.txt',
                    ]
                    cat_scores = collections.defaultdict(list)  # cat -> [prob, ...]
                    cat_labels_map = collections.defaultdict(list)
                    for ann_f in possible_ann:
                        if not Path(ann_f).exists():
                            continue
                        with open(ann_f, encoding='utf-8') as _f:
                            _lines = _f.readlines()
                        for _line in _lines:
                            _line = _line.strip()
                            if not _line: continue
                            _parts = _line.rsplit('\t', 1) if '\t' in _line else _line.rsplit(None, 1)
                            if len(_parts) != 2: continue
                            _p, _l = _parts[0], int(_parts[1])
                            _pn = _p.lower().replace('\\', '/')
                            if 'doubao' in _pn and _l == 0:
                                _cat = 'Doubao_Real'
                            elif 'doubao' in _pn and _l == 1:
                                _cat = 'Doubao_Fake'
                            elif 'sdxl' in _pn:
                                _cat = 'SDXL'
                            elif 'flux' in _pn:
                                _cat = 'Flux'
                            else:
                                _cat = 'Real'
                            cat_labels_map[_cat].append(_l)

                    # 对应 val_loader 中的顺序提取每个样本的 prob
                    # 由于 val_loader shuffle=False，所以 final_probs 和 val_dataset 的 index 是对齐的
                    # 这里我们简化处理：拿 val_dataset 的路径列表匹配
                    all_paths_in_val = []
                    for _idx in range(len(val_dataset.test_list)):
                        _imgpath, _ = val_dataset.test_list[_idx]
                        all_paths_in_val.append(_imgpath.lower().replace('\\', '/'))

                    path_cat_scores = collections.defaultdict(list)
                    for _idx, (_prob, _path_n) in enumerate(zip(final_probs, all_paths_in_val)):
                        if 'doubao' in _path_n:
                            _lbl_here = val_dataset.test_list[_idx][1]
                            _cat = 'Doubao_Real' if _lbl_here == 0 else 'Doubao_Fake'
                        elif 'sdxl' in _path_n:
                            _cat = 'SDXL'
                        elif 'flux' in _path_n:
                            _cat = 'Flux'
                        else:
                            _cat = 'Real'
                        path_cat_scores[_cat].append(_prob)

                    if path_cat_scores:
                        _cats_sorted = ['Real', 'Doubao_Real', 'SDXL', 'Flux', 'Doubao_Fake']
                        _box_data = [path_cat_scores.get(c, [0]) for c in _cats_sorted]
                        _colors = ['steelblue', 'dodgerblue', 'tomato', 'salmon', 'firebrick']
                        fig_box, ax_box = plt.subplots(figsize=(10, 5))
                        bp = ax_box.boxplot(_box_data, patch_artist=True, notch=False,
                                            medianprops=dict(color='black', lw=2))
                        for patch, color in zip(bp['boxes'], _colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        ax_box.set_xticklabels([f"{c}\n(n={len(path_cat_scores.get(c,[]))})" for c in _cats_sorted])
                        ax_box.axhline(y=0.5, color='black', linestyle='--', lw=1.2, label='Threshold=0.5')
                        ax_box.set_ylabel('P(AI Fake)')
                        ax_box.set_title('Per-Category Score Distribution (Boxplot)')
                        ax_box.legend(fontsize=9)
                        plt.tight_layout()
                        plt.savefig(out_dir / 'per_category_boxplot.png', dpi=150)
                        plt.close()
                        logger.info(f"Saved Per-Category Boxplot to {out_dir / 'per_category_boxplot.png'}")
                except Exception as _e:
                    logger.warning(f"Per-category boxplot failed: {_e}")

                # 8. Bad Cases Visualisation (original #5)
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
