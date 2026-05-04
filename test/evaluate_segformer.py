import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.dataset_segformer import SegFormerForgeryDataset
from script.train_segformer import build_model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_annotation_paths(ann_file):
    paths = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sep = '\t' if '\t' in line else ' '
            parts = line.rsplit(sep, 1)
            if len(parts) == 2:
                paths.append(parts[0])
    return paths


def denormalize_rgb(image_tensor):
    rgb = image_tensor[:3].detach().cpu().numpy().transpose(1, 2, 0)
    rgb = (rgb * IMAGENET_STD) + IMAGENET_MEAN
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def overlay_mask(image_rgb, mask, color=(255, 80, 80), alpha=0.45):
    out = image_rgb.copy()
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return out

    tint = np.zeros_like(out, dtype=np.uint8)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    out[mask_bool] = ((1 - alpha) * out[mask_bool] + alpha * tint[mask_bool]).astype(np.uint8)
    return out


def make_visualization_panel(image_rgb, pred_mask, gt_mask):
    pred_overlay = overlay_mask(image_rgb, pred_mask, color=(255, 80, 80), alpha=0.45)
    gt_overlay = overlay_mask(image_rgb, gt_mask, color=(80, 255, 80), alpha=0.45)

    # 三联图：输入假图 / 预测叠加 / GT mask 叠加
    return np.concatenate([image_rgb, pred_overlay, gt_overlay], axis=1)


def accumulate_metrics(logits, masks, stats):
    preds = logits.argmax(dim=1)
    preds_bool = preds.bool()
    masks_bool = masks.bool()

    tp = torch.logical_and(preds_bool, masks_bool).sum().item()
    fp = torch.logical_and(preds_bool, ~masks_bool).sum().item()
    fn = torch.logical_and(~preds_bool, masks_bool).sum().item()
    tn = torch.logical_and(~preds_bool, ~masks_bool).sum().item()

    stats['tp'] += tp
    stats['fp'] += fp
    stats['fn'] += fn
    stats['tn'] += tn

    batch_size = masks.shape[0]
    for index in range(batch_size):
        pred_i = preds_bool[index]
        mask_i = masks_bool[index]
        if not mask_i.any():
            continue

        tp_i = torch.logical_and(pred_i, mask_i).sum().item()
        fp_i = torch.logical_and(pred_i, ~mask_i).sum().item()
        fn_i = torch.logical_and(~pred_i, mask_i).sum().item()
        eps = 1e-6
        dice_i = (2.0 * tp_i + eps) / (2.0 * tp_i + fp_i + fn_i + eps)
        iou_i = (tp_i + eps) / (tp_i + fp_i + fn_i + eps)
        precision_i = (tp_i + eps) / (tp_i + fp_i + eps)
        recall_i = (tp_i + eps) / (tp_i + fn_i + eps)

        stats['sample_dice_sum'] += dice_i
        stats['sample_iou_sum'] += iou_i
        stats['sample_precision_sum'] += precision_i
        stats['sample_recall_sum'] += recall_i
        stats['positive_samples'] += 1


def finalize_metrics(stats):
    eps = 1e-6
    tp = stats['tp']
    fp = stats['fp']
    fn = stats['fn']
    tn = stats['tn']
    positive_samples = max(stats['positive_samples'], 1)

    return {
        'pixel_dice': (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        'pixel_iou': (tp + eps) / (tp + fp + fn + eps),
        'pixel_precision': (tp + eps) / (tp + fp + eps),
        'pixel_recall': (tp + eps) / (tp + fn + eps),
        'pixel_specificity': (tn + eps) / (tn + fp + eps),
        'sample_dice': stats['sample_dice_sum'] / positive_samples if stats['positive_samples'] else None,
        'sample_iou': stats['sample_iou_sum'] / positive_samples if stats['positive_samples'] else None,
        'sample_precision': stats['sample_precision_sum'] / positive_samples if stats['positive_samples'] else None,
        'sample_recall': stats['sample_recall_sum'] / positive_samples if stats['positive_samples'] else None,
        'positive_samples': stats['positive_samples'],
    }


def main():
    class Args:
        model = "outputs/segformer_rgb/best.pth"
        ann_file = str(PROJECT_ROOT / 'annotation' / 'eval_brgen_stuff_val.txt')
        model_name = 'nvidia/segformer-b2-finetuned-ade-512-512'
        img_size = 512
        batch_size = 12
        workers = 0
        use_ssfr = False
        ssfr_map_file = str(PROJECT_ROOT / 'dift.pt' / 'ann.txt')
        out_dir = ''
        save_visualizations = False
        max_visualizations = 30
        viz_all_samples = False
    
    args = Args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = PROJECT_ROOT / 'data'
    mask_dirs = [
        data_root / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff' / 'COCO',
        data_root / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff' / 'ImageNet',
        data_root / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff' / 'Places',
        data_root / '新数据' / 'BRGen_Stuff_Val' / 'Gt',
        # legacy fallback
        data_root / 'change' / 'masks',
        data_root / 'doubao' / 'masks',
    ]

    dataset = SegFormerForgeryDataset(
        args.ann_file,
        img_size=args.img_size,
        mask_dirs=mask_dirs,
        is_train=False,
        use_ssfr=args.use_ssfr,
        ssfr_map_file=args.ssfr_map_file,
    )
    ann_paths = parse_annotation_paths(args.ann_file)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == 'cuda',
        persistent_workers=args.workers > 0,
    )

    num_channels = 10 if args.use_ssfr else 3
    model = build_model(args.model_name, num_channels=num_channels, device=device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    out_dir = None
    summary_path = None
    viz_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.save_visualizations:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = PROJECT_ROOT / 'test' / 'outputs' / f'segformer_eval_{stamp}'

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / 'summary.json'

    if args.save_visualizations:
        if out_dir is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = PROJECT_ROOT / 'test' / 'outputs' / f'segformer_eval_{stamp}'
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = out_dir / 'summary.json'
        viz_dir = out_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'tp': 0.0,
        'fp': 0.0,
        'fn': 0.0,
        'tn': 0.0,
        'sample_dice_sum': 0.0,
        'sample_iou_sum': 0.0,
        'sample_precision_sum': 0.0,
        'sample_recall_sum': 0.0,
        'positive_samples': 0,
    }

    print(f'Device: {device}')
    print(f'Checkpoint: {args.model}')
    print(f'Annotations: {args.ann_file}')
    print(f"Mode: {'RGB+SSFR (10ch)' if args.use_ssfr else 'RGB (3ch)'}")
    if out_dir is not None:
        print(f'Output dir: {out_dir}')

    saved_visualizations = 0
    seen_samples = 0

    with torch.no_grad():
        progress = tqdm(loader, desc='Eval', dynamic_ncols=True, ascii=True)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(pixel_values=images)
            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

            preds = logits.argmax(dim=1)

            accumulate_metrics(logits, masks, stats)
            current = finalize_metrics(stats)
            progress.set_postfix(
                pixel_dice=f"{current['pixel_dice']:.4f}",
                sample_dice='-' if current['sample_dice'] is None else f"{current['sample_dice']:.4f}",
            )

            if args.save_visualizations and saved_visualizations < args.max_visualizations:
                images_cpu = images.detach().cpu()
                masks_cpu = masks.detach().cpu().numpy().astype(np.uint8)
                preds_cpu = preds.detach().cpu().numpy().astype(np.uint8)

                for i in range(images_cpu.shape[0]):
                    if saved_visualizations >= args.max_visualizations:
                        break

                    gt_mask = masks_cpu[i]
                    if (not args.viz_all_samples) and (gt_mask.max() == 0):
                        continue

                    img_rgb = denormalize_rgb(images_cpu[i])
                    pred_mask = preds_cpu[i]
                    panel = make_visualization_panel(img_rgb, pred_mask, gt_mask)

                    sample_idx = seen_samples + i
                    stem = f'sample_{sample_idx:06d}'
                    if sample_idx < len(ann_paths):
                        stem = Path(ann_paths[sample_idx]).stem

                    save_path = viz_dir / f'{saved_visualizations:03d}_{stem}.png'
                    cv2.imwrite(str(save_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    saved_visualizations += 1

            seen_samples += images.shape[0]

    result = finalize_metrics(stats)
    print('\n' + '=' * 68)
    print(f"Positive samples:  {result['positive_samples']}")
    print(f"Pixel Dice:        {result['pixel_dice']:.4f}")
    print(f"Pixel IoU:         {result['pixel_iou']:.4f}")
    print(f"Pixel Precision:   {result['pixel_precision']:.4f}")
    print(f"Pixel Recall:      {result['pixel_recall']:.4f}")
    print(f"Pixel Specificity: {result['pixel_specificity']:.4f}")
    if result['sample_dice'] is not None:
        print(f"Sample Dice:       {result['sample_dice']:.4f}")
        print(f"Sample IoU:        {result['sample_iou']:.4f}")
        print(f"Sample Precision:  {result['sample_precision']:.4f}")
        print(f"Sample Recall:     {result['sample_recall']:.4f}")
    print('=' * 68)

    if summary_path is not None:
        summary = {
            'checkpoint': str(args.model),
            'annotation_file': str(args.ann_file),
            'mode': 'RGB+SSFR (10ch)' if args.use_ssfr else 'RGB (3ch)',
            'metrics': result,
            'visualizations_saved': int(saved_visualizations),
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'Summary saved to: {summary_path}')

    if args.save_visualizations:
        print(f'Visualizations saved: {saved_visualizations}')
        print(f'Visualization dir: {viz_dir}')


if __name__ == '__main__':
    main()