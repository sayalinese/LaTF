import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.dataset_segformer import SegFormerForgeryDataset
from script.train_segformer import build_model


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
    parser = argparse.ArgumentParser(description='Evaluate SegFormer localization model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--ann_file', default=str(PROJECT_ROOT / 'annotation' / 'val_seg.txt'))
    parser.add_argument('--model_name', default='nvidia/segformer-b2-finetuned-ade-512-512')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--use_ssfr', action='store_true')
    parser.add_argument('--ssfr_map_file', default=str(PROJECT_ROOT / 'dift.pt' / 'ann.txt'))
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = PROJECT_ROOT / 'data'
    mask_dirs = [data_root / 'change' / 'masks', data_root / 'doubao' / 'masks']

    dataset = SegFormerForgeryDataset(
        args.ann_file,
        img_size=args.img_size,
        mask_dirs=mask_dirs,
        is_train=False,
        use_ssfr=args.use_ssfr,
        ssfr_map_file=args.ssfr_map_file,
    )
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
    print(f'Mode: {'RGB+SSFR (10ch)' if args.use_ssfr else 'RGB (3ch)'}')

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

            accumulate_metrics(logits, masks, stats)
            current = finalize_metrics(stats)
            progress.set_postfix(
                pixel_dice=f"{current['pixel_dice']:.4f}",
                sample_dice='-' if current['sample_dice'] is None else f"{current['sample_dice']:.4f}",
            )

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


if __name__ == '__main__':
    main()