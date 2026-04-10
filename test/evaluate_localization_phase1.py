"""
Phase 1: Zero-cost localization improvements — threshold sweep + multi-scale fusion.
No retraining required. Operates on saved model predictions.

Usage:
    python test/evaluate_localization_phase1.py --model outputs/stage3_finetune/best.pth --ann_file annotation/eval_change.txt
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from service.cam_visualizer import overlay_heatmap
from service.dataset import ImageDataset
from service.heatmap_utils import refine_map_for_visibility
from service.model_v11_fusion import LaREDeepFakeV11

# ─── Reuse helpers from evaluate_localization ───
from evaluate_localization import (
    resolve_first_existing, resolve_checkpoint_path,
    load_model_checkpoint, infer_category, build_autocast_context,
    ANN_CANDIDATES, MAP_CANDIDATES,
)

# ─── Metric helpers ───

def pixel_metrics(prob_map: np.ndarray, gt_mask: np.ndarray, threshold: float):
    pred = prob_map >= threshold
    gt = gt_mask >= 0.5
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    eps = 1e-6
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall,
            'gt_positive': bool(gt.any())}


def aggregate_pixel_metrics(all_metrics):
    """Aggregate pixel-level metrics across samples (micro-average)."""
    eps = 1e-6
    tp = sum(m['tp'] for m in all_metrics)
    fp = sum(m['fp'] for m in all_metrics)
    fn = sum(m['fn'] for m in all_metrics)
    tn = sum(m['tn'] for m in all_metrics)
    pixel_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    pixel_iou = (tp + eps) / (tp + fp + fn + eps)
    pixel_prec = (tp + eps) / (tp + fp + eps)
    pixel_recall = (tp + eps) / (tp + fn + eps)
    # sample-average
    positive = [m for m in all_metrics if m['gt_positive']]
    sample_dice = np.mean([m['dice'] for m in positive]) if positive else 0.0
    sample_iou = np.mean([m['iou'] for m in positive]) if positive else 0.0
    return {
        'pixel_dice': float(pixel_dice), 'pixel_iou': float(pixel_iou),
        'pixel_precision': float(pixel_prec), 'pixel_recall': float(pixel_recall),
        'sample_dice': float(sample_dice), 'sample_iou': float(sample_iou),
        'positive_count': len(positive),
    }


# ─── Multi-scale fusion strategies ───

FUSION_STRATEGIES = {
    'coarse32': lambda probs: probs['coarse32'],
    'mid64': lambda probs: probs['mid64'],
    'fine128': lambda probs: probs['fine128'],
    'avg_equal': lambda probs: (probs['coarse32'] + probs['mid64'] + probs['fine128']) / 3.0,
    'avg_weighted': lambda probs: 0.15 * probs['coarse32'] + 0.30 * probs['mid64'] + 0.55 * probs['fine128'],
    'max_fusion': lambda probs: np.maximum(np.maximum(probs['coarse32'], probs['mid64']), probs['fine128']),
}


def fuse_to_target(probs_dict, target_size=128):
    """Upsample all scales to target_size and return as dict."""
    result = {}
    for name, prob in probs_dict.items():
        if prob.shape[0] != target_size or prob.shape[1] != target_size:
            result[name] = cv2.resize(prob, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            result[name] = prob
    return result


# ─── Visualization ───

def save_fusion_panel(image_path, gt_mask, best_pred, best_label, metrics, save_path):
    """Save side-by-side: original | GT mask | best prediction overlay."""
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image_np = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image_np is None:
        return
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_np)
    w, h = image_pil.size

    gt_resized = cv2.resize(gt_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    gt_overlay = overlay_heatmap(image_pil, gt_resized, alpha=0.45)

    pred_resized = cv2.resize(best_pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    refined = refine_map_for_visibility(pred_resized, image_np).astype(np.float32) / 255.0
    pred_overlay = overlay_heatmap(image_pil, refined, alpha=0.45)

    def add_caption(img, text):
        canvas = Image.new('RGB', (img.width, img.height + 30), color=(18, 18, 18))
        canvas.paste(img, (0, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), text, fill=(245, 245, 245), font=ImageFont.load_default())
        return canvas

    tiles = [
        add_caption(image_pil, 'Original'),
        add_caption(gt_overlay, 'GT Mask'),
        add_caption(pred_overlay, f'{best_label} d={metrics["dice"]:.3f} i={metrics["iou"]:.3f}'),
    ]
    panel_w = sum(t.width for t in tiles)
    panel_h = max(t.height for t in tiles)
    panel = Image.new('RGB', (panel_w, panel_h), (8, 8, 8))
    x = 0
    for t in tiles:
        panel.paste(t, (x, 0))
        x += t.width
    save_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(save_path)


# ─── Main ───

def parse_args():
    p = argparse.ArgumentParser(description='Phase 1: Threshold sweep + multi-scale fusion')
    p.add_argument('--model', type=str, default='')
    p.add_argument('--ann_file', type=str, default='')
    p.add_argument('--map_file', type=str, default='')
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--out_dir', type=str, default='outputs/phase1_eval')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--highres_size', type=int, default=512)
    p.add_argument('--clip_type', type=str, default='RN50x64')
    p.add_argument('--texture_model', type=str, default='convnext_tiny')
    p.add_argument('--max_visualizations', type=int, default=30)
    p.add_argument('--flh_weights', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    ann_file = Path(args.ann_file) if args.ann_file else resolve_first_existing(ANN_CANDIDATES, 'annotation')
    map_file = Path(args.map_file) if args.map_file else resolve_first_existing(
        [os.getenv('MAP_FILE', ''), 'dift.pt/ann.txt'], 'map file')
    ckpt_path = resolve_checkpoint_path(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Annotations: {ann_file}')

    # ── Dataset ──
    dataset = ImageDataset(
        data_root=args.data_root,
        train_file=str(ann_file),
        map_file=str(map_file),
        data_size=448,
        highres_size=args.highres_size,
        enable_v11=True,
        enable_luma=True,
        is_train=False,
        drop_no_map=False,
    )
    dataset.set_val_mode(True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )

    # ── Model ──
    model = LaREDeepFakeV11(num_classes=2, clip_type=args.clip_type, texture_model=args.texture_model)
    load_model_checkpoint(model, ckpt_path, flh_weights_path=args.flh_weights or None)
    model.to(device).eval()

    # ── Collect all predictions ──
    print('\n[1/3] Collecting raw predictions from model ...')
    all_samples = []  # list of dicts: {probs: {scale: np}, gt: np, path: str, label: int}

    with torch.no_grad():
        for batch in loader:
            img_clip = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            loss_maps = batch[2].to(device, non_blocking=True)
            img_highres = batch[3].to(device, non_blocking=True)
            masks = batch[4].to(device, non_blocking=True)
            luma = batch[5].to(device, non_blocking=True)
            paths = batch[-1]

            with build_autocast_context(device):
                logits, seg_outputs = model(
                    img_clip, img_highres, loss_maps,
                    luma_map=luma, return_seg_pyramid=True,
                )

            coarse, mid, fine = seg_outputs
            for i in range(len(paths)):
                probs = {
                    'coarse32': torch.sigmoid(coarse[i, 0].float()).cpu().numpy(),
                    'mid64': torch.sigmoid(mid[i, 0].float()).cpu().numpy(),
                    'fine128': torch.sigmoid(fine[i, 0].float()).cpu().numpy(),
                }
                gt = masks[i, 0].cpu().numpy()
                all_samples.append({
                    'probs': probs,
                    'gt_128': cv2.resize(gt.astype(np.float32), (128, 128), interpolation=cv2.INTER_NEAREST),
                    'gt_full': gt,
                    'path': paths[i],
                    'label': int(labels[i].item()),
                    'category': infer_category(paths[i], int(labels[i].item())),
                })

    print(f'  Collected {len(all_samples)} samples.')

    # ── Threshold sweep × fusion strategy ──
    print('\n[2/3] Sweeping thresholds × fusion strategies ...')
    thresholds = np.arange(0.25, 0.85, 0.05).round(3).tolist()

    results = {}  # {strategy_name: {threshold: aggregated_metrics}}
    best_overall = {'pixel_dice': 0, 'strategy': '', 'threshold': 0}

    for strat_name, strat_fn in FUSION_STRATEGIES.items():
        results[strat_name] = {}
        for thresh in thresholds:
            all_m = []
            for sample in all_samples:
                probs_128 = fuse_to_target(sample['probs'], 128)
                fused = strat_fn(probs_128)
                m = pixel_metrics(fused, sample['gt_128'], thresh)
                all_m.append(m)
            agg = aggregate_pixel_metrics(all_m)
            results[strat_name][thresh] = agg

            if agg['pixel_dice'] > best_overall['pixel_dice']:
                best_overall = {
                    'pixel_dice': agg['pixel_dice'],
                    'pixel_iou': agg['pixel_iou'],
                    'pixel_precision': agg['pixel_precision'],
                    'pixel_recall': agg['pixel_recall'],
                    'sample_dice': agg['sample_dice'],
                    'sample_iou': agg['sample_iou'],
                    'strategy': strat_name,
                    'threshold': thresh,
                }

    # ── Print compact table ──
    print('\n' + '=' * 105)
    print(f'{"Strategy":<16} {"Thresh":>6} | {"pxDice":>8} {"pxIoU":>8} {"pxPrec":>8} {"pxRecall":>8} | {"smDice":>8} {"smIoU":>8}')
    print('-' * 105)

    for strat_name in FUSION_STRATEGIES:
        # Find best threshold for this strategy
        best_t = max(results[strat_name], key=lambda t: results[strat_name][t]['pixel_dice'])
        m = results[strat_name][best_t]
        marker = ' <<<' if strat_name == best_overall['strategy'] and best_t == best_overall['threshold'] else ''
        print(f'{strat_name:<16} {best_t:>6.2f} | {m["pixel_dice"]:>8.4f} {m["pixel_iou"]:>8.4f} '
              f'{m["pixel_precision"]:>8.4f} {m["pixel_recall"]:>8.4f} | '
              f'{m["sample_dice"]:>8.4f} {m["sample_iou"]:>8.4f}{marker}')
    print('=' * 105)

    # Print full threshold sweep for the best strategy
    best_strat = best_overall['strategy']
    print(f'\nFull threshold sweep for best strategy: [{best_strat}]')
    print(f'{"Thresh":>8} | {"pxDice":>8} {"pxIoU":>8} {"pxPrec":>8} {"pxRecall":>8} | {"smDice":>8} {"smIoU":>8}')
    print('-' * 80)
    for t in thresholds:
        m = results[best_strat][t]
        marker = ' <<<' if t == best_overall['threshold'] else ''
        print(f'{t:>8.3f} | {m["pixel_dice"]:>8.4f} {m["pixel_iou"]:>8.4f} '
              f'{m["pixel_precision"]:>8.4f} {m["pixel_recall"]:>8.4f} | '
              f'{m["sample_dice"]:>8.4f} {m["sample_iou"]:>8.4f}{marker}')

    print(f'\n>>> BEST: strategy={best_overall["strategy"]}, threshold={best_overall["threshold"]:.3f}')
    print(f'    pixel_dice={best_overall["pixel_dice"]:.4f}  pixel_iou={best_overall["pixel_iou"]:.4f}')
    print(f'    pixel_prec={best_overall["pixel_precision"]:.4f}  pixel_recall={best_overall["pixel_recall"]:.4f}')
    print(f'    sample_dice={best_overall["sample_dice"]:.4f}  sample_iou={best_overall["sample_iou"]:.4f}')

    # Comparison with baseline
    baseline = results['fine128'].get(0.5, results['fine128'][thresholds[0]])
    if 0.5 in results['fine128']:
        baseline = results['fine128'][0.5]
    print(f'\n>>> BASELINE (fine128 @ 0.5): pixel_dice={baseline["pixel_dice"]:.4f}  pixel_iou={baseline["pixel_iou"]:.4f}')
    delta_dice = best_overall['pixel_dice'] - baseline['pixel_dice']
    delta_iou = best_overall['pixel_iou'] - baseline['pixel_iou']
    print(f'>>> IMPROVEMENT: pixel_dice +{delta_dice:.4f} ({delta_dice/max(baseline["pixel_dice"],1e-6)*100:.1f}%)'
          f'  pixel_iou +{delta_iou:.4f} ({delta_iou/max(baseline["pixel_iou"],1e-6)*100:.1f}%)')

    # ── Save visualizations with best config ──
    print(f'\n[3/3] Saving visualizations with best config ({best_strat} @ {best_overall["threshold"]:.3f}) ...')
    viz_dir = out_dir / 'visualizations'
    best_fn = FUSION_STRATEGIES[best_strat]
    best_thresh = best_overall['threshold']
    saved = 0

    for sample in all_samples:
        if saved >= args.max_visualizations:
            break
        if not (sample['gt_full'] >= 0.5).any():
            continue
        probs_128 = fuse_to_target(sample['probs'], 128)
        fused = best_fn(probs_128)
        m = pixel_metrics(fused, sample['gt_128'], best_thresh)
        safe_stem = Path(sample['path']).stem.replace(' ', '_')
        save_fusion_panel(
            sample['path'], sample['gt_full'], fused,
            f'{best_strat}@{best_thresh:.2f}', m,
            viz_dir / f'{saved:03d}_{sample["category"]}_{safe_stem}.png',
        )
        saved += 1

    print(f'  Saved {saved} visualizations to {viz_dir}')

    # ── Save JSON ──
    summary = {
        'checkpoint': str(ckpt_path),
        'annotation_file': str(ann_file),
        'total_samples': len(all_samples),
        'baseline_fine128_t05': baseline,
        'best_config': best_overall,
        'all_results': {
            strat: {str(t): m for t, m in thresh_results.items()}
            for strat, thresh_results in results.items()
        },
    }
    summary_path = out_dir / 'phase1_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  Saved summary to {summary_path}')


if __name__ == '__main__':
    main()
