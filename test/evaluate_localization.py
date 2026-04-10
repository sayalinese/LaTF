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

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

from service.cam_visualizer import overlay_heatmap
from service.dataset import ImageDataset
from service.heatmap_utils import refine_map_for_visibility
from service.model_v11_fusion import LaREDeepFakeV11


ANN_CANDIDATES = [
    'annotation/val_sdxl.txt',
    'annotation/test_sdxl.txt',
    'annotation/temp_eval_Doubao_Fake.txt',
    'annotation/temp_eval_Doubao_Real.txt',
    'annotation/val_doubao_focused.txt',
    'annotation/test_doubao_focused.txt',
]

MAP_CANDIDATES = [
    os.getenv('MAP_FILE', ''),
    'dift.pt/ann.txt',
    'annotation/map_sdxl_train.txt',
]

CKPT_CANDIDATES = [
    'outputs/v17_joint/best.pth',
    'outputs/v13_doubao_focused/best.pth',
    'outputs/v11_fusion/best.pth',
]


def resolve_first_existing(candidates, label):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f'No existing {label} found in candidates: {candidates}')


def resolve_checkpoint_path(manual_value=''):
    if manual_value:
        path = Path(manual_value)
        if path.is_dir():
            path = path / 'best.pth'
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path

    env_out_dir = os.getenv('OUT_DIR', '').strip()
    env_candidate = str(Path(env_out_dir) / 'best.pth') if env_out_dir else ''
    candidates = [env_candidate] + CKPT_CANDIDATES
    return resolve_first_existing(candidates, 'checkpoint file')


def infer_category(image_path, label):
    path_norm = str(image_path).lower().replace('\\', '/')
    if 'doubao' in path_norm:
        return 'Doubao_Real' if label == 0 else 'Doubao_Fake'
    if '/change/' in path_norm:
        return 'Change'
    if '/real/' in path_norm or path_norm.startswith('real/') or label == 0:
        return 'Real'
    if 'sdxl' in path_norm:
        return 'SDXL'
    if 'flux' in path_norm:
        return 'Flux'
    return 'Fake'


def init_loc_bucket():
    return {
        'tp': 0.0,
        'fp': 0.0,
        'fn': 0.0,
        'tn': 0.0,
        'sample_count': 0,
        'positive_samples': 0,
        'empty_samples': 0,
        'dice_sum': 0.0,
        'iou_sum': 0.0,
        'precision_sum': 0.0,
        'recall_sum': 0.0,
        'empty_pred_ratio_sum': 0.0,
        'empty_max_prob_sum': 0.0,
    }


def init_cls_bucket():
    return {
        'correct': 0,
        'total': 0,
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
    }


def compute_sample_metrics(prob_map, gt_mask, threshold):
    pred = prob_map >= threshold
    gt = gt_mask >= 0.5
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    eps = 1e-6
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'gt_positive': bool(gt.any()),
        'pred_ratio': float(pred.mean()),
        'max_prob': float(prob_map.max()),
    }


def update_loc_bucket(bucket, sample_metrics):
    bucket['tp'] += sample_metrics['tp']
    bucket['fp'] += sample_metrics['fp']
    bucket['fn'] += sample_metrics['fn']
    bucket['tn'] += sample_metrics['tn']
    bucket['sample_count'] += 1
    if sample_metrics['gt_positive']:
        bucket['positive_samples'] += 1
        bucket['dice_sum'] += sample_metrics['dice']
        bucket['iou_sum'] += sample_metrics['iou']
        bucket['precision_sum'] += sample_metrics['precision']
        bucket['recall_sum'] += sample_metrics['recall']
    else:
        bucket['empty_samples'] += 1
        bucket['empty_pred_ratio_sum'] += sample_metrics['pred_ratio']
        bucket['empty_max_prob_sum'] += sample_metrics['max_prob']


def finalize_loc_bucket(bucket):
    eps = 1e-6
    tp = bucket['tp']
    fp = bucket['fp']
    fn = bucket['fn']
    tn = bucket['tn']
    positive_samples = max(bucket['positive_samples'], 1)
    empty_samples = max(bucket['empty_samples'], 1)
    return {
        'sample_count': int(bucket['sample_count']),
        'positive_samples': int(bucket['positive_samples']),
        'empty_samples': int(bucket['empty_samples']),
        'pixel_precision': float((tp + eps) / (tp + fp + eps)),
        'pixel_recall': float((tp + eps) / (tp + fn + eps)),
        'pixel_dice': float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)),
        'pixel_iou': float((tp + eps) / (tp + fp + fn + eps)),
        'pixel_specificity': float((tn + eps) / (tn + fp + eps)),
        'sample_dice': float(bucket['dice_sum'] / positive_samples) if bucket['positive_samples'] else None,
        'sample_iou': float(bucket['iou_sum'] / positive_samples) if bucket['positive_samples'] else None,
        'sample_precision': float(bucket['precision_sum'] / positive_samples) if bucket['positive_samples'] else None,
        'sample_recall': float(bucket['recall_sum'] / positive_samples) if bucket['positive_samples'] else None,
        'empty_pred_ratio': float(bucket['empty_pred_ratio_sum'] / empty_samples) if bucket['empty_samples'] else None,
        'empty_max_prob': float(bucket['empty_max_prob_sum'] / empty_samples) if bucket['empty_samples'] else None,
    }


def update_cls_bucket(bucket, pred_label, true_label):
    bucket['total'] += 1
    if pred_label == true_label:
        bucket['correct'] += 1
    if true_label == 1 and pred_label == 1:
        bucket['tp'] += 1
    elif true_label == 0 and pred_label == 0:
        bucket['tn'] += 1
    elif true_label == 0 and pred_label == 1:
        bucket['fp'] += 1
    else:
        bucket['fn'] += 1


def finalize_cls_bucket(bucket):
    eps = 1e-6
    tp = bucket['tp']
    tn = bucket['tn']
    fp = bucket['fp']
    fn = bucket['fn']
    total = max(bucket['total'], 1)
    return {
        'total': int(bucket['total']),
        'accuracy': float(bucket['correct'] / total) if bucket['total'] else None,
        'fake_precision': float((tp + eps) / (tp + fp + eps)),
        'fake_recall': float((tp + eps) / (tp + fn + eps)),
        'real_recall': float((tn + eps) / (tn + fp + eps)),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


FLH_KEY_MAP = {
    'luma_gate.': 'luma_gate_loc.',
    'ch_interact.': 'ch_interact_loc.',
    'stem.': 'loc_stem.',
    'cbam1.': 'loc_cbam1.',
    'cbam2.': 'loc_cbam2.',
    'cbam3.': 'loc_cbam3.',
    'head.': 'loc_head.',
}


def load_model_checkpoint(model, ckpt_path, flh_weights_path=None):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    if flh_weights_path:
        flh_state = torch.load(flh_weights_path, map_location='cpu')
        mapped = {}
        for key, value in flh_state.items():
            new_key = key
            for old_prefix, new_prefix in FLH_KEY_MAP.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            mapped[new_key] = value
        model.load_state_dict(mapped, strict=False)
        print(f'[FLH] Injected {len(mapped)} weights from {flh_weights_path}')


def load_rgb_image(image_path):
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to decode image: {image_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_caption(image, title):
    canvas = Image.new('RGB', (image.width, image.height + 30), color=(18, 18, 18))
    canvas.paste(image, (0, 30))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(245, 245, 245), font=font)
    return canvas


def save_visualization_panel(image_path, gt_mask, pred_maps, metrics_by_scale, cls_info, save_path):
    image_np = load_rgb_image(image_path)
    image_pil = Image.fromarray(image_np)
    width, height = image_pil.size

    gt_resized = cv2.resize(gt_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    gt_overlay = overlay_heatmap(image_pil, gt_resized, alpha=0.45)

    tiles = [
        add_caption(image_pil, f"orig cls={cls_info['pred']} prob={cls_info['prob']:.3f}"),
        add_caption(gt_overlay, 'gt mask'),
    ]

    for scale_name, prob_map in pred_maps.items():
        prob_resized = cv2.resize(prob_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        refined = refine_map_for_visibility(prob_resized, image_np).astype(np.float32) / 255.0
        overlay = overlay_heatmap(image_pil, refined, alpha=0.45)
        metric = metrics_by_scale[scale_name]
        title = f"{scale_name} d={metric['dice']:.3f} i={metric['iou']:.3f}"
        tiles.append(add_caption(overlay, title))

    panel_width = sum(tile.width for tile in tiles)
    panel_height = max(tile.height for tile in tiles)
    panel = Image.new('RGB', (panel_width, panel_height), color=(8, 8, 8))

    offset_x = 0
    for tile in tiles:
        panel.paste(tile, (offset_x, 0))
        offset_x += tile.width

    save_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(save_path)


def build_autocast_context(device):
    if device.type != 'cuda':
        return nullcontext()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return autocast('cuda', dtype=amp_dtype, enabled=True)


def print_summary(summary):
    print('\n' + '=' * 88)
    print('Classification Summary')
    print('=' * 88)
    cls_overall = summary['classification']['overall']
    print(
        f"overall: total={cls_overall['total']} acc={cls_overall['accuracy']:.4f} "
        f"fake_recall={cls_overall['fake_recall']:.4f} real_recall={cls_overall['real_recall']:.4f}"
    )
    for category, category_stats in sorted(summary['classification']['by_category'].items()):
        if category_stats['total'] == 0:
            continue
        print(
            f"{category:<16} total={category_stats['total']:<4} acc={category_stats['accuracy']:.4f} "
            f"fake_recall={category_stats['fake_recall']:.4f} real_recall={category_stats['real_recall']:.4f}"
        )

    print('\n' + '=' * 88)
    print('Localization Summary')
    print('=' * 88)
    for scale_name, scale_stats in summary['localization'].items():
        overall = scale_stats['overall']
        print(
            f"{scale_name:<10} pos={overall['positive_samples']:<4} empty={overall['empty_samples']:<4} "
            f"sample_dice={overall['sample_dice'] if overall['sample_dice'] is not None else 'n/a'} "
            f"sample_iou={overall['sample_iou'] if overall['sample_iou'] is not None else 'n/a'} "
            f"empty_fp={overall['empty_pred_ratio'] if overall['empty_pred_ratio'] is not None else 'n/a'}"
        )
        for category, category_stats in sorted(scale_stats['by_category'].items()):
            if category_stats['sample_count'] == 0:
                continue
            print(
                f"  {category:<14} pos={category_stats['positive_samples']:<4} empty={category_stats['empty_samples']:<4} "
                f"sample_dice={category_stats['sample_dice'] if category_stats['sample_dice'] is not None else 'n/a'} "
                f"sample_iou={category_stats['sample_iou'] if category_stats['sample_iou'] is not None else 'n/a'}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate localization quality for multi-scale forensic head')
    parser.add_argument('--model', type=str, default='', help='Checkpoint path; auto-resolves if omitted')
    parser.add_argument('--ann_file', type=str, default='', help='Annotation file to evaluate; auto-resolves if omitted')
    parser.add_argument('--map_file', type=str, default='', help='Feature map annotation file; auto-resolves if omitted')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='outputs/localization_eval')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=0, help='Limit samples for quick smoke tests')
    parser.add_argument('--max_visualizations', type=int, default=24)
    parser.add_argument('--highres_size', type=int, default=512)
    parser.add_argument('--clip_type', type=str, default='RN50x64')
    parser.add_argument('--texture_model', type=str, default='convnext_tiny')
    parser.add_argument('--flh_weights', type=str, default='', help='Standalone FLH checkpoint to inject into full model')
    return parser.parse_args()


def main():
    args = parse_args()

    ann_file = Path(args.ann_file) if args.ann_file else resolve_first_existing(ANN_CANDIDATES, 'annotation file')
    map_file = Path(args.map_file) if args.map_file else resolve_first_existing(MAP_CANDIDATES, 'map file')
    ckpt_path = resolve_checkpoint_path(args.model)

    if args.ann_file and not ann_file.exists():
        raise FileNotFoundError(f'Annotation file not found: {ann_file}')
    if args.map_file and not map_file.exists():
        raise FileNotFoundError(f'Map file not found: {map_file}')

    out_dir = Path(args.out_dir)
    viz_dir = out_dir / 'visualizations'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Annotations: {ann_file}')
    print(f'Map file: {map_file}')

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
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    model = LaREDeepFakeV11(num_classes=2, clip_type=args.clip_type, texture_model=args.texture_model)
    load_model_checkpoint(model, ckpt_path, flh_weights_path=args.flh_weights or None)
    model.to(device)
    model.eval()

    loc_buckets = {
        'coarse32': {'overall': init_loc_bucket(), 'by_category': defaultdict(init_loc_bucket)},
        'mid64': {'overall': init_loc_bucket(), 'by_category': defaultdict(init_loc_bucket)},
        'fine128': {'overall': init_loc_bucket(), 'by_category': defaultdict(init_loc_bucket)},
    }
    cls_bucket = {'overall': init_cls_bucket(), 'by_category': defaultdict(init_cls_bucket)}

    saved_viz = 0
    total_processed = 0

    with torch.no_grad():
        for batch in loader:
            img_clip = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            loss_maps = batch[2].to(device, non_blocking=True)
            img_highres = batch[3].to(device, non_blocking=True)
            masks = batch[4].to(device, non_blocking=True)
            luma = batch[5].to(device, non_blocking=True)
            image_paths = batch[-1]

            with build_autocast_context(device):
                logits, seg_outputs = model(
                    img_clip,
                    img_highres,
                    loss_maps,
                    luma_map=luma,
                    return_seg_pyramid=True,
                )

            coarse_logits, mid_logits, fine_logits = seg_outputs
            scale_probs = {
                'coarse32': torch.sigmoid(coarse_logits.float()),
                'mid64': torch.sigmoid(mid_logits.float()),
                'fine128': torch.sigmoid(fine_logits.float()),
            }
            scale_masks = {
                scale_name: F.interpolate(masks.float(), size=prob.shape[-2:], mode='nearest')
                for scale_name, prob in scale_probs.items()
            }

            cls_probs = torch.softmax(logits.float(), dim=1)[:, 1]
            cls_preds = torch.argmax(logits.float(), dim=1)

            for idx, image_path in enumerate(image_paths):
                label = int(labels[idx].item())
                pred_label = int(cls_preds[idx].item())
                category = infer_category(image_path, label)
                total_processed += 1

                update_cls_bucket(cls_bucket['overall'], pred_label, label)
                update_cls_bucket(cls_bucket['by_category'][category], pred_label, label)

                sample_metrics_by_scale = {}
                has_positive_mask = bool(masks[idx].sum().item() > 0.5)
                for scale_name in ('coarse32', 'mid64', 'fine128'):
                    prob_map = scale_probs[scale_name][idx, 0].detach().cpu().numpy()
                    gt_mask = scale_masks[scale_name][idx, 0].detach().cpu().numpy()
                    sample_metrics = compute_sample_metrics(prob_map, gt_mask, args.threshold)
                    sample_metrics_by_scale[scale_name] = sample_metrics
                    update_loc_bucket(loc_buckets[scale_name]['overall'], sample_metrics)
                    update_loc_bucket(loc_buckets[scale_name]['by_category'][category], sample_metrics)

                if has_positive_mask and saved_viz < args.max_visualizations:
                    gt_full = masks[idx, 0].detach().cpu().numpy()
                    pred_maps = {
                        scale_name: scale_probs[scale_name][idx, 0].detach().cpu().numpy()
                        for scale_name in ('coarse32', 'mid64', 'fine128')
                    }
                    safe_stem = Path(str(image_path)).stem.replace(' ', '_')
                    viz_path = viz_dir / f'{saved_viz:03d}_{category}_{safe_stem}.png'
                    save_visualization_panel(
                        image_path=image_path,
                        gt_mask=gt_full,
                        pred_maps=pred_maps,
                        metrics_by_scale=sample_metrics_by_scale,
                        cls_info={'pred': pred_label, 'prob': float(cls_probs[idx].item())},
                        save_path=viz_path,
                    )
                    saved_viz += 1

    summary = {
        'checkpoint': str(ckpt_path),
        'annotation_file': str(ann_file),
        'map_file': str(map_file),
        'threshold': args.threshold,
        'processed_samples': total_processed,
        'visualizations_saved': saved_viz,
        'classification': {
            'overall': finalize_cls_bucket(cls_bucket['overall']),
            'by_category': {
                category: finalize_cls_bucket(bucket)
                for category, bucket in sorted(cls_bucket['by_category'].items())
            },
        },
        'localization': {
            scale_name: {
                'overall': finalize_loc_bucket(scale_bucket['overall']),
                'by_category': {
                    category: finalize_loc_bucket(bucket)
                    for category, bucket in sorted(scale_bucket['by_category'].items())
                },
            }
            for scale_name, scale_bucket in loc_buckets.items()
        },
    }

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f'\nSaved summary to: {summary_path}')
    if saved_viz:
        print(f'Saved visualizations to: {viz_dir}')


if __name__ == '__main__':
    main()