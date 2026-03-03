import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

from service.lare_extractor_module import LareExtractor
from service.processors import SDXLProcessor, GenImageProcessor
from service.feature_extractors import MultiFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Extract LaRE features')
    parser.add_argument('--input_path', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--output_path', type=str, default='dift.pt', help='Output directory')
    parser.add_argument('--dataset_type', type=str, default='auto', choices=['auto', 'genimage', 'sdxl'])
    parser.add_argument('--batch_size', type=int, default=int(os.getenv('EXTRACT_BATCH_SIZE', '8')))
    parser.add_argument('--model_type', type=str, default='sdxl', choices=['sdxl', 'sd21', 'sd15'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--timestep', type=int, default=280)
    parser.add_argument('--fp16', action='store_true', help='Use FP16 for extraction (default FP32 for stability)')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 for extraction (Recommended for RTX 30/40 series)')
    parser.add_argument('--save_aux_features', action='store_true', help='Also extract handcrafted aux features (freq/noise/edge)')
    parser.add_argument('--aux_size', type=int, default=28, help='Spatial size for aux features')
    args = parser.parse_args()

    # Setup
    output_root = Path(args.output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Load Annotations
    if not Path(args.input_path).exists():
        print(f"Error: Input file {args.input_path} not found")
        return

    # Try utf-8 first, fallback to gb18030 (superset of gbk/gb2312) which handles Chinese Windows paths better
    # Or just ignore errors to be robust
    try:
        with open(args.input_path, encoding='utf-8') as f:
            input_infos = f.readlines()
    except UnicodeDecodeError:
        print("UTF-8 decode failed, trying gb18030...")
        try:
            with open(args.input_path, encoding='gb18030') as f:
                input_infos = f.readlines()
        except UnicodeDecodeError:
             print("GB18030 failed, trying with errors='ignore'...")
             with open(args.input_path, encoding='utf-8', errors='ignore') as f:
                input_infos = f.readlines()
    
    if not input_infos:
        print("Empty input file")
        return

    # Determine Processor
    if args.dataset_type == 'auto':
        first_line = input_infos[0]
        if '\t' in first_line or 'sdxl' in args.input_path.lower():
            args.dataset_type = 'sdxl'
        else:
            args.dataset_type = 'genimage'
    
    processor = SDXLProcessor() if args.dataset_type == 'sdxl' else GenImageProcessor()
    print(f"Dataset: {args.dataset_type}, Model: {args.model_type}")

    # Initialize Extractor
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Using dtype: {dtype}")
    extractor = LareExtractor(device=args.device, model_type=args.model_type, dtype=dtype)
    aux_extractor = None
    if args.save_aux_features:
        aux_extractor = MultiFeatureExtractor(target_size=args.aux_size).to(args.device)
        aux_extractor.eval()
    
    # Processing Loop
    processed_infos = []
    batch_infos = []
    
    for i, info in enumerate(tqdm(input_infos)):
        batch_infos.append(info)
        
        if len(batch_infos) >= args.batch_size or i == len(input_infos) - 1:
            # Prepare Batch
            images = []
            valid_batch_infos = []
            
            for b_info in batch_infos:
                try:
                    if args.dataset_type == 'sdxl':
                        img_path, label, fname, _ = processor(b_info)
                    else:
                        img_path, label, fname, _ = processor(b_info, use_full_name=False)
                    
                    # Create Save Path
                    folder_name = Path(img_path).parent.name
                    save_dir = output_root / folder_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / (Path(fname).stem + '.pt')
                    
                    aux_path = save_path.with_name(save_path.stem + '_aux.pt') if args.save_aux_features else None

                    if save_path.exists() and (not aux_path or aux_path.exists()):
                        # Skip if exists
                        info_line = f"{str(img_path).replace('\\', '/')}\t{str(save_path).replace('\\', '/')}\t{label}"
                        if aux_path:
                            info_line += f"\t{str(aux_path).replace('\\', '/')}"
                        info_line += "\n"
                        valid_batch_infos.append({
                            'save_path': save_path,
                            'aux_path': aux_path,
                            'label': label,
                            'info_line': info_line,
                            'exists': True
                        })
                        continue

                    # Load Image
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    info_line = f"{str(img_path).replace('\\', '/')}\t{str(save_path).replace('\\', '/')}\t{label}"
                    if aux_path:
                        info_line += f"\t{str(aux_path).replace('\\', '/')}"
                    info_line += "\n"
                    valid_batch_infos.append({
                        'save_path': save_path,
                        'aux_path': aux_path,
                        'label': label,
                        'info_line': info_line,
                        'exists': False
                    })
                except Exception as e:
                    print(f"Skipping {b_info.strip()[:50]}...: {e}")

            # Process valid new images
            if images:
                try:
                    # Preprocess batch
                    img_tensors = torch.stack([extractor.img_transform(img) for img in images]).to(extractor.device).to(extractor.dtype)

                    # Run Model
                    features = extractor.extract_batch(img_tensors, timestep=args.timestep) # [B, 4, 32, 32]
                    aux_features = None
                    if aux_extractor is not None:
                        # Force FP32 for fft stability regardless of LaRE dtype
                        with torch.no_grad():
                            aux_features = aux_extractor(img_tensors.float())
                    
                    # Save features
                    feat_idx = 0
                    for meta in valid_batch_infos:
                        if not meta['exists']:
                            torch.save(features[feat_idx].cpu(), meta['save_path'])
                            if aux_features is not None and meta.get('aux_path'):
                                torch.save(aux_features[feat_idx].cpu(), meta['aux_path'])
                            feat_idx += 1
                            
                except Exception as e:
                    print(f"Batch failed: {e}")
                    # Fallback: Save zeros for failed items
                    for meta in valid_batch_infos:
                        if not meta['exists']:
                            torch.save(torch.zeros(4, 32, 32), meta['save_path'])
                            if meta.get('aux_path'):
                                torch.save(torch.zeros(3, args.aux_size, args.aux_size), meta['aux_path'])

            # Collect info lines
            for meta in valid_batch_infos:
                processed_infos.append(meta['info_line'])
            
            batch_infos = []

    # Save Annotation Map
    ann_path = output_root / 'ann.txt'
    with open(ann_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_infos)
    
    print(f"Done. Features saved to {output_root}")
    print(f"Map file: {ann_path}")

if __name__ == '__main__':
    main()

