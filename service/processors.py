import json
import os
from pathlib import Path

# 获取项目根目录，假设该文件在 service/ 下
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class SDXLProcessor:
    def __call__(self, info):
        # Supports tab and space
        info = info.strip()
        if '\t' in info:
            image_path, label = info.rsplit('\t', 1)
        else:
            image_path, label = info.rsplit(' ', 1)
        
        # [Fix] Handle Chinese Path Mojibake (涓夊垱 -> 三创)
        if '涓夊垱' in image_path:
            image_path = image_path.replace('涓夊垱', '三创')
            
        # [Fix] Fallback: If absolute path fails, try finding relative to project root
        if not os.path.exists(image_path) and 'data/' in image_path:
            # Extract relative path starting from 'data/'
            idx = image_path.find('data/')
            rel_path = image_path[idx:]
            abs_path = PROJECT_ROOT / rel_path
            if abs_path.exists():
                image_path = str(abs_path)

        image_path = image_path.replace('\\', '/')
        label = int(label)
        
        filename = Path(image_path).stem + Path(image_path).suffix
        return image_path, str(label), filename, "ai" if label == 1 else "real"

class GenImageProcessor:
    def __init__(self,
                 file_idx_to_folder='./anns/idx_to_folder.txt',
                 file_idx_to_clsname='./anns/idx_to_clsname.json',
                 filename_to_folder='./anns/filename_to_folder.json'):
        # Only load if files exist, otherwise might be partial usage
        self.valid = False
        if Path(file_idx_to_folder).exists():
            self.label_map_idx_to_folder = {}
            self.label_map_folder_to_idx = {}
            with open(file_idx_to_folder, encoding='utf-8') as f:
                for line in f:
                    cmd, idx, folder = line.strip().split(' ')
                    idx = idx.split('/')[0]
                    folder = folder.split('/')[0]
                    self.label_map_idx_to_folder[idx] = folder
                    self.label_map_folder_to_idx[folder] = idx
            
            self.label_map_idx_to_clsname = json.load(open(file_idx_to_clsname, encoding='utf-8'))
            self.filename_to_folder = json.load(open(filename_to_folder, encoding='utf-8'))

            self.label_map_folder_to_clsname = {}
            for folder, idx in self.label_map_folder_to_idx.items():
                self.label_map_folder_to_clsname[folder] = self.label_map_idx_to_clsname[idx]
            self.valid = True

    def __call__(self, info, use_full_name=False):
        info = info.lstrip('\ufeff').strip()
        if '\t' in info:
            image_path, label = info.split('\t')
        else:
            image_path, label = info.rsplit(' ', 1)

        image_path = image_path.lstrip('\ufeff')
        filename = Path(image_path).name
        
        clsname_full = 'unknown'
        if self.valid:
            folder_or_index = filename.split('_')[0]
            try:
                if len(folder_or_index) <= 3:  # is a index
                    index = str(int(folder_or_index))
                    clsname_full = self.label_map_idx_to_clsname[index]
                elif folder_or_index == 'ILSVRC2012':  # is val image:
                    folder = self.filename_to_folder[filename]
                    clsname_full = self.label_map_folder_to_clsname[folder]
                elif folder_or_index == 'GLIDE':
                    index = str(int(filename.split('_')[4]))
                    clsname_full = self.label_map_idx_to_clsname[index]
                elif folder_or_index == 'VQDM':
                    index = str(int(filename.split('_')[4]))
                    clsname_full = self.label_map_idx_to_clsname[index]
                else:  # is a folder
                    folder = folder_or_index
                    clsname_full = self.label_map_folder_to_clsname[folder]
            except Exception:
                pass

        if use_full_name:
            return image_path, label, filename, clsname_full
        else:
            clsname_simple = clsname_full.split(',')[0].strip()
            return image_path, label, filename, clsname_simple

