import os
import random

# 目标文件夹路径
dir_path = r'd:\三创\LaRE-main\test\修改增加'

# 获取所有文件（假设都是图片）
# [修改] 使用 sorted 确保按文件名顺序排列
files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

# 生成顺序新文件名（保持原扩展名）
new_names = []
for i, old_name in enumerate(files):
    ext = os.path.splitext(old_name)[1]
    # [修改] 移除 shuffle，保持序号对应
    new_names.append(f'{i+1:04d}{ext}')

# 避免重名，先全部改为临时名
temp_names = []
for i, old_name in enumerate(files):
    temp_name = f'temp_{i}{os.path.splitext(old_name)[1]}'
    os.rename(os.path.join(dir_path, old_name), os.path.join(dir_path, temp_name))
    temp_names.append(temp_name)

# 再改为最终新名
for temp_name, new_name in zip(temp_names, new_names):
    os.rename(os.path.join(dir_path, temp_name), os.path.join(dir_path, new_name))

print('重命名完成！')
