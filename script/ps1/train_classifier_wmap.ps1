# PowerShell script for training classifier

python 3_模型训练.py `
    --model CLipClassifierWMapV6 `
    --train_file annotation/train_sdv5.txt `
    --num_class 2 `
    --val_ratio 0 `
    --val_file annotation/val_sdv5_12k.txt `
    --test_file "" `
    --isTrain 1 `
    --lr 0.0001 `
    --resume "" `
    --data_size 224 `
    --batch_size 48 `
    --test_batch_size 40 `
    --out_dir ./outputs/saved_models `
    --no_strong_aug `
    --exp_name SDV5_MapV5 `
    --clip_type RN50 `
    --gpu 0 `
    --world-size 1 `
    --rank 0 `
    --val_method con
