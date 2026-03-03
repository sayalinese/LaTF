# PowerShell script for LaRE extraction

# Example datasets to process
$strings = @(
    "val_mj", "val_imagenet_mj", "val_adm", "val_imagenet_adm", 
    "val_biggan", "val_imagenet_biggan", "val_glide", "val_imagenet_glide", 
    "val_sd1d4", "val_imagenet_sd1d4", "val_vqdm", "val_imagenet_vqdm", 
    "val_wukong", "val_imagenet_wukong"
)

foreach ($string in $strings) {
    python 2_特征提取.py `
        --input_path "D:/datasets/GenImage/anns/${string}.txt" `
        --output_path "D:/outputs/${string}" `
        --t 200 `
        --prompt "a photo" `
        --ensemble_size 4 `
        --pretrained_model_name_or_path "D:/pretrained_models/stable-diffusion-v1-5" `
        --img_size 256 256 `
        --use_prompt_template `
        --n-gpus 1
}
