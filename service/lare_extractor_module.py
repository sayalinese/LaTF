"""
LaTF (LaRE + TruFor Fusion) - LaRE Feature Extractor Module
Latent Reconstruction Error feature extraction from Diffusion models.

Supports multiple Stable Diffusion backbones:
- SDXL (1024x1024) with Lightning optimization
- SD 2.1 (768x768)
- SD 1.5 (512x512)
"""

import os
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDIMScheduler
from PIL import Image
from torchvision import transforms

class LareExtractor:
    def __init__(self, device="cuda", dtype=torch.float16, model_type="sdxl"):
        """
        LaRE Feature Extractor supporting multiple SD backbones
        
        Args:
            device: 'cuda' or 'cpu'
            dtype: torch.float16 or torch.float32
            model_type: 'sd15', 'sdxl', or 'sd21'
        """
        self.device = device
        self.dtype = dtype
        self.model_type = model_type
        
        # Model configurations
        if model_type == "sdxl":
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.resolution = 1024  # SDXL native resolution
            self.vae_scale = 0.13025  # SDXL uses different scale
            print(f"[LaRE] Using SDXL Lightning backbone (ByteDance 4-step, 1024x1024)")
        elif model_type == "sd21":
            model_id = "stabilityai/stable-diffusion-2-1"
            self.resolution = 768
            self.vae_scale = 0.18215
            print(f"[LaRE] Using SD 2.1 backbone (768x768)")
        else:  # sd15
            model_id = "runwayml/stable-diffusion-v1-5"
            self.resolution = 512
            self.vae_scale = 0.18215
            print(f"[LaRE] Using SD 1.5 backbone (512x512)")
            
        print(f"[LaRE] Loading components from {model_id}...")
        
        try:
            # Force VAE to float32 to avoid NaN/Overflow issues common in SDXL VAE with FP16/BF16
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            
            # Load Base UNet first
            print(f"[LaRE] Loading Base UNet from {model_id}...")
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

            if model_type == "sdxl":
                # Load Lightning Weights manually
                print("[LaRE] Downloading & Overlaying SDXL-Lightning weights...")
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                
                # Download single safetensors file
                ckpt_path = hf_hub_download(repo_id="ByteDance/SDXL-Lightning", filename="sdxl_lightning_4step_unet.safetensors")
                
                # Load state dict
                state_dict = load_file(ckpt_path)
                
                # Apply to UNet
                m, u = self.unet.load_state_dict(state_dict, strict=False)
                print(f"[LaRE] Lightning weights loaded. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
            
            self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        except Exception as e:
            print(f"[LaRE] Failed to load from HuggingFace, trying local cache: {e}")
            raise e

        # Freeze components
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        self.vae.to(device)
        self.unet.to(device)
        
        # Pre-compute empty text embeddings
        print("[LaRE] Computing empty text embedding...")
        
        if model_type == "sdxl":
            # SDXL uses dual text encoders
            from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
            
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
            text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
            
            # Force dtype to ensure consistency
            text_encoder = text_encoder.to(dtype=dtype)
            text_encoder_2 = text_encoder_2.to(dtype=dtype)
            
            with torch.no_grad():
                # CLIP L
                text_input_1 = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                prompt_embeds_1 = text_encoder(text_input_1.input_ids.to(device))[0]
                
                # CLIP G (OpenCLIP)
                text_input_2 = tokenizer_2("", padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
                # output[0] is pooled_embeds, output[1] is last_hidden_state
                # We need last_hidden_state for cross-attention
                enc_2_out = text_encoder_2(text_input_2.input_ids.to(device))
                prompt_embeds_2 = enc_2_out[1] 
                self.pooled_prompt_embeds = enc_2_out[0]
                
                # Concatenate embeddings (SDXL expects [B, 77+77, 2048])
                self.empty_embeddings = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            
            del text_encoder, text_encoder_2, tokenizer, tokenizer_2
        else:
            # SD 1.5 / 2.1 use single text encoder
            from transformers import CLIPTextModel, CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
            
            with torch.no_grad():
                text_input = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                self.empty_embeddings = text_encoder(text_input.input_ids.to(device))[0]
                self.pooled_prompt_embeds = None
            
            del text_encoder, tokenizer
        
        torch.cuda.empty_cache()
        print(f"[LaRE] Ready. Embedding shape: {self.empty_embeddings.shape}")

        # Image preprocessing - adaptive to model resolution
        self.img_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _get_add_time_ids(self, bsz):
        # target_size = (self.resolution, self.resolution)
        # original_size = (self.resolution, self.resolution)
        # crops_coords_top_left = (0, 0)
        # add_time_ids = list(original_size + crops_coords_top_left + target_size)
        
        # Simplified for fixed resolution
        time_ids = torch.tensor([[self.resolution, self.resolution, 0, 0, self.resolution, self.resolution]], device=self.device, dtype=self.dtype)
        return time_ids.repeat(bsz, 1)

    def encode_latents(self, img_tensor):
        """Encode image to latent space"""
        # img_tensor: [1, 3, H, W] in range [-1, 1]
        # Force input to float32 for VAE
        latents = self.vae.encode(img_tensor.float()).latent_dist.sample()
        # Cast back to model dtype (e.g. BF16) for UNet
        latents = latents.to(dtype=self.dtype) * self.vae_scale
        return latents

    def get_epsilon(self, latents, t):
        # Predict noise
        # latents: [1, 4, h, w]
        # t: int timestep
        
        # Prepare inputs
        latent_model_input = latents
        t_tensor = torch.tensor([t], device=self.device, dtype=self.dtype) # .long() ?
        
        # SDXL handling
        kwargs = {}
        if self.model_type == "sdxl":
            bsz = latents.shape[0]
            kwargs["added_cond_kwargs"] = {
                "text_embeds": self.pooled_prompt_embeds.repeat(bsz, 1),
                "time_ids": self._get_add_time_ids(bsz)
            }
        
        # Predict
        noise_pred = self.unet(latent_model_input, t_tensor, encoder_hidden_states=self.empty_embeddings, **kwargs).sample
        return noise_pred

    def extract_single(self, pil_img, timestep=280, ensemble_size=1):
        """
        Extract LaRE features from a single image
        
        Args:
            pil_img: PIL Image
            timestep: Noise timestep (default 280, can tune for different models)
            ensemble_size: Number of forward passes to average (reduces noise)
        
        Returns:
            diff: [1, 4, H, W] error map (H=resolution/8, e.g., 128 for SDXL@1024)
        """
        
        # Preprocess
        img_tensor = self.img_transform(pil_img).unsqueeze(0).to(self.device).to(self.dtype)
        
        maps = []
        
        # Optimization: Encode once if ensemble_size > 1?
        # To strictly match "run script N times", we should re-encode.
        # But VAE is expensive. Let's try to encode once and reuse latent.
        # If results are bad, we can revert to full re-run.
        
        with torch.no_grad():
            # 1. Encode to latent (Sampled once)
            latents = self.encode_latents(img_tensor)
            
            for _ in range(ensemble_size):
                # 2. Add noise (Stochastic)
                noise = torch.randn_like(latents)
                timesteps = torch.tensor([timestep], device=self.device, dtype=torch.long)
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # 3. Predict noise using UNet
                kwargs = {}
                if self.model_type == "sdxl":
                    bsz = 1
                    kwargs["added_cond_kwargs"] = {
                        "text_embeds": self.pooled_prompt_embeds.repeat(bsz, 1),
                        "time_ids": self._get_add_time_ids(bsz)
                    }

                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=self.empty_embeddings, **kwargs).sample
                
                # 4. Compute reconstruction error (LaRE core)
                diff = (model_pred - noise).abs()  # L1 error
                maps.append(diff)
            
            # Average maps
            avg_diff = torch.stack(maps).mean(dim=0)
            
            # For SDXL: latents are [1, 4, 128, 128] at 1024x1024 input
            # Need to downsample to match classifier input [1, 4, 32, 32]
            if avg_diff.shape[-1] != 32:
                avg_diff = torch.nn.functional.interpolate(avg_diff, size=(32, 32), mode='bilinear', align_corners=False)
            
            return avg_diff  # [1, 4, 32, 32]

    def extract_batch(self, img_tensor, timestep=280):
        """
        Extract LaRE features from a batch of images
        
        Args:
            img_tensor: [B, 3, H, W] normalized tensor
            timestep: int
            
        Returns:
            diff: [B, 4, 32, 32]
        """
        # img_tensor is expected to be on device and correct dtype
        
        with torch.no_grad():
            # 1. Encode
            # Force input to float32 for VAE to prevent NaN/Overflow
            latents = self.vae.encode(img_tensor.float()).latent_dist.sample()
            # Cast back to model dtype (e.g. BF16)
            latents = latents.to(dtype=self.dtype) * self.vae_scale
            
            bsz = latents.shape[0]
            
            # 2. Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.full((bsz,), timestep, device=self.device, dtype=torch.long)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            # 3. Predict
            encoder_hidden_states = self.empty_embeddings.repeat(bsz, 1, 1)
            
            kwargs = {}
            if self.model_type == "sdxl":
                kwargs["added_cond_kwargs"] = {
                    "text_embeds": self.pooled_prompt_embeds.repeat(bsz, 1),
                    "time_ids": self._get_add_time_ids(bsz)
                }
            
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, **kwargs).sample
            
            # 4. Error
            diff = (model_pred - noise).abs()
            
            # 5. Resize
            if diff.shape[-1] != 32:
                diff = torch.nn.functional.interpolate(diff, size=(32, 32), mode='bilinear', align_corners=False)
                
            return diff