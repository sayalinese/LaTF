"""
Cascade Inference Strategy for V9-Lite
Implements two-stage inference: Global Scan -> Local Zoom

Optimized for detecting local inpainting and partial edits.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict, Optional
import cv2


class CascadeInference:
    """
    Two-stage cascade inference for improved local anomaly detection.
    
    Stage 1: Global Scan - Resize entire image to model input size and predict
    Stage 2: Local Zoom - If suspicious, crop high-activation region and re-predict
    """
    
    def __init__(
        self,
        model,
        lare_extractor,
        trufor_extractor=None,
        device: str = "cuda",
        input_size: int = 448,
        threshold: float = 0.5,
        zoom_padding: float = 0.2,
        min_crop_ratio: float = 0.25,
        max_crop_ratio: float = 0.75,
    ):
        """
        Args:
            model: V9-Lite model instance
            lare_extractor: LaRE feature extractor instance
            trufor_extractor: TruFor feature extractor instance (Optional)
            device: Device to run inference on
            input_size: Model input size (448 for CLIP RN50x64)
            threshold: Probability threshold to trigger local zoom
            zoom_padding: Padding ratio around detected region (0.2 = 20% each side)
            min_crop_ratio: Minimum crop size as ratio of original image
            max_crop_ratio: Maximum crop size as ratio of original image
        """
        self.model = model
        self.lare_extractor = lare_extractor
        self.trufor_extractor = trufor_extractor
        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        self.zoom_padding = zoom_padding
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        
        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        # [V11] HighRes Transform (Texture Branch)
        self.highres_transform = transforms.Compose([
            transforms.Resize((512, 512)), # Use 512 for texture branch
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # [V13] TruFor Preprocessing (Match training gen script)
        self.trufor_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
    
    def _get_bbox_from_locmap(
        self, 
        loc_map: torch.Tensor, 
        original_size: Tuple[int, int],
        activation_threshold: float = 0.5
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding box of high-activation region from localization map.
        
        Args:
            loc_map: [1, 1, H, W] localization map
            original_size: (width, height) of original image
            activation_threshold: Threshold for considering a pixel as "activated"
        
        Returns:
            (x1, y1, x2, y2) bounding box in original image coordinates, or None
        """
        loc_np = loc_map.squeeze().cpu().numpy()  # [H, W]
        
        # Suppress boundary artifacts before thresholding
        h, w = loc_np.shape
        if h > 4 and w > 4:
            # Mask out 2 pixels from the border
            mask = np.ones_like(loc_np)
            mask[:2, :] = 0
            mask[-2:, :] = 0
            mask[:, :2] = 0
            mask[:, -2:] = 0
            loc_np = loc_np * mask
        
        # Threshold to binary mask
        binary_mask = (loc_np > activation_threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Scale to original image size
        scale_x = original_size[0] / loc_np.shape[1]
        scale_y = original_size[1] / loc_np.shape[0]
        
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        
        return (x1, y1, x2, y2)
    
    def _add_padding(
        self, 
        bbox: Tuple[int, int, int, int], 
        img_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Add padding around bounding box while keeping it within image bounds.
        """
        x1, y1, x2, y2 = bbox
        w, h = img_size
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        pad_x = int(box_w * self.zoom_padding)
        pad_y = int(box_h * self.zoom_padding)
        
        # Add padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Ensure minimum and maximum crop size
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        min_size = int(min(w, h) * self.min_crop_ratio)
        max_size = int(min(w, h) * self.max_crop_ratio)
        
        # Expand if too small
        if crop_w < min_size or crop_h < min_size:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_size = min_size // 2
            
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)
        
        # Shrink if too large
        if crop_w > max_size or crop_h > max_size:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_size = max_size // 2
            
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)
        
        return (x1, y1, x2, y2)
    
    def _predict_single(
        self, 
        img_tensor: torch.Tensor, 
        loss_map: torch.Tensor,
        return_loc: bool = False,
        highres_tensor: Optional[torch.Tensor] = None,
        trufor_map: Optional[torch.Tensor] = None
    ) -> Tuple[float, int, Optional[torch.Tensor]]:
        """
        Run single prediction.
        
        Returns:
            prob: Probability of being AI-generated (class 1)
            pred: Predicted class (0=Real, 1=AI)
            loc_map: Localization map if return_loc=True
        """
        self.model.eval()
        
        # Determine mixed precision dtype
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            from torch.amp import autocast
        except ImportError:
            from torch.cuda.amp import autocast

        with torch.no_grad():
            with autocast('cuda', dtype=amp_dtype):
                if highres_tensor is not None:
                    # [V11] Inference with Texture Branch
                    if return_loc:
                        # V11 returns (logits, seg_logits)
                        # Check if model supports trufor_map
                        try:
                            outputs = self.model(img_tensor, highres_tensor, loss_map, trufor_map=trufor_map, return_seg=True)
                        except TypeError:
                             # Fallback for models without trufor_map
                            outputs = self.model(img_tensor, highres_tensor, loss_map, return_seg=True)
                            
                        logits, seg_logits = outputs
                        loc_map = torch.sigmoid(seg_logits) # Convert logits to 0-1 prob
                    else:
                        try:
                            logits = self.model(img_tensor, highres_tensor, loss_map, trufor_map=trufor_map)
                        except TypeError:
                            logits = self.model(img_tensor, highres_tensor, loss_map)
                        loc_map = None
                elif return_loc:
                    outputs = self.model(img_tensor, loss_map, return_loc=True)
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        logits, loc_map = outputs[0], outputs[1]
                    else:
                        logits = outputs
                        loc_map = None
                else:
                    outputs = self.model(img_tensor, loss_map)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loc_map = None
            
            # Ensure outputs are float32 for post-processing
            logits = logits.float()
            if loc_map is not None:
                loc_map = loc_map.float()
            
            probs = F.softmax(logits, dim=1)
            prob_ai = probs[0, 1].item()
            pred = torch.argmax(probs, dim=1).item()
        
        return prob_ai, pred, loc_map
    
    def inference(
        self, 
        image: Image.Image,
        return_details: bool = False
    ) -> Dict:
        """
        Run cascade inference on an image.
        
        Args:
            image: PIL Image (any size)
            return_details: Whether to return detailed intermediate results
        
        Returns:
            dict with keys:
                - 'prob': Final AI probability
                - 'pred': Final prediction (0=Real, 1=AI)
                - 'class_name': 'Real' or 'AI Generated'
                - 'global_prob': Stage 1 probability
                - 'local_prob': Stage 2 probability (if triggered)
                - 'loc_map': Localization heatmap (numpy array, 0-1)
                - 'crop_bbox': Crop region used in stage 2 (if triggered)
        """
        original_size = image.size  # (width, height)
        
        # === Stage 1: Global Scan ===
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # [V11] Prepare HighRes Tensor
        highres_tensor = None
        if hasattr(self, 'highres_transform'):
             highres_tensor = self.highres_transform(image).unsqueeze(0).to(self.device)
        
        # [V13] Extract TruFor Map (if available)
        trufor_map = None
        if self.trufor_extractor is not None:
             # TruFor expects [0,1] tensor [B, 3, H, W]
             # Use 1024x1024 resize to match training data generation and ensure consistent receptive field
             tf_input = self.trufor_transform(image).unsqueeze(0).to(self.device)
             
             # Returns numpy [B, H, W] (1024x1024)
             tf_np = self.trufor_extractor.extract_batch(tf_input) 
             
             if tf_np is not None:
                 # Convert to Tensor [1, 1, H, W]
                 tf_tensor = torch.from_numpy(tf_np).unsqueeze(1).float()
                 
                 # [V13 Fix] Resize to 448x448 to match dataset.py input size (data_size=448)
                 # Although model downsamples to 32x32, aligning the intermediate size is safer
                 tf_tensor = F.interpolate(tf_tensor, size=(448, 448), mode='bilinear', align_corners=False)
                 
                 trufor_map = tf_tensor.to(self.device)
                 # print(f"[DEBUG] TruFor Map Extracted: {trufor_map.shape}")

        # Extract LaRE features
        loss_map = self.lare_extractor.extract_single(image)  # [1, 4, 32, 32]
        
        # Debug Print
        lm_mean = loss_map.float().mean().item()
        lm_max = loss_map.float().max().item()
        print(f"[DEBUG] Cascade Global Loss Map - Mean: {lm_mean:.4f}, Max: {lm_max:.4f}")
        
        # Generate raw LaRE heatmap for valid visualization (Reconstruction Error)
        # Average across 4 channels -> [32, 32]
        raw_lare_map = loss_map.mean(dim=1).squeeze().cpu().numpy()

        loss_map = loss_map.to(self.device)
        
        global_prob, global_pred, loc_map = self._predict_single(
            img_tensor, loss_map, return_loc=True, highres_tensor=highres_tensor, trufor_map=trufor_map
        )
        
        result = {
            'global_prob': global_prob,
            'local_prob': None,
            'crop_bbox': None,
            'loc_map': loc_map.squeeze().cpu().numpy() if loc_map is not None else None,
            'raw_lare_map': raw_lare_map, # Add raw LaRE map for visualization
        }
        
        # === Stage 2: Local Zoom (if suspicious) ===
        if global_prob >= self.threshold and loc_map is not None:
            # Get high-activation region
            bbox = self._get_bbox_from_locmap(loc_map, original_size, activation_threshold=0.5)
            
            if bbox is not None:
                # Add padding
                bbox = self._add_padding(bbox, original_size)
                result['crop_bbox'] = bbox
                
                # Crop from ORIGINAL high-resolution image
                x1, y1, x2, y2 = bbox
                cropped_img = image.crop((x1, y1, x2, y2))
                
                # Preprocess cropped region
                crop_tensor = self.preprocess(cropped_img).unsqueeze(0).to(self.device)
                
                # [V11] Prepare HighRes Tensor for crop
                crop_highres = None
                if hasattr(self, 'highres_transform'):
                    crop_highres = self.highres_transform(cropped_img).unsqueeze(0).to(self.device)

                # [V13] Extract TruFor for crop
                crop_trufor = None
                if self.trufor_extractor is not None:
                     # [V13 Fix] Resize crop to 1024x1024 for TruFor to ensure consistent feature extraction
                     # Previous: tf_input_crop = transforms.ToTensor()(cropped_img).unsqueeze(0).to(self.device)
                     tf_input_crop = self.trufor_transform(cropped_img).unsqueeze(0).to(self.device)
                     
                     tf_np_crop = self.trufor_extractor.extract_batch(tf_input_crop)
                     if tf_np_crop is not None:
                         tf_tensor_crop = torch.from_numpy(tf_np_crop).unsqueeze(1).float()
                         
                        # [V13 Fix] Resize to 448x448 to match model input expectation
                         tf_tensor_crop = F.interpolate(tf_tensor_crop, size=(448, 448), mode='bilinear', align_corners=False)
                         
                         crop_trufor = tf_tensor_crop.to(self.device)

                # Extract LaRE for cropped region
                crop_loss_map = self.lare_extractor.extract_single(cropped_img)
                crop_loss_map = crop_loss_map.to(self.device)
                
                # Predict on cropped region
                local_prob, local_pred, _ = self._predict_single(
                    crop_tensor, crop_loss_map, return_loc=False, highres_tensor=crop_highres, trufor_map=crop_trufor
                )
                
                result['local_prob'] = local_prob
        
        # === Final Decision ===
        # Take max of global and local probabilities
        if result['local_prob'] is not None:
            final_prob = max(global_prob, result['local_prob'])
        else:
            final_prob = global_prob
        
        final_pred = 1 if final_prob >= 0.5 else 0
        
        result['prob'] = final_prob
        result['pred'] = final_pred
        result['class_name'] = 'AI Generated' if final_pred == 1 else 'Real Photo'
        
        return result
    
    def inference_with_heatmap(
        self, 
        image: Image.Image
    ) -> Tuple[Dict, np.ndarray]:
        """
        Run inference and return result with visualization-ready heatmap.
        
        Returns:
            result: Same as inference()
            heatmap_overlay: RGB numpy array with heatmap overlaid on image
        """
        result = self.inference(image, return_details=True)
        
        # Create heatmap overlay
        img_np = np.array(image.resize((self.input_size, self.input_size)))
        
        if result['loc_map'] is not None:
            # Resize localization map to image size
            loc_map_resized = cv2.resize(
                result['loc_map'], 
                (self.input_size, self.input_size)
            )
            
            # Colorize heatmap
            heatmap = np.uint8(255 * loc_map_resized)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay
            alpha = 0.5
            heatmap_overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
        else:
            heatmap_overlay = img_np
        
        return result, heatmap_overlay


def inference_cascade(
    model, 
    lare_extractor,
    image_path: str,
    trufor_extractor=None,
    device: str = "cuda",
    threshold: float = 0.5
) -> Dict:
    """
    Convenience function for cascade inference on a single image file.
    
    Args:
        model: V9-Lite model
        lare_extractor: LaRE extractor
        image_path: Path to image file
        trufor_extractor: Optional TruFor extractor
        device: Device to use
        threshold: Threshold for local zoom
    
    Returns:
        Inference result dict
    """
    image = Image.open(image_path).convert('RGB')
    
    cascade = CascadeInference(
        model=model,
        lare_extractor=lare_extractor,
        trufor_extractor=trufor_extractor,
        device=device,
        threshold=threshold
    )
    
    return cascade.inference(image)

