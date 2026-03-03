import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The PyTorch model instance.
            target_layer: The specific layer to compute gradients on (e.g., model.map_conv or model.attn_pool).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook registration
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, usually the first element is what we want
        self.gradients = grad_output[0]

    def __call__(self, clip_input, loss_map, class_idx=None):
        """
        Compute CAM for a specific class.
        
        Args:
            clip_input: Input image tensor for CLIP.
            loss_map: LaRE loss map tensor.
            class_idx: Target class index (0 for Real, 1 for AI). If None, uses max prediction.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(clip_input, loss_map)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0][class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling on gradients (weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU -> Normalize
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0], class_idx

def overlay_heatmap(img_pil, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image.
    
    Args:
        img_pil: Original PIL Image.
        cam_mask: 2D numpy array (0-1) from GradCAM.
        alpha: Transparency of heatmap.
        colormap: OpenCV colormap to use.
    """
    img_np = np.array(img_pil)
    
    # Resize mask to image size
    heatmap = cv2.resize(cam_mask, (img_np.shape[1], img_np.shape[0]))
    
    # Colorize
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Overlay
    # Convert heatmap to RGB (OpenCV uses BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed)

