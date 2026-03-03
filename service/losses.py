import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (adapted for image-level labels)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] normalized embeddings
            labels: [B] int labels
        Returns:
            Scalar contrastive loss
        """
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positive pairs
        pos_mask_sum = mask.sum(1)
        pos_mask_sum = torch.clamp(pos_mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_mask_sum

        loss = -mean_log_prob_pos
        return loss.mean()
