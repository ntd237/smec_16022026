import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDimensionSelection(nn.Module):
    """
    Adaptive Dimension Selection (ADS) Layer.
    Uses Gumbel-Softmax to learn a mask for selecting important dimensions.
    """
    def __init__(self, input_dim: int, output_dim: int, temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        self.gate_logits = nn.Parameter(torch.randn(output_dim, input_dim))

    def sample_gumbel(self, shape, eps=1e-20):
        """
        Sample Gumbel(0,1) noise:
           g = -log(-log(u))
        where u ~ Uniform(0,1)
        """
        U = torch.rand(shape).to(self.gate_logits.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (Batch, Input_Dim)
            hard: Whether to return hard one-hot vectors (discrete selection)
        Returns:
            Compressed embeddings (Batch, Output_Dim) 
        """
        device = x.device
        if self.gate_logits.device != device:
             self.gate_logits.data = self.gate_logits.data.to(device)
        # Sample Gumbel noise
        gumbel_noise = self.sample_gumbel(self.gate_logits.shape)
        # Add noise and scale by temperature
        logits_with_noise = (self.gate_logits + gumbel_noise) / self.temperature
        selection_matrix = F.softmax(logits_with_noise, dim=-1)  # (Output_Dim, Input_Dim)
        
        return x @ selection_matrix.T  # (Batch, Output_Dim)

class TopKSelector(nn.Module):
    """
    Selects top-k dimensions based on learned importance.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, importance_scores: torch.Tensor):
        # Implementation of taking top-k features
        topk_indices = torch.topk(importance_scores, self.k).indices
        return x[:, topk_indices]
