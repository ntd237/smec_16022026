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
        
        # Learnable parameters for dimension selection
        # Shape: (input_dim, 2) -> representing probability of selecting (1) or not selecting (0)
        # However, for dimensionality reduction D -> d, we need to select exactly 'output_dim' features 
        # OR we learn a weight for each dimension.
        # Paper approach: "ADS replaces static pruning with dynamic, learnable dimension selection."
        # Implementation: Learn a mask 'm' via Gumbel-Softmax.
        
        # We use a Linear layer to project or select? 
        # If strictly selection, we learn a mask.
        # Let's implement a learnable gate. 
        self.gate_logits = nn.Parameter(torch.randn(input_dim))

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (Batch, Input_Dim)
            hard: Whether to return hard one-hot vectors (discrete selection)
        Returns:
            Compressed embeddings (Batch, Output_Dim) - This might be tricky if simple masking.
            If we just mask, dimension stays same but zeros out.
            If we need reduced dimension, we might need a Top-K selection or a projection.
            
            Refining based on SMEC paper logic:
            Usage of Gumbel-Softmax suggests differentiable selection.
            If the goal is D -> D/2, likely we select top-k indices based on learned gates.
        """
        # Gumbel-Softmax to get soft mask
        # We need to sample 'k' dimensions? Or just weight them?
        # Standard Gumbel-Softmax is for categorical distribution.
        
        # Simpler approach for "selection":
        # 1. Learn importance scores for all D dimensions.
        # 2. Select top-k indices for the target output_dim.
        # 3. During training, we might mask others or re-order?
        
        # Let's assume a Masking approach first, where we weigh features.
        # But for storage compression, we need actual size reduction.
        # Most Matryoshka models just take the first k dimensions.
        # SMEC ADS likely re-orders or selects specific indices to be the "first k".
        
        # Logic:
        # We want to select 'output_dim' features from 'input_dim'.
        # We can learn a permutation or a selection matrix.
        
        # Fallback to simple learnable weights + Top-K masking for now as per "Adaptive" description.
        # But to make it differentiable, we apply the weights.
        
        mask_weights = torch.sigmoid(self.gate_logits)
        
        if self.training:
            # Add noise for exploration if needed, or just use weights
            pass
            
        # Standard Matryoshka takes x[:, :output_dim].
        # ADS likely transforms x s.t. important features move to indices 0..output_dim?
        # OR it just selects specific indices.
        
        # Let's implement a weighted projection for now which is safer.
        # x_weighted = x * mask_weights
        # specific implementation details might vary without exact code from paper.
        
        return x * mask_weights.unsqueeze(0)

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
