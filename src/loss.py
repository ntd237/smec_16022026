import torch
import torch.nn as nn
import torch.nn.functional as F

class SMECContrastiveLoss(nn.Module):
    """
    Contrastive Loss with S-XBM support for SMEC.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query: torch.Tensor, positive: torch.Tensor, memory_queue = None):
        """
        Args:
            query: (Batch, Dim)
            positive: (Batch, Dim)
            memory_queue: Instance of SelectiveCrossBatchMemory (optional)
        """
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        
        # Calculate similarity with positives (Batch, 1)
        # We want logits: (Batch, 1 + Negatives)
        
        # In-batch negatives
        # Sim matrix: (Batch, Batch)
        logits = torch.matmul(query, positive.T) / self.temperature
        
        # Standard InfoNCE: Labels are diagonal (0, 1, 2...)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        if memory_queue is not None:
             # Retrieve hard negatives from memory
             # We want to add these as additional negative samples
             # Hard negatives for each query: (Batch, K, Dim)
             k_negatives = memory_queue.retrieve_hard_negatives(query, k=10) # Hardcoded k for now
             
             # Similarity with hard negatives: (Batch, K)
             # Reshape k_negatives: (Batch * K, Dim) ??? No.
             
             # We compute (Batch, Dim) dot (Batch, K, Dim) -> (Batch, K)
             hard_neg_logits = torch.bmm(k_negatives, query.unsqueeze(2)).squeeze(2)
             hard_neg_logits = hard_neg_logits / self.temperature
             
             # Concatenate with in-batch logits?
             # Standard InfoNCE usually just does Batch vs Batch.
             # If we add hard negatives, we extend the logits dim 1.
             
             # Logits: (Batch, Batch + K)
             # Labels still point to the positive index (diagonal).
             logits = torch.cat([logits, hard_neg_logits], dim=1)
             
        loss = self.cross_entropy(logits, labels)
        
        # Enqueue new embeddings to memory
        if memory_queue is not None:
            # We can enqueue positives or queries? Likely positives (passages).
            memory_queue.enqueue(positive.detach())
            
        return loss
