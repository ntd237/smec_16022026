import torch

class SelectiveCrossBatchMemory:
    """
    Selective Cross-Batch Memory (S-XBM).
    Maintains a queue of past embeddings (from frozen backbone) to provide hard negatives.
    """
    def __init__(self, memory_size: int, embedding_dim: int, device: torch.device):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize queue with zeros or random vectors
        self.memory_queue = torch.randn(memory_size, embedding_dim, device=device)
        self.ptr = 0
        self.full = False

    def enqueue(self, embeddings: torch.Tensor):
        """
        Add new embeddings to the queue.
        Args:
            embeddings: (Batch_Size, Embedding_Dim)
        """
        batch_size = embeddings.shape[0]
        
        if self.ptr + batch_size > self.memory_size:
            # Wrap around or just fill until end and reset?
            # Standard implementation: FIFO circular buffer
            remaining = self.memory_size - self.ptr
            self.memory_queue[self.ptr:] = embeddings[:remaining]
            self.memory_queue[:batch_size - remaining] = embeddings[remaining:]
            self.ptr = batch_size - remaining
            self.full = True
        else:
            self.memory_queue[self.ptr : self.ptr + batch_size] = embeddings
            self.ptr += batch_size
            
            if self.ptr == self.memory_size:
                self.ptr = 0
                self.full = True

    def retrieve_hard_negatives(self, query: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Retrieve top-k most similar embeddings from memory (Hard Negatives).
        Args:
            query: (Batch_Size, Embedding_Dim)
            k: Number of negatives to retrieve per query
        Returns:
            negatives: (Batch_Size, k, Embedding_Dim)
        """
        # Calculate similarity (Dot product)
        # Query shape: (B, D), Memory shape: (M, D)
        # Scores: (B, M)
        sim_scores = torch.matmul(query, self.memory_queue.T)
        
        # Get top-k indices
        topk_scores, topk_indices = torch.topk(sim_scores, k, dim=1)
        
        # Select embeddings
        # memory_queue[indices] -> (B, k, D)
        negatives = self.memory_queue[topk_indices]
        
        return negatives
