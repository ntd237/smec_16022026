import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .models.smec_wrapper import SMECModel
from .models.memory import SelectiveCrossBatchMemory
from .loss import SMECContrastiveLoss
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class SMECTrainer:
    """
    Trainer for Sequential Matryoshka Embedding Compression.
    """
    def __init__(
        self,
        model: SMECModel,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 2e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./checkpoints",
        max_length: int = 128
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.max_length = max_length
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = SMECContrastiveLoss()
        
        # Mixed Precision support
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        
        # Memory Queue (S-XBM)
        # Initialize with max size, e.g., 65536 or related to batch size
        self.memory = SelectiveCrossBatchMemory(
            memory_size=1024, # Small for demo/testing
            embedding_dim=model.embedding_dim, # Stores FULL dimension or formatted?
            # Actually S-XBM likely stores the dimension currently being trained?
            # Or the frozen backbone output?
            # Paper: "S-XBM stores embeddings from the frozen backbone."
            # So embedding_dim should correspond to backbone output.
            device=device
        )
        
        os.makedirs(output_dir, exist_ok=True)

    def train_one_epoch(self, epoch_idx: int, target_dim: int):
        self.model.train()
        total_loss = 0
        
        # Determine if we use memory
        # S-XBM usually used when reducing dimension to maintain contrast with high-dim features?
        # Or just standard hard negatives.
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Dim {target_dim}]")
        for batch in pbar:
            # Batch: {'queries': [...], 'positives': [...], 'negatives': [...]}
            # Tokenize
            queries = self._tokenize(batch['queries'])
            positives = self._tokenize(batch['positives'])
            
            # Forward with Mixed Precision
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                # Forward
                q_emb = self.model(queries['input_ids'].to(self.device), queries['attention_mask'].to(self.device))
                p_emb = self.model(positives['input_ids'].to(self.device), positives['attention_mask'].to(self.device))
                
                # Loss
                loss = self.criterion(q_emb, p_emb, memory_queue=self.memory)
            
            # Backward with Scaler
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"End Epoch {epoch_idx} [Dim {target_dim}] - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train_sequential(self, dimensions: list[int], epochs_per_dim: int = 3):
        """
        Main Sequential Loop (SMRL).
        Args:
            dimensions: List of target dimensions descending, e.g. [768, 384, 192]
        """
        for i, dim in enumerate(dimensions):
            logger.info(f"=== Starting Training for Dimension: {dim} ===")
            
            # 1. Set model target dimension (activates ADS for this dim)
            self.model.set_ads_target_dim(dim)
            
            # 2. Freeze/Unfreeze Logic
            # SMRL: "Parameters optimized in previous steps are frozen."
            # If i > 0: freeze previous ADS layers?
            # Our simpler ADS implementation might just be one layer.
            # If we learn a mask, maybe we freeze mask values for kept dimensions?
            # For simplicity in this implementation plan:
            # We assume we retrain/finetune or keep training.
            # Strict SMEC: Freeze backbone always? Or just after first step?
            # "Backbone frozen during ADS training." -> Implies backbone always frozen if just training ADS?
            # Or Sequence: Train Backbone (Full) -> Freeze Backbone -> Train ADS (D/2) -> Freeze ADS(D/2) -> ...
            
            if i == 0:
                 # First step: Train backbone? Or assumes backbone is pretrained?
                 # Usually MRL starts with finetuning backbone.
                 self.model.unfreeze_backbone()
            else:
                 # Subsequent steps: Freeze backbone, train ADS
                 self.model.freeze_backbone()
            
            # 3. Train loop
            for epoch in range(epochs_per_dim):
                self.train_one_epoch(epoch, dim)
                
            # 4. Save checkpoint
            self.save_checkpoint(f"checkpoint_dim_{dim}")

    def save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")

    def _tokenize(self, texts):
        # Helper to tokenize batch
        # Assuming tokenizer is accessible or passed in logic
        # For this skeleton, we'll need the tokenizer attached to model or passed in
        # We can get it from model.model_name
        from transformers import AutoTokenizer
        if not hasattr(self, 'tokenizer'):
             self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)
             
        return self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
