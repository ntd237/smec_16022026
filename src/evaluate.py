import logging
import torch
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMECModelWrapper:
    """
    Wrapper for MTEB compatibility.
    MTEB expects an encode() method.
    """
    def __init__(self, smec_model, batch_size=32, device='cuda'):
        self.model = smec_model
        self.batch_size = batch_size
        self.device = device
        self.model.to(device)
        self.model.eval()

    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs):
        """
        Encodes a list of sentences to embeddings.
        """
        # We need a tokenize function here or inside the model
        # Re-using the logic from Trainer (simplified)
        from transformers import AutoTokenizer
        if not hasattr(self, 'tokenizer'):
             self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)
        
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(inputs['input_ids'], inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0).numpy()

def run_evaluation(model, tasks=['QuoraRetrieval'], output_folder="results"):
    """
    Run MTEB evaluation.
    Args:
        model: SMECModel instance
        tasks: List of MTEB tasks to run
    """
    wrapper = SMECModelWrapper(model)
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(wrapper, output_folder=output_folder)
    
    logger.info(f"Evaluation Results: {results}")
    return results
