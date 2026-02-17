import logging
import torch
import mteb
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMECModelWrapper(mteb.EncoderProtocol):
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

    @property
    def mteb_model_meta(self):
        from mteb.models.model_meta import ModelMeta
        return ModelMeta.model_construct(
            name="SMEC-Model",
            revision="1.0.0",
            release_date="2026-02-17",
            languages=["eng-Latn"],
            loader=None,
            max_tokens=512,
            embed_dim=self.model.current_dim,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=set(),
            adapted_from=None,
            superseded_by=None,
            modalities=["text"],
            model_type=["dense"],
            citation=None,
            contacts=None,
            reference=None,
        )

    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs):
        """
        Encodes a list of sentences to embeddings.
        """
        # MTEB might pass a DataLoader or a list of batches
        if not isinstance(sentences, list):
            try:
                sentences = list(sentences)
            except Exception:
                pass
        
        # Handle list of batches (often dicts or lists)
        if len(sentences) > 0 and not isinstance(sentences[0], str):
            all_texts = []
            for item in sentences:
                if isinstance(item, dict) and 'text' in item:
                    # If it's a batch dict from MTEB/HF
                    batch_texts = item['text']
                    if isinstance(batch_texts, list):
                        all_texts.extend(batch_texts)
                    else:
                        all_texts.append(batch_texts)
                elif isinstance(item, (list, tuple)):
                    all_texts.extend(item)
                elif isinstance(item, str):
                    all_texts.append(item)
            sentences = all_texts

        logger.info(f"Encoding {len(sentences)} sentences. First samples: {sentences[:2]}")
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
                max_length=self.model.current_dim # Or fixed 512? Let's use 512 for safety in eval
            ).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(inputs['input_ids'], inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0).numpy()

    def similarity(self, embeddings1, embeddings2):
        import numpy as np
        return np.dot(embeddings1, embeddings2.T)

    def similarity_pairwise(self, embeddings1, embeddings2):
        import numpy as np
        return np.multiply(embeddings1, embeddings2).sum(axis=1)

def run_evaluation(model, tasks=['QuoraRetrieval'], output_folder="results"):
    """
    Run MTEB evaluation.
    Args:
        model: SMECModel instance
        tasks: List of MTEB tasks to run
    """
    wrapper = SMECModelWrapper(model)
    
    # Get actual task objects to avoid AttributeError in newer MTEB versions
    task_objs = mteb.get_tasks(tasks=tasks)
    evaluation = MTEB(tasks=task_objs)
    results = evaluation.run(wrapper, output_folder=output_folder)
    
    logger.info(f"Evaluation Results: {results}")
    return results
