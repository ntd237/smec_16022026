import argparse
import logging
import os
import torch
from src.models.smec_wrapper import SMECModel
from src.trainer import SMECTrainer
from src.data.loader import get_dataloader
from src.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SMEC: Sequential Matryoshka Embedding Compression")
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode: train or eval")
    
    # Model
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Backbone model name")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    
    # Data
    parser.add_argument("--dataset_name", type=str, default="quora", help="Dataset name/path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per dimension")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    
    args = parser.parse_args()
    
    # Initialize Model
    logger.info(f"Initializing model: {args.model_name}")
    model = SMECModel(args.model_name)
    
    if args.mode == "train":
        logger.info("Starting Training Mode")
        
        # DataLoader
        train_loader = get_dataloader(
            args.dataset_name, 
            batch_size=args.batch_size, 
            split="train",
            max_length=args.max_length
        )
        
        # Trainer
        trainer = SMECTrainer(
            model=model,
            train_loader=train_loader,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            max_length=args.max_length
        )
        
        # Sequential Training
        # Example dimensions: [768, 384, 192, 96]
        # Depending on backbone, e.g. BERT=768
        full_dim = model.embedding_dim
        dimensions = [full_dim, full_dim // 2, full_dim // 4]
        
        trainer.train_sequential(dimensions, epochs_per_dim=args.epochs)
        
    elif args.mode == "eval":
        logger.info("Starting Evaluation Mode")
        
        checkpoints_to_run = []
        
        if args.checkpoint:
            checkpoints_to_run.append(args.checkpoint)
        else:
            # Look for all checkpoints in output_dir
            if os.path.exists(args.output_dir):
                all_files = os.listdir(args.output_dir)
                dim_checkpoints = [os.path.join(args.output_dir, f) for f in all_files if "checkpoint_dim_" in f]
                # Sort them descending (768, 384, 192)
                dim_checkpoints.sort(key=lambda x: int(x.split("checkpoint_dim_")[-1]) if x.split("checkpoint_dim_")[-1].isdigit() else 0, reverse=True)
                checkpoints_to_run.extend(dim_checkpoints)
        
        if not checkpoints_to_run:
            logger.warning("No checkpoints found. Evaluating base model only.")
            run_evaluation(model, tasks=["STSBenchmark"], output_folder=os.path.join("results", "results_base"))
        else:
            for cp_path in checkpoints_to_run:
                logger.info(f"\n{'='*20}\nEvaluating Checkpoint: {cp_path}\n{'='*20}")
                
                # Reset/Set target dimension
                if "checkpoint_dim_" in cp_path:
                    try:
                        dim_str = cp_path.split("checkpoint_dim_")[-1]
                        target_dim = int(dim_str)
                        logger.info(f"Setting target dimension: {target_dim}")
                        model.set_ads_target_dim(target_dim)
                    except ValueError:
                        pass
                
                # Load weights
                try:
                    model.load_state_dict(torch.load(cp_path, map_location="cpu"))
                    # Run Evaluation
                    eval_subfolder = f"results_{os.path.basename(cp_path)}"
                    eval_path = os.path.join("results", eval_subfolder)
                    run_evaluation(model, tasks=["STSBenchmark"], output_folder=eval_path)
                except Exception as e:
                    logger.error(f"Failed to evaluate {cp_path}: {e}")

if __name__ == "__main__":
    main()
