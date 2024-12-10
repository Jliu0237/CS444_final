import os
import torch
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.utils.events import EventStorage, get_event_storage
from tqdm import tqdm

from config import get_cfg
from model import FashionMaskRCNN
from data_utils import register_fashion_datasets

class FashionTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_loss = float('inf')
        self.log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("fashion_trainer")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, "train.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Progress bar
        self.pbar = None
        
    def build_evaluator(self, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(self.cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder,
            tasks=("bbox", "segm"),
        )
        
    def run_step(self):
        assert self.model.training, "Model was changed to eval mode!"
        
        data = next(self._data_loader_iter)
        
        # Initialize progress bar if not exists
        if self.pbar is None:
            self.pbar = tqdm(total=self.max_iter, desc="Training")
        
        with EventStorage(self.iter) as storage:
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            storage.put_scalars(total_loss=losses, **loss_dict)
            
            # Log metrics
            if self.iter % 20 == 0:  # Log every 20 iterations
                metrics = {
                    "iter": self.iter,
                    "total_loss": losses.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                metrics.update({k: v.item() for k, v in loss_dict.items()})
                
                # Log to file
                self.logger.info(f"Iteration {self.iter}: " + 
                               " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                
                # Save best model
                if losses.item() < self.best_loss:
                    self.best_loss = losses.item()
                    self.save_model("model_best.pth")
            
            # Update progress bar
            self.pbar.update(1)
            self.pbar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
    def save_model(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.cfg.OUTPUT_DIR, filename)
        torch.save({
            'iteration': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
        }, path)
        self.logger.info(f"Saved model checkpoint to {path}")

def setup(args):
    cfg = get_cfg()
    
    # Update cfg with command line arguments
    cfg.merge_from_list(args.opts)
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.OUTPUT_DIR = os.path.join("output", timestamp)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Register datasets
    register_fashion_datasets(cfg.DATASETS.DATA_DIR)
    
    return cfg

def train(cfg, args):
    model = FashionMaskRCNN(cfg)
    
    logger = logging.getLogger("fashion_trainer")
    logger.setLevel(logging.INFO)
    
    # Load pre-trained weights if specified
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    trainer = FashionTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    logger.info("Starting training...")
    return trainer.train()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fashion Mask R-CNN Training")
    parser.add_argument("--resume", help="path to checkpoint to resume from")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg = setup(args)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "train.log")),
            logging.StreamHandler()
        ]
    )
    
    train(cfg, args)

if __name__ == "__main__":
    main() 