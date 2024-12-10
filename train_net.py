#!/usr/bin/env python

import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from data_utils import register_dataset
import argparse
from detectron2.data import MetadataCatalog

class FashionTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Fashion Dataset')
    parser.add_argument('--data-dir', default='./data', help='path to dataset')
    parser.add_argument('--output-dir', default='./output', help='path to output directory')
    parser.add_argument('--num-gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--max-iterations', type=int, default=10000, help='maximum number of iterations')
    parser.add_argument('--max-images', type=int, default=0, help='maximum number of images to use (0 for all)')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--eval-only', action='store_true', help='perform evaluation only')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                      help='modify config options using the command-line')
    args = parser.parse_args()
    
    print("Command line arguments:", args)

    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset config
    if not hasattr(cfg, 'DATASETS'):
        cfg.DATASETS = type('', (), {})()
    if args.max_images > 0:
        cfg.DATASETS.MAX_IMAGES = args.max_images
    
    # Register datasets
    train_dataset = register_dataset(args.data_dir, "train", cfg)
    test_dataset = register_dataset(args.data_dir, "test", cfg)
    
    # Get number of categories from metadata
    num_categories = len(MetadataCatalog.get("fashion_train").thing_classes)
    print(f"\nNumber of categories: {num_categories}")
    
    cfg.DATASETS.TRAIN = ("fashion_train",)
    cfg.DATASETS.TEST = ("fashion_test",)
    
    # Input config
    cfg.INPUT.MASK_FORMAT = "bitmask"  # Use bitmask format for RLE masks
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Model config
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_categories
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.MASK_ON = True
    
    # Solver config
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iterations
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    
    # Output config
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Training config
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size * args.num_gpus
    
    if args.eval_only:
        print("Evaluation mode")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        trainer = FashionTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.test(cfg, trainer.model)
    else:
        print(f"Starting training with {args.max_images} images for {args.max_iterations} iterations...")
        trainer = FashionTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()

if __name__ == "__main__":
    main() 