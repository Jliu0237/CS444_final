from detectron2.config import get_cfg as get_detectron_cfg
from detectron2.config import CfgNode as CN

def get_cfg():
    # Start with Detectron2's default config
    cfg = get_detectron_cfg()
    
    # Add custom configs
    cfg.DATASETS.DATA_DIR = ""
    cfg.DATASETS.MAX_IMAGES = -1  # -1 means use all images
    
    # Model configs
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 92  # Updated to match actual number of categories
    
    # Backbone configs
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    
    # RPN configs
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    # ROI Head configs
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 92  # Updated to match actual number of categories
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # Box Head configs
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_BOX_HEAD.NORM = ""
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
    cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlign"
    
    # Mask Head configs
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.NORM = ""
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlign"
    
    # Initialize from COCO weights
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    
    # Solver configs
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 50000
    cfg.SOLVER.STEPS = (30000, 40000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    
    # Input configs
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Augmentation configs
    cfg.AUGMENTATION = CN()
    cfg.AUGMENTATION.FLIP_PROB = 0.5
    cfg.AUGMENTATION.BRIGHTNESS_RANGE = (0.8, 1.2)
    cfg.AUGMENTATION.CONTRAST_RANGE = (0.8, 1.2)
    cfg.AUGMENTATION.SATURATION_RANGE = (0.8, 1.2)
    cfg.AUGMENTATION.ROTATION_RANGE = (-15, 15)
    
    # Dataset configs
    cfg.DATASETS.TRAIN = "fashion_train"
    cfg.DATASETS.TEST = "fashion_test"
    cfg.DATASETS.NUM_ATTRIBUTES = 294  # Number of attributes in Fashionpedia
    
    return cfg 