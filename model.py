import torch
import torch.nn as nn
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler

@ROI_HEADS_REGISTRY.register()
class FashionROIHeads(StandardROIHeads):
    """
    Custom ROI heads for fashion attribute classification and instance segmentation.
    Extends Detectron2's StandardROIHeads with an additional attribute classification head.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        
        # Attribute classification head
        self.num_attributes = cfg.DATASETS.NUM_ATTRIBUTES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        
        self.attribute_head = nn.Sequential(
            nn.Linear(self.box_head.output_shape.channels * pooler_resolution ** 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_attributes)
        )
        
    def forward(self, images, features, proposals, targets=None):
        """
        Forward pass of the ROI heads.
        """
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            
        features_list = [features[f] for f in self.in_features]
        
        # Box features for both detection and attribute classification
        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        
        # Predictions for detection and segmentation
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        
        # Mask predictions
        if self.mask_on:
            mask_features = self.mask_pooler(features_list, [x.proposal_boxes for x in proposals])
            mask_features = self.mask_head(mask_features)
            mask_pred = self.mask_predictor(mask_features)
        else:
            mask_pred = None
            
        # Attribute predictions
        flattened_features = box_features.view(box_features.size(0), -1)
        attribute_pred = self.attribute_head(flattened_features)
        
        if self.training:
            losses = self.box_predictor.losses(pred_class_logits, pred_proposal_deltas, proposals)
            
            if self.mask_on and mask_pred is not None:
                losses.update(self.mask_head.losses(mask_pred, proposals))
                
            # Attribute classification loss
            attribute_targets = torch.cat([p.gt_attributes for p in proposals], dim=0)
            attribute_loss = nn.BCEWithLogitsLoss()(attribute_pred, attribute_targets)
            losses["loss_attributes"] = attribute_loss
            
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(pred_class_logits, pred_proposal_deltas, proposals)
            
            if self.mask_on:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
                
            # Add attribute predictions to instances
            pred_instances.pred_attributes = torch.sigmoid(attribute_pred)
            
            return pred_instances

class FashionMaskRCNN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(cfg)
        
    def train(self, do_train=True):
        self.model.train(do_train)
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.model.to(device)
        
    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        
    def forward(self, batched_inputs):
        return self.model(batched_inputs) 