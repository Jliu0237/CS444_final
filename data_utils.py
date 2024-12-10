import os
import json
import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split

def decode_rle(rle_str, height, width):
    """Decode RLE string to binary mask."""
    try:
        # Split the RLE string into counts
        counts = [int(x) for x in rle_str.split()]
        if len(counts) % 2 != 0:
            return np.zeros((height, width), dtype=np.uint8)
            
        # Initialize mask with writeable flag
        mask = np.zeros((height, width), dtype=np.uint8, order='C')
        mask.flags.writeable = True
        
        # Current position in the flattened image
        pos = 0
        val = 0  # Start with background (0)
        
        # Process pairs of counts
        for count in counts:
            mask.flat[pos:pos + count] = val
            pos += count
            val = 1 - val  # Toggle between 0 and 1
            
        return mask
        
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        return np.zeros((height, width), dtype=np.uint8)

def encode_binary_mask(mask):
    """Encode binary mask to COCO RLE format."""
    # Ensure mask is binary, uint8, and writable
    mask = np.asarray(mask, dtype=np.uint8, order='F').copy()
    mask.flags.writeable = True
    
    # Encode mask
    encoded = mask_utils.encode(mask)
    
    # Convert to format expected by detectron2
    encoded['counts'] = encoded['counts'].decode('ascii')
    
    return encoded

def get_fashion_dicts(data_dir, split="train"):
    """Get dataset dictionaries for fashion dataset."""
    # Load annotations
    train_csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Could not find {train_csv_path}")
        
    # Load the data
    df = pd.read_csv(train_csv_path)
    
    # Get unique image IDs and their first occurrence indices
    unique_images = df["image_id"].unique()
    
    # Split image IDs into train and test sets using sklearn
    train_ids, test_ids = train_test_split(
        unique_images,
        test_size=0.2,  # 20% for testing
        random_state=42,  # For reproducibility
        shuffle=True,  # Ensure data is shuffled
        stratify=None  # No stratification as we don't have class info at image level
    )
    
    # Convert to sets for faster lookup
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    
    # Select images based on split
    if split == "train":
        df = df[df["image_id"].isin(train_ids)]
        print(f"Training set: {len(train_ids)} images, {len(df)} annotations")
    else:  # test
        df = df[df["image_id"].isin(test_ids)]
        print(f"Test set: {len(test_ids)} images, {len(df)} annotations")
    
    # Group by image_id for faster processing
    grouped = df.groupby("image_id")
    
    dataset_dicts = []
    for image_id, group in grouped:
        record = {}
        
        # Get image info
        image_path = os.path.join(data_dir, "images", f"{image_id}.jpg")
        height, width = group.iloc[0][["height", "width"]].values
        
        record["file_name"] = image_path
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width
        
        objs = []
        for _, row in group.iterrows():
            # Get category info
            category_id = row["category_id"]
            
            # Decode RLE mask
            rle_str = row["segmentation"]
            binary_mask = decode_rle(rle_str, height, width)
            
            # Encode mask in COCO RLE format
            rle = encode_binary_mask(binary_mask)
            
            # Get bounding box from mask
            bbox = mask_utils.toBbox(rle).tolist()
            
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": rle,
                "category_id": category_id,
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_fashion_datasets(data_dir):
    """Register fashion datasets for both training and testing."""
    # Register training dataset (80% of data)
    DatasetCatalog.register(
        "fashion_train", 
        lambda: get_fashion_dicts(data_dir, "train")
    )
    MetadataCatalog.get("fashion_train").set(
        thing_classes=[f"category_{i}" for i in range(92)]
    )
    
    # Register test dataset (20% of data)
    DatasetCatalog.register(
        "fashion_test", 
        lambda: get_fashion_dicts(data_dir, "test")
    )
    MetadataCatalog.get("fashion_test").set(
        thing_classes=[f"category_{i}" for i in range(92)]
    ) 