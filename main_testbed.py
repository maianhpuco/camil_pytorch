import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
import openslide
from data.merge_dataset import SuperpixelDataset, PatchDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from patch_merging import tome 
from utils import utils  

PROJECT_DIR = os.environ.get('PROJECT_DIR')
# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
# example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
example_list =['normal_048', 'tumor_026', 'tumor_031'] 

SLIDE_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/images' #replace you path 
JSON_PATH = '/project/hnguyen2/mvu9/camelyon16/json_files' # replace your path 

 
sys.path.append(os.path.join(PROJECT_DIR))

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  

def main(): 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the patch to 256x256
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
    ])

    wsi_paths = glob.glob(os.path.join(SLIDE_PATH, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    json_folder = JSON_PATH  
    
    for wsi_path in wsi_paths: 
        print(wsi_path) 
        
        dataset = SuperpixelDataset(
            wsi_path=wsi_path,
            json_folder=json_folder,
            )
        slide = openslide.open_slide(wsi_path)
        print("number of superpixel", len(dataset))   # list all the superpixel in the wsi image
        
        for foreground_idx, xywh_abs_bbox, superpixel_extrapolated in dataset:
            start = time.time()
            try:
                # Create region from slide based on the bounding box
                region = utils.get_region_original_size(slide, xywh_abs_bbox)
                region_np = np.array(region)
                print(f"Slicing time: {time.time() - start} seconds")

                print(f"Bounding Box (XYWH): {xywh_abs_bbox}")
                print(f"Shape of Superpixel: {region_np.shape}, Extrapolated Mask Shape: {superpixel_extrapolated.shape}")
                print(f"Superpixel {foreground_idx} foreground count: {np.sum(superpixel_extrapolated)}")
                
                patch_dataset = PatchDataset(
                    region_np,
                    superpixel_extrapolated, 
                    patch_size=(224, 224),
                    transform=transform,
                    coverage_threshold=0.5,
                    return_feature=True,  # Enable feature extraction
                    model=model
                )
                
                patch_dataloader = DataLoader(patch_dataset, batch_size=64, shuffle=False)
                
                _all_features_spixel = []
                _all_idxes_spixel = []
            
                for batch_features, batch_patches, batch_bboxes, batch_idxes in patch_dataloader:
                    print(f"Batch Features Shape: {batch_features.shape}")
                    _flatten_features = batch_features.view(-1, batch_features.shape[-1])
                    _all_features_spixel.append(_flatten_features)
                    _all_idxes_spixel.append(batch_idxes)
                
                spixel_features = torch.cat(_all_features_spixel)
                print(f"Final feature shape for superpixel {foreground_idx}: {spixel_features.shape}")
                
                spixel_foreground_idxes = torch.cat(_all_idxes_spixel, dim=0).detach().cpu().numpy().tolist()
                print(f"Foreground Indices Count: {len(spixel_foreground_idxes)}")
            
            except Exception as e:
                print(f"Error processing superpixel {foreground_idx} in WSI {wsi_path}: {e}")
                continue  # Skip this superpixel and move to the next one
            break 
        break
    
if __name__ == '__main__':
    main()
    # TODO
    # get argument: dry run only get 1 wsi, 1 superpixel, run thru from the start to the end 
    