import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse
from data.merge_dataset import SuperpixelDataset, PatchDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import timm 


 # Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the patch to 256x256
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
])
 

PROJECT_DIR = os.environ.get('PROJECT_DIR')
# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']

SLIDE_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/images'
JSON_PATH = '/project/hnguyen2/mvu9/camelyon16/json_files'
 
 
sys.path.append(os.path.join(PROJECT_DIR))
# Load a pre-trained ViT model from timm

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  # Set the model to evaluation mode 

def main():
    wsi_paths = glob.glob(os.path.join(SLIDE_PATH, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    json_folder = JSON_PATH  
    
    dataset = SuperpixelDataset(
        slide_paths=wsi_paths,
        json_folder=json_folder,
        )
    
    import time 
    start = time.time()
    # llop though each WSI image 
    for wsi_data in dataset:
        #loop though each superpixel
        for foreground_idx, region_np, superpixel_extrapolated in wsi_data:
            print(foreground_idx)
            print(region_np.shape)
            print(superpixel_extrapolated.shape) 
            patch_dataset = PatchDataset(
                region_np,
                superpixel_extrapolated,
                patch_size = (224, 224),
                transform = transform, 
                return_feature=True,  # Enable feature extraction
                model=model 
            )
            patch_dataloader = DataLoader(patch_dataset, batch_size=4, shuffle=True)
            all_features_of_superpixel = [] 
            for patches, bboxes in patch_dataloader: 
                print(f"Batch of patches shape: {patches.shape}")
                print(f"Batch of bounding boxes: {bboxes}")
            
                # Stack all features
                all_features_of_superpixel.append(patches)
            ts_all_features_of_superpixel = torch.cat(all_features, dim=0) 
            print(ts_all_features_of_superpixelts.shape)              

        break
    print("Time to finish", time.time() - start, "second")
     
    
if __name__ == '__main__':
    main()
#checkpoint_path camelyon16_20241118-1932_completed.pth
# /project/hnguyen2/mvu9/camelyon16_features_data

# cp -r /home/mvu9/digital_pathology/camil_pytorch/data/camelyon16_features/* /project/hnguyen2/mvu9/camelyon16_features_data/ 
  