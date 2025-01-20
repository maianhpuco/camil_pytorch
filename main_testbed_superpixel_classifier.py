import os
import sys 
import torch
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
import yaml 

import openslide

from data.merge_dataset import (
    SuperpixelDataset, 
    PatchDataset, 
    SuperpixelPatchesDataset)
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from patch_merging import tome 
from utils import utils  
from testbed.importance_scores import get_scoring_do_nothing
from testbed.pruning import get_pruning_do_nothing

SCORING_FUNCTION_MAP = {
    "get_scoring_do_nothing": get_scoring_do_nothing,
}


PRUNING_FUNCTION_MAP = {
    "get_pruning_do_nothing": get_pruning_do_nothing,
}   

def load_config(config_file):
    # Load configuration from the provided YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config



PROJECT_DIR = os.environ.get('PROJECT_DIR')
sys.path.append(os.path.join(PROJECT_DIR))  

# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']

model = timm.create_model('vit_base_patch16_224', pretrained=True)  # You can choose any model
model.eval()  

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the patch to 224x224
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
    ])
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data")
    start_slide = time.time()
    
    wsi_paths = glob.glob(os.path.join(args.slide_path, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    json_folder = args.json_path
    
    superpixel_dataset = SuperpixelDataset(
        slide_paths=wsi_paths,
        json_folder=json_folder,
        )
    print("Number of slide in dataset:", len(superpixel_dataset)) 
    
    _slide_features = []
    _slide_patch_idxes = []
    
    for slide_index in range(len(superpixel_dataset)):
        
        superpixel_datas, wsi_path = superpixel_dataset[slide_index]
        print(wsi_path)
        #slide = openslide.open_slide(wsi_path)  
        print(len(superpixel_datas))
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        print("Basename:", slide_basename)
        save_dir = os.path.join(args.spixel_path, slide_basename) 
        start_slide = time.time()
        total = 0 
        
        _spixel_features = []
        _spixel_patch_idxes = []
        
        for each_superpixel in tqdm(superpixel_datas):
            start_spixel = time.time()
            foreground_idx = each_superpixel['foreground_idx'] 
            xywh_abs_bbox = each_superpixel['xywh_abs_bbox']
            superpixel_extrapolated = each_superpixel['superpixel_extrapolated']

            
            # superpixel_np = utils.read_region_from_npy(
            #     args.spixel_path, 
            #     slide_basename, 
            #     foreground_idx
            #     )

            spixel_patches_dataset = SuperpixelPatchesDataset(
                patch_dir=os.path.join(args.patch_path, slide_basename),
                transform=transform,
                preferred_spixel_idx=foreground_idx
            )

            # Create DataLoader
            dataloader = DataLoader(spixel_patches_dataset, batch_size=args.batch_size, shuffle=True)
            
            _spixel_features = []
            _spixel_patch_idxes = []
            
            for batch in dataloader:
                batch_image = batch['image']
                batch_patch_info = batch['patch_info']
                
                parsed_batch_info = [] 
                for i in range(len(batch_patch_info['ymin'])):
                    parsed_info = {
                        'ymin': batch_patch_info['ymin'][i].item(),
                        'ymax': batch_patch_info['ymax'][i].item(),
                        'xmin': batch_patch_info['xmin'][i].item(),
                        'xmax': batch_patch_info['xmax'][i].item(),
                        'spixel_idx': batch_patch_info['spixel_idx'][i].item(),
                        'patch_idx': batch_patch_info['patch_idx'][i].item()
                    }
                    parsed_batch_info.append(parsed_info)
                    
                batch_idxes = [info['patch_idx'] for info in parsed_batch_info]  
                with torch.no_grad():  # Disable gradient calculation for inference
                    _batch_features = model.forward_features(batch_image)
                    class_token_features = _batch_features[:, 0, :]   
                    
                # 0. apply feature extraction here on batch_image
                    # input: batch_image
                    # output: slide_features (remember to cat them into a slide's features)
                    
                _spixel_features.append(class_token_features)
                _spixel_patch_idxes.append(batch_idxes) 
                
                # print("Parsed Batch Info of a sample:", parsed_batch_info[0])
                # print("Batch Image Shape:", batch_image.shape) 
                
            spixel_features = torch.cat(_spixel_features, dim=0)  # Concatenate all features for the slide on GPU
            spixel_patch_idxes = torch.cat([torch.tensor(idxes) for idxes in _spixel_patch_idxes], dim=0)
            _slide_features.append(spixel_features)
            _slide_patch_idxes.append(spixel_patch_idxes)
            print(f"> Finish a Superpixel after: {(time.time()-start_spixel)/60.0000} mins")
        
        slide_feature = torch.cat(_slide_features, dim=0)
        slide_patch_idxes = torch.cat([torch.tensor(idxes) for idxes in _slide_patch_idxes], dim=0)       
        print(f"--> Finish a slide after: {(time.time()-start_slide)/60.0000} mins") 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=bool, default=False)
    parser.add_argument('--config_file', default='ma_exp002')
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        
        args.slide_path = config.get('SLIDE_PATH')
        args.json_path = config.get('JSON_PATH')
        args.spixel_path = config.get('SPIXEL_PATH')
        args.patch_path = config.get('PATCH_PATH')
        
        args.scoring_function = SCORING_FUNCTION_MAP.get(
            config.get("scoring_function")
        )
        args.pruning_function = PRUNING_FUNCTION_MAP.get(
            config.get('pruning_function') 
        )
        args.batch_size = config.get('batch_size')
        args.scoring_function("")
        args.pruning_function("")
    
    main(args) 
     