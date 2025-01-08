import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob

example_list = ['normal_072', 'normal_001', 'normal_048', 'tumor_026', 'tumor_031', 'tumor_032']
SLIDE_PATH = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16/images'
JSON_PATH = '/project/hnguyen2/mvu9/camelyon16/json_files'

class SuperpixelDataset(Dataset):
    def __init__(self, slide_root, superpixel_root, basename):
        self.slide = None 
        self.basename = os.path.basename(slide_path)
    def __getitem__(self, index):
        return None  
    
class PatchDataset(Dataset):
    def __init__(self):
        pass 
    def __getitem__(self, idx):
        pass 

if __name__ == '__main__':
    wsi_paths = glob.glob(os.path.join(SLIDE_PATH, '*.tif'))
    wsi_paths = [path for path in wsi_paths if os.path.basename(path).split(".")[0] in example_list]
    
    for wsi_path in wsi_paths: 
        print(wsi_path)
        slide = openslide.open_slide(wsi_path)
        print(slide.dimensions)
        break
        
        
        
        