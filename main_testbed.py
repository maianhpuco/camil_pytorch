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
from testbed.scoring_sampling import sampling_method_
import openslide

from data.merge_dataset import SuperpixelDataset, PatchDataset
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
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


PROJECT_DIR = os.environ.get("/Users/nam.le/Desktop/research/camil_pytorch")
sys.path.append(PROJECT_DIR)

# SLIDE_DIR = '/project/hnguyen2/hqvo3/Datasets/digital_pathology/public/CAMELYON16'
example_list = [
    "normal_072",
    "normal_001",
    "normal_048",
    "tumor_026",
    "tumor_031",
    "tumor_032",
]

model = timm.create_model(
    "vit_base_patch16_224", pretrained=True
)  # You can choose any model
model.eval()


def main(args):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize the patch to 256x256
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize with ImageNet stats
            # You can add other transformations like RandomHorizontalFlip, RandomRotation, etc.
        ]
    )
    if args.dry_run:
        print("Running the dry run")
    else:
        print("Running on full data")
    start_slide = time.time()

    wsi_paths = glob.glob(os.path.join(args.slide_path, "*.tif"))
    wsi_paths = [
        path
        for path in wsi_paths
        if os.path.basename(path).split(".")[0] in example_list
    ]
    json_folder = args.json_path

    superpixel_dataset = SuperpixelDataset(
        slide_paths=wsi_paths,
        json_folder=json_folder,
    )
    print("Number of slide in dataset:", len(superpixel_dataset))

    for slide_index in range(len(superpixel_dataset)):
        superpixel_datas, wsi_path = superpixel_dataset[slide_index]
        print(wsi_path)
        # slide = openslide.open_slide(wsi_path)
        print(len(superpixel_datas))
        superpixel_datas = superpixel_datas[:2]
        slide_basename = os.path.basename(wsi_path).split(".")[0]
        print("Basename:", slide_basename)
        save_dir = os.path.join(args.spixel_path, slide_basename)
        start_slide = time.time()

        store_array = []

        for each_superpixel in superpixel_datas:
            start_spixel = time.time()
            foreground_idx = each_superpixel["foreground_idx"]
            xywh_abs_bbox = each_superpixel["xywh_abs_bbox"]
            superpixel_extrapolated = each_superpixel["superpixel_extrapolated"]

            start_spixel = time.time()

            superpixel_np = utils.read_region_from_npy(
                args.spixel_path, slide_basename, foreground_idx
            )
            print("- Complete reading after: ", time.time() - start_spixel)

            patch_dataset = PatchDataset(
                superpixel_np,
                superpixel_extrapolated,
                patch_size=(224, 224),
                transform=transform,
                coverage_threshold=0.5,
                return_feature=True,  # Enable feature extraction
                model=model,
            )
            patch_dataloader = DataLoader(
                patch_dataset, batch_size=args.batch_size, shuffle=False
            )

            _all_features_spixel = []
            _all_idxes_spixel = []

            for batch_features, batch_patches, batch_bboxes, batch_idxes in tqdm(
                patch_dataloader
            ):
                _flatten_features = batch_features.view(-1, batch_features.shape[-1])
                _all_features_spixel.append(_flatten_features)
                _all_idxes_spixel.append(batch_idxes)

            spixel_patch_features = torch.cat(_all_features_spixel)  # of a
            # print("Tensor details:")
            # print("Shape:", spixel_patch_features.shape)
            # print("Data type:", spixel_patch_features.dtype)
            # print("Device:", spixel_patch_features.device)
            # spixel_patch_features = sampling_method_(spixel_patch_features)
            store_array.append(spixel_patch_features)
            output_tensor_check = torch.cat(store_array, dim=0)
            print("Shape:", output_tensor_check.shape)
            # Print all elements of the tensor
            print("Elements:")
            print(spixel_patch_features)
            # scoring - sampling
            # add classifier (SSL) - input spixel_patch_features
            print(
                f"Final feature shape for superpixel {foreground_idx}: {spixel_patch_features.shape})"
            )
            print(
                "Complete processing a superpixel after :", time.time() - start_spixel
            )
        output_tensor_final = torch.cat((store_array), dim=0)
        output_tensor_final = sampling_method_(output_tensor_final)
        print("Complete an Slide after: ", time.time() - start_slide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", type=bool, default=False)
    parser.add_argument("--config_file", default="ma_exp001")
    args = parser.parse_args()

    if os.path.exists(f"./testbest_config/{args.config_file}.yaml"):
        config = load_config(f"./testbest_config/{args.config_file}.yaml")
        args.use_features = config.get("use_features", True)

        args.slide_path = config.get("SLIDE_PATH")
        args.json_path = config.get("JSON_PATH")
        args.spixel_path = config.get("SPIXEL_PATH")

        args.scoring_function = SCORING_FUNCTION_MAP.get(config.get("scoring_function"))
        args.pruning_function = PRUNING_FUNCTION_MAP.get(config.get("pruning_function"))
        args.batch_size = config.get("batch_size")
        args.scoring_function("")
        args.pruning_function("")

    main(args)
