import os
import sys 
import torch
import glob
import pandas as pd
import numpy as np
import argparse



PROJECT_DIR = os.environ.get('PROJECT_DIR') 
DATA_DIR  = '/project/hnguyen2/mvu9/camelyon16_features_data/h5_files'
sys.path.append(os.path.join(PROJECT_DIR))
sys.path.append(os.path.join(DATA_DIR))
# sys.path.append(os.path.join(DATA_DIR))

sys.path.append(os.path.join(PROJECT_DIR, "utils"))
sys.path.append(os.path.join(PROJECT_DIR, "src")) 
sys.path.append(os.path.join(PROJECT_DIR, "datase_utils")) 


from src.camil import CAMIL  # Assuming CAMIL is the model class from camil.py
from data.camelyon16_dataset import CustomDataset

from camil_utils.utils import train


def parse_arguments():
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Training configuration for CAMIL model")
    parser.add_argument('--dry_run', type=bool, default=False, help="test running okay? ")    
    # Dataset and paths
    parser.add_argument('--train_or_test', type=str, choices=["train", "test"], default='train', help="training or inferencing")  
    parser.add_argument('--dataset_name', type=str, choices=["camelyon16"], default='camelyon16', help="dataset name") 
    parser.add_argument('--input_shape', type=int, default=512, help="Input feature dimension (default: 512)")
    parser.add_argument('--n_classes', type=int, default=2, help="Number of output classes (default: 2)")
    parser.add_argument('--subtyping', type=bool, default=False, help="Whether to use subtyping (default: False)")
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train (default: 10)")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate (default: 1e-3)")
    parser.add_argument('--checkpoint_filename', type=str, default=None)
    # Device (GPU/CPU)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda", help="Device for training (default: cuda)")

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    #TODO: later args should be save in a YML file ! 
    print(f"Training for dataset {format(args.dataset_name)}")
    train_or_test_or_val = 'train'
    # args.epochs = NUM_EPOCH 
    if args.dataset_name == 'camelyon16':     
        args.feature_folder =os.path.join(DATA_DIR)  
        args.label_file = os.path.join(PROJECT_DIR, "data/label_files/camelyon_data.csv")
        args.split_filepath = os.path.join(PROJECT_DIR, "data/camelyon_csv_splits/splits_3.csv")
        # args.feature_folder =os.path.join(PROJECT_DIR,'data/camelyon16_features/h5_files') 
        args.save_dir = os.path.join(PROJECT_DIR, "data/weights") 
        args.log_dir = os.path.join(PROJECT_DIR, "data/logs")   
    # Initialize the model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
 
    model = CAMIL(args)
    model.to(device)   

    import datetime
    # Generate the dynamic save path with dataset name, timestamp, and "completed"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    save_path = f"{args.save_dir}/{args.dataset_name}_{timestamp}_completed.pth" 
    log_path = f"{args.log_dir}/{args.dataset_name}_{timestamp}.json"  
    print(args)
    if args.dry_run == True: 
        args.epochs = 30 
        print("RUNING THE DRY RUN---->>> ")
        train_dataset = CustomDataset(
            train_or_test_or_val = 'train',
            split_filepath=args.split_filepath, 
            label_file=args.label_file,
            feature_folder=args.feature_folder, 
            feature_file_end ='h5',  
            shuffle=True,  
            dry_run=True

        )
            # Create dataset
        val_dataset = CustomDataset(
            train_or_test_or_val = 'val',
            split_filepath=args.split_filepath, 
            label_file=args.label_file,
            feature_folder=args.feature_folder, 
            feature_file_end ='h5',  
            shuffle=True, 
            dry_run=True
        )
    else: 
        print(f"Running the training with {args.epochs} epochs")
        # Create dataset
        train_dataset = CustomDataset(
            train_or_test_or_val = 'train',
            split_filepath=args.split_filepath, 
            label_file=args.label_file,
            feature_folder=args.feature_folder, 
            feature_file_end ='h5',  
            shuffle=True, 
        )
            # Create dataset
        val_dataset = CustomDataset(
            train_or_test_or_val = 'val',
            split_filepath=args.split_filepath, 
            label_file=args.label_file,
            feature_folder=args.feature_folder, 
            feature_file_end ='h5',  
            shuffle=True
        )

    
    # Print the length of the train and validation datasets
    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of val_dataset: {len(val_dataset)}")
    # # Set device (GPU or CPU)
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    # # Call the train function
    if args.checkpoint_filename is not None: 
        checkpoint_path = os.path.join(PROJECT_DIR, 'data/weights', args.checkpoint_filename) 
    else: 
        checkpoint_path = None
    train(
        model, 
        train_dataset,
        val_dataset, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate, 
        device=device, 
        save_path=save_path, 
        log_file=log_path, 
        checkpoint_path=checkpoint_path 
        )

    
if __name__ == '__main__':
    main()
#checkpoint_path camelyon16_20241118-1932_completed.pth
# /project/hnguyen2/mvu9/camelyon16_features_data

# cp -r /home/mvu9/digital_pathology/camil_pytorch/data/camelyon16_features/* /project/hnguyen2/mvu9/camelyon16_features_data/ 
  