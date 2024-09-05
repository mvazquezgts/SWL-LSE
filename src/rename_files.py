import pickle
import numpy as np
import os, sys
import csv
import argparse
import shutil
from tqdm import tqdm

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            parts = filename.split('_')
            if "test" in parts:
                index = parts.index("test")
            elif "val" in parts:
                index = parts.index("val")
            elif "train" in parts:
                index = parts.index("train")
            else:
                continue

            new_name = f"{parts[index]}_" + "_".join(parts[:index] + parts[index+1:])
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} to {new_path}")

def rename_files2(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            parts = filename.split('_')
            if "test" in parts:
                index = parts.index("test")
            elif "val" in parts:
                index = parts.index("val")
            elif "train" in parts:
                index = parts.index("train")
            else:
                continue

            # Construct new filename with "test_", "val_", or "train_" at the beginning
            new_name = f"{parts[index]}_" + "_".join(parts[:index] + parts[index+1:])
            
            # Remove "_data" if it is at the end
            if new_name.endswith("_data.npy"):
                new_name = new_name.replace("_data.npy", ".npy")
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} to {new_path}")

def main(args):
    folder_in = args.folder_in
    rename_files2(folder_in)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in', required=True, type=str)

    arg = parser.parse_args()
    main(arg)

    """
    python rename_files.py \
        --folder_in /home/tmpvideos/mvazquez/ISLR_bbdd/SIGNAMED_JULIO_2024/DATASET/NO_NORM/HP_TEST

    """
